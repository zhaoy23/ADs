import os,torch,datetime
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from tqdm import tqdm
import numpy as np
from utils.utils import set_random_seed, normalize,AverageMeter
from scipy.stats import entropy

error_k = torch.topk

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# hflip = TL.HorizontalFlipLayer().to(device)



def Select_selector(select_indices, score,args):
    # this select best samples based on the scores
    # score = torch.tensor(score)
    finally_selector, query_inside = torch.topk(score,args.target)
    query_inside = query_inside.cpu().data
    finally_indices = np.asarray(select_indices)[query_inside]

    return finally_indices, finally_selector





def get_scores_similarity(P, feats_dict,labels):
    """
    :param P:it contains axis attribute contain mean on sample axis of feature data from a class for which we want simlarity score
    :param feats_dict: contain features of unlabeld data
    :param labels: contain labeles of each element in train data for which feature is passed in P.axis
    :return: similarty score and label of that most similar sample from traning data using indexies
    """
    # convert to gpu tensor
    feats_sim = feats_dict['simclr'].to(device)#[1600, 40, 128]
    # if "shift" in feats_dict: feats_shi = feats_dict['shift'].to(device)#[1600, 40, 4]
    N = feats_sim.size(0)

    # compute scores
    maxs = []
    labels_similarity = []

    for i in range(len(feats_sim)):
        # terverse the unlabled features 1 by 1
        # calculate the mean at 1 dimention and normalize that
        f_sim = [normalize(f.mean(dim=0, keepdim=True), dim=1) for f in feats_sim[i].chunk(P.K_shift)]  # list of (1, d)

        max_simi = 0
        # Calculate the Similarity score between training feature and Unlabled Feature All saved max_smi
        value_sim, indices_sim = ((f_sim[0] * P.axis[0]).sum(dim=1)).sort(descending=True)
        max_simi += value_sim.max().item() * P.weight_sim[0]
        labels_similarity.append(labels[indices_sim[0].item()])
        maxs.append(max_simi)
    maxs = torch.tensor(maxs)
    labels_similarity = torch.tensor(labels_similarity)

    assert maxs.dim() == 1 and maxs.size(0) == N  # (N)
    return maxs.cpu(),labels_similarity.cpu()


def get_features(P, Data_set,indecies,batch_size,model_similarity=None, sample_num=1, layers=('simclr', 'shift')):

    if not isinstance(layers, (list, tuple)): layers = [layers]
    labels,feats_dict_similarity,index = _get_features(P,Data_set, indecies,batch_size, sample_num,model_similarity, layers=layers)
    return labels,feats_dict_similarity,index


def _get_features(P, Data_set, indecies,batch_size, sample_num=1,model_similarity=None, layers=('simclr', 'shift')):
    # return the calculated feature on Augmented data of  similarity models
    # retun shape is feats_all_distinctive,labels,feats_all_similarity,index and
    # feats_all_distinctive and feats_all_similarity shape is (N,sample_num,featureOutputShape) N is Number DataSamples in indecies
    if not isinstance(layers, (list, tuple)):
        layers = [layers]

    # compute features in full dataset
    labels,index = [],[]
    if model_similarity is not None: model_similarity.eval()
    feats_all_similarity = {layer: [] for layer in layers}
    for i in tqdm(range(0,len(indecies),batch_size)):
        # compute features in one batch
        feats_batch_similarity = {layer: [] for layer in layers}
        for seed in range(sample_num):
            # get different Augmentation for each sample
            set_random_seed(seed)
            x,x_t,label,indexes=Data_set.__getitem__(indecies[i:i+batch_size],mode="eval",s=seed)
            x_t=x_t.to(device)
            # compute  features from augmented input data
            with torch.no_grad():
                kwargs = {layer: True for layer in layers}  # only forward selected layers
                if model_similarity is not None:
                    _, output_aux_similarity = model_similarity(x_t, **kwargs)
            # Save the feature for different layers we are using the 1 layer only
            for layer in layers:
                if model_similarity is not None:
                    feats_similarity = output_aux_similarity[layer].cpu()
                    feats_batch_similarity[layer] += feats_similarity.chunk(P.K_shift)

        labels.extend(label)
        index.extend(indexes)

        # concatenate features in one batch
        if model_similarity is not None:
            for key, val in feats_batch_similarity.items():
                feats_batch_similarity[key] = torch.stack(val, dim=1)  # (B, T, d)
        # add features in full dataset
        for layer in layers:
            if model_similarity is not None: feats_all_similarity[layer] += [feats_batch_similarity[layer]]

    # concatenate features in full dataset
    if model_similarity is not None:
        for key, val in feats_all_similarity.items():
            feats_all_similarity[key] = torch.cat(val, dim=0)  # (N, T, d)

    return np.array(labels),feats_all_similarity,index





def getSimilarityScore(P, Data_set, unlabeled_index, train_index,  batch_size, model_similarity):
    # Calculate Similarity Score for the Unlabeled Data

    print("Size of unlabeled Data",len(unlabeled_index),"training Data",len(train_index))

    base_path = os.path.split(P.load_path)[0]  # checkpoint directory
    os.makedirs(base_path,exist_ok=True) # basepath to save features


    kwargs = {
        'sample_num': P.ood_samples,  # 2 because we have 2 possible augmentation
        'layers': ['simclr'] #, 'shift']
    }
    feautrename="similarity"

    print("----------------Get unlabeled data's "+feautrename+" feature-------------")
    # get feautre for unlabled data these feaures contain the feaures of different augmentation and shape will be [Number of sample,OOD_samples,output shape of layer]
    labels, feats_u_similarity, index = get_features(P, Data_set,  unlabeled_index,batch_size, model_similarity, **kwargs)  # (N, T, d)
    # feaure shape is [Number of sample,ood_samples,output feature shape]

    #  get feature for training Data
    print("------------------Get labeled data's "+feautrename+" feature-------------")
    label_l, feats_l_similarity, index_l = get_features(P,Data_set,train_index,batch_size, model_similarity, **kwargs)  # (M, T, d)

    start = datetime.datetime.now()

    # Compare the Labled and unlabeld Data to get Most Similar data from Unlabled Data
    if model_similarity is not None:
        P.axis = []
        #  make the list of normalized mean of a features of simclr of labeld data used to calculate score
        for f in feats_l_similarity['simclr'].chunk(P.K_shift, dim=1):
            P.axis.append(normalize(f.mean(dim=1), dim=1).to(device))

        P.weight_sim = [1, 0, 0, 0]
        # weight_sim formula is   1/mean(normalize(mean(simclr)))
        print(f'weight_sim:\t' + '\t'.join(map('{:.4f}'.format, P.weight_sim)))

        print('Pre-compute features...')
        #  calculate the sementic score for unlabeld data with the features caclulated with train data
        max_similarity,labels_similarity = get_scores_similarity(P, feats_u_similarity, label_l)
        # normalize the sementic score
        max_similarity_score = (max_similarity - max_similarity.min()) / (max_similarity.max() - max_similarity.min())
        # max_similarity contain similarty score and label of that most similar sample from training data using indecies for eacj instance of unlabel data

    # labels_similarity = torch.tensor(labels_similarity)
    # In this we perform some exponation based normalizarion
    similarity_score = (1 - np.exp(-P.k * (max_similarity_score - P.t))) / (1 + np.exp(-P.k * (max_similarity_score - P.t)))
    # set value for nan
    similarity_score[torch.isnan(similarity_score)]=1e-6
    end = datetime.datetime.now()
    print("Time of calculate Similarity score:" + str((end - start).seconds) + "seconds")
    #Normalize Similarity score
    similarity_score=(similarity_score-similarity_score.min())/(similarity_score.max()-similarity_score.min())
    return labels_similarity, similarity_score, index, labels


def getUncertaintyScore(model, Data_set,loader_indecies, device, n,batch_size):
    # Calculate the Uncertainty score for unlabeled data with Probability given by model
    # It return Uncertainty max for more uncertain sample
    # build the loder
    loader = data.DataLoader(Data_set, sampler=loader_indecies,batch_size=batch_size)
    predicitons, indexes = [], []
    model.eval()
    with torch.no_grad():
        # do prediction on all data point and save the prediciton probability
        for d,_,_, _,_,_, i in loader:
            predicitons.append(model(d.to(device), sigmoid=True).cpu())
            indexes.append(i)
    # concat the data as it is list of batches
    predicitons, indexes = torch.cat(predicitons), torch.cat(indexes)
    if predicitons.shape[1] == 1:
        # if output have 1 element (mostaly happen for binary classification) convert it into probability
        label0 = torch.ones_like(predicitons) - predicitons
        predicitons = torch.cat([label0, predicitons], dim=1)
    # calculate the entropy score
    uncertaintyScore = entropy(predicitons.T)
    # print(predicitons.shape,uncertaintyScore.shape,len(loader_indecies))
    # for i,p,u in zip(indexes,predicitons,uncertaintyScore):
    #     if u>0.2:
    #         print(p,u,Data_set[i][4])
    # exit()
    # normalize that entropy score
    uncertaintyScore=(uncertaintyScore-uncertaintyScore.min())/(uncertaintyScore.max()-uncertaintyScore.min())
    return indexes,torch.Tensor(uncertaintyScore)


def sampleSelection(P, Data_set, unlabeled_index, train_index,  batch_size, model_similarity, uncertainityModel,disableContrastive=False,disableUncertainiy=False):
    """
    This Function select the samples based on ubcertain and contrastive method if any one is disable another will be used only
    :param P:
    :param unlabeled_indecies: Unlabaled Data can be any class
    :param train_indecies:  number of target budget from target list
    :param label_i_indecies: list of index of target list classes
    :param model_similarity: similarity model
    :return: query index is most similar target(passed in P) number of sample for each class and their corosponding label
    """
    assert disableUncertainiy is False or disableContrastive is False, "One of method must use"
    # calculate uncertainity score for each unlabeled data point
    if disableUncertainiy is False:
        index,uncertaintyScore=getUncertaintyScore(uncertainityModel,Data_set,unlabeled_index, device, P.target,batch_size)
    else:
        uncertaintyScore=torch.ones(len(unlabeled_index))

    # calculate Contrastive score for each unlabeled data point
    if disableContrastive is False:
        _,score_contrastive,index,_=getSimilarityScore(P, Data_set, unlabeled_index, train_index,  batch_size, model_similarity)
    else:
        score_contrastive=torch.ones(len(unlabeled_index))

    # set the contrative score for best 25% data and set 0 for rest
    if disableContrastive is False:
        args=torch.argsort(score_contrastive,descending=True)
        score_contrastive[args[:args.shape[0]//4]]=1
        score_contrastive[args[args.shape[0]//4:]]=0
    if disableContrastive is False and disableUncertainiy is False:
        # Join the both scores
        join_score = (P.contrastive_alpha*score_contrastive)*(P.uncertainity_beta*(uncertaintyScore))
    elif disableUncertainiy is False:
        join_score = uncertaintyScore
    else:
        join_score = score_contrastive
    # select the Sample with best joint score
    query_index_i,  query_sampleSelectionscore_i= Select_selector(index,  join_score,  P)
    query_index = list(query_index_i)
    predictedlabel={0:query_index_i}
    return query_index,predictedlabel


