import os,pickle,random,shutil,sys,stat,seaborn as sns,copy
from datetime import datetime
import os.path,pandas as pd
import numpy as np
import torch
import tensorflow as tf
from matplotlib import pyplot as plt
# from tensorboardX import SummaryWriter
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
# random.seed(142)

class Logger(object):

    def __init__(self, fn, ask=True, local_rank=0):
        self.local_rank = local_rank
        if self.local_rank == 0:
            if not os.path.exists("./logs/"):
                os.mkdir("./logs/")

            logdir = self._make_dir(fn)
            if not os.path.exists(logdir):
                os.mkdir(logdir)

            if len(os.listdir(logdir)) != 0 and ask:
                # ans = input("log_dir is not empty. All data inside log_dir will be deleted. "
                #             "Will you proceed [y/N]? ")
                ans = 'y'
                if ans in ['y', 'Y'] and os.path.exists(logdir):
                    shutil.rmtree(logdir,ignore_errors=True)
                else:
                    exit(1)

            # self.set_dir(logdir)

    def _make_dir(self, fn):
        today = datetime.today().strftime("%y%m%d")
        logdir = 'logs/' + fn
        return logdir

    # def set_dir(self, logdir, log_fn='log.txt'):
    #     self.logdir = logdir
    #     if not os.path.exists(logdir):
    #         os.mkdir(logdir)
    #     self.writer = SummaryWriter(logdir)
    #     self.log_file = open(os.path.join(logdir, log_fn), 'a')

    def log(self, string):
        if self.local_rank == 0:
            self.log_file.write('[%s] %s' % (datetime.now(), string) + '\n')
            self.log_file.flush()

            print('[%s] %s' % (datetime.now(), string))
            sys.stdout.flush()

    def log_dirname(self, string):
        if self.local_rank == 0:
            self.log_file.write('%s (%s)' % (string, self.logdir) + '\n')
            self.log_file.flush()

            print('%s (%s)' % (string, self.logdir))
            sys.stdout.flush()

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        if self.local_rank == 0:
            self.writer.add_scalar(tag, value, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""
        if self.local_rank == 0:
            self.writer.add_image(tag, images, step)

    def histo_summary(self, tag, values, step):
        """Log a histogram of the tensor of values."""
        if self.local_rank == 0:
            self.writer.add_histogram(tag, values, step, bins='auto')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.average = self.sum / self.count


def load_checkpoint(logdir, mode='last'):
    if mode == 'last':
        model_path = os.path.join(logdir, 'last.model')
        optim_path = os.path.join(logdir, 'last.optim')
        config_path = os.path.join(logdir, 'last.config')
    elif mode == 'best':
        model_path = os.path.join(logdir, 'best.model')
        optim_path = os.path.join(logdir, 'best.optim')
        config_path = os.path.join(logdir, 'best.config')

    else:
        raise NotImplementedError()

    print("=> Loading checkpoint from '{}'".format(logdir))
    if os.path.exists(model_path):
        model_state = torch.load(model_path)
        optim_state = torch.load(optim_path)
        with open(config_path, 'rb') as handle:
            cfg = pickle.load(handle)
    else:
        return None, None, None

    return model_state, optim_state, cfg


def save_checkpoint(epoch, model_state, optim_state, modeldir,modelinfo=""):
    last_config = os.path.join(modeldir, 'last_'+modelinfo+"_"+str(epoch)+'.config')
    opt = {'epoch': epoch}
    torch.save({"Model":model_state,"Optim":optim_state},os.path.join(modeldir, 'last_'+modelinfo+"_"+str(epoch)+'.pt'))
    with open(last_config, 'wb') as handle:
        pickle.dump(opt, handle, protocol=pickle.HIGHEST_PROTOCOL)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    tf.random.set_seed(142)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def normalize(x, dim=1, eps=1e-8):
    return x / (x.norm(dim=dim, keepdim=True) + eps)
    #2范数，归一化
    #

def make_model_diagrams(probs, labels, n_bins=10):
    """
    outputs - a torch tensor (size n x num_classes) with the outputs from the final linear layer
    - NOT the softmaxes
    labels - a torch tensor (size n) with the labels
    """
    confidences, predictions = probs.max(1)
    accuracies = torch.eq(predictions, labels)
    f, rel_ax = plt.subplots(1, 2, figsize=(4, 2.5))

    # Reliability diagram
    bins = torch.linspace(0, 1, n_bins + 1)
    bins[-1] = 1.0001
    width = bins[1] - bins[0]
    bin_indices = [confidences.ge(bin_lower) * confidences.lt(bin_upper) for bin_lower, bin_upper in
                   zip(bins[:-1], bins[1:])]
    bin_corrects = [torch.mean(accuracies[bin_index]) for bin_index in bin_indices]
    bin_scores = [torch.mean(confidences[bin_index]) for bin_index in bin_indices]

    confs = rel_ax.bar(bins[:-1], bin_corrects.numpy(), width=width)
    gaps = rel_ax.bar(bins[:-1], (bin_scores - bin_corrects).numpy(), bottom=bin_corrects.numpy(), color=[1, 0.7, 0.7],
                      alpha=0.5, width=width, hatch='//', edgecolor='r')
    rel_ax.plot([0, 1], [0, 1], '--', color='gray')
    rel_ax.legend([confs, gaps], ['Outputs', 'Gap'], loc='best', fontsize='small')

    # Clean up
    rel_ax.set_ylabel('Accuracy')
    rel_ax.set_xlabel('Confidence')
    f.tight_layout()
    return f


def get_label_index(dataset, L_index,target_list,labelindex=3):
    label_i_index = [[] for i in range(len(target_list))]
    for i in L_index:
        for k in range(len(target_list)):
            if dataset[i][labelindex] == target_list[k]:
                label_i_index[k].append(i)
    return label_i_index

def PlotGraph(datalist,labels,Xylabel,filename):
    plt.figure()
    xlabels = [i for i in range(1, len(datalist[0]) + 1)]
    for d,l in zip(datalist,labels):
        plt.plot(xlabels,d, label=l)
    plt.xlabel(Xylabel[0])
    plt.ylabel(Xylabel[1])
    plt.xticks(xlabels)
    plt.legend()
    plt.savefig(filename)

def PlotErrorGraph(trainaccuracylist,testaccuracylist,filename):
    plt.figure()
    x = ["Train", "Test"]
    y = [np.mean(t) for t in [trainaccuracylist, testaccuracylist]]
    e = [np.std(t) for t in [trainaccuracylist, testaccuracylist]]
    plt.errorbar(x, y, e, linestyle='None', marker='^')
    plt.legend()
    plt.savefig(filename)

def saveResult(ExpName,EpochExperiment,P,modelname,dataName,dataSelection,results,resultFolder,disableContrastive=False,disableUncertainiy=False):
    os.makedirs(resultFolder,exist_ok=True)
    countName="L" if dataSelection=="S" else "S"
    OutFile=resultFolder+"Results_ALL.csv"
    metricslist=[("TrainingAcc","TrainingAcc"), ("STestAcc","STestAcc"), ("LTestAcc","LTestAcc"), ("TrainF1","TrainF1"), ("STestF1","STestF1"), ("LTestF1","LTestF1"), ("TrainRocAuc","TrainRocAuc"),("STestRocAuc","STestRocAuc"),("LTestRocAuc","LTestRocAuc")]
    columns= "DataSelection,ContrastiveEpoches,Uncertainepoch,ContrastivConvLayers,uncertainityConvLayers,ContrastiveActivation1,ContrastiveActivation2,UncertainityActivation1,UncertainityActivation2,UncertainityLR,ContrastiveLR,ContrastiveOptimizer,UncertinityOptimizer,Contrastiveloss,SimLambda,dis_lambda,contrastive_alpha,uncertainity_beta,BatchSize,Number of cycle,Number of selected per cycle,InitialData Percentage,Seed,ExperimentName,EpochExperiment,disableContrastive,disableUncertainiy,"
    Pparmas=["DataSelection","ContrastiveEpoches", "Uncertainepoch","ContrastivConvLayers","uncertainityConvLayers","ContrastiveActivation1","ContrastiveActivation2","UncertainityActivation1","UncertainityActivation2","UncertainityLR","ContrastiveLR","ContrastiveOptimizer","UncertinityOptimizer","Contrastiveloss","SimLambda","dis_lambda","contrastive_alpha","uncertainity_beta","BatchSize","NumberOfCycles","target_Samples","traindatapercentage","seed"]
    commanresults=[P[p] for p in Pparmas]+[ExpName,EpochExperiment,str(disableContrastive),str(disableUncertainiy)]
    if not os.path.exists(OutFile):
        with open(OutFile,"w") as f:
            f.write(columns+"ResultType,Results\n")
    result=[[*commanresults,"TrainDataCount"]+[r for r in results["TrainDataCount"]],
            [*commanresults,"SCountList"]+["%.2f"%(r) for r in results["SCountList"]],
            [*commanresults,"LCountList"]+["%.2f"%(r) for r in results["LCountList"]]]
    result+=[[*commanresults,name]+["%.2f"%(r) for r in results[acc]] for acc,name in metricslist]
    with open(OutFile,"a") as f:
        for res in result:
            f.write(",".join([str(r) for r in res])+"\n")
    print("Results Added at",OutFile)

    AccNames=",".join([name for acc,name in metricslist])
    OutFile=resultFolder+"Results.csv"
    if not os.path.exists(OutFile):
        with open(OutFile,"w") as f:
            f.write(columns+"TrainInstances,"+AccNames+",Scount,Lcount\n")
    result=commanresults+[results["TrainDataCount"][-1]]+["%.2f"%(results[lss][-1]) for lss,_ in metricslist]+["%.2f"%(np.sum(results["SCountList"]))]+["%.2f"%(np.sum(results["LCountList"]))]
    with open(OutFile,"a") as f:
        f.write(",".join([str(r) for r in result])+"\n")
    print("Results Added at",OutFile)



def showResults(ExpName,EpochExperiment,P,modelname,dataName,dataSelection,results,plotGraphs=False,resultFolder="Results",disableContrastive=False,disableUncertainiy=False):
    print("Saving results at",resultFolder,"........................")
    resultFolder+="/"
    countName="L" if dataSelection=="S" else "S"
    saveResult(ExpName,EpochExperiment,P,modelname,dataName,dataSelection,results, resultFolder,disableContrastive,disableUncertainiy)
    if plotGraphs:
        os.makedirs(resultFolder+"Images",exist_ok=True)
        trainaccuracy, stestaccuracy,ltestaccuracy, S_count_list, L_count_list = [results[m] for m in ["TrainingAcc", "STestAcc", "LTestAcc", "SCountList", "LCountList"]]
        imageFolder=resultFolder+"Images/"+dataName+"_"+dataSelection+"/"
        os.makedirs(imageFolder,exist_ok=True)
        PlotGraph([trainaccuracy, stestaccuracy,ltestaccuracy], ["Train","STest","LTest"], ["Number of Al Cycle","Accuaracy(%)"], imageFolder+modelname+"_Accuracy.jpg")
        PlotGraph([S_count_list,L_count_list], ["SCount,LCount"], ["Number of Al Cycle","number of selected SL in each cycle"], imageFolder+modelname+"_"+countName+"_Count.jpg")
        # PlotErrorGraph(trainaccuracylist, testaccuracylist, imageFolder+modelname+"_Accuracy_Mean_Std.jpg")
        # PlotGraph([contrastivetrainloss], ["Train Loss"], ["Number of Iteration","Contrastive Loss"], resultFolder+"Images/"+modelname+"_Constrastive_Loss.jpg")
        # plt.show()
    stoalcount,ltotalcount=0,0
    for i,trainacc,stestacc,ltestacc,scount,lcount in zip(range(1000),results["TrainingAcc"],results["STestAcc"],results["LTestAcc"],results["SCountList"],results["LCountList"]):
        stoalcount,ltotalcount=stoalcount+scount,ltotalcount+lcount
        print(i+1,trainacc,stestacc,ltestacc,stoalcount/(P["target_Samples"]*(i+1)),stoalcount,ltotalcount/(P["target_Samples"]*(i+1)),ltotalcount,P["target_Samples"]*(i+1))

def saveModels(ExpName,contrastiveModel,UncertainModel,model_name,modeldir):
    os.makedirs(modeldir, exist_ok=True)
    # model_name+="_"+ExpName
    if contrastiveModel is not None:
        torch.save(contrastiveModel, modeldir+"Contrastive_"+model_name+".pth")
    if UncertainModel is not None:
        torch.save(UncertainModel, modeldir+"Uncertainty_"+model_name+".pth")

def PlotDistribution(Data_set,train_index,graph_name):
    def getData(Data_Set,indexes=None):
        if indexes is None:
            indexes=list(range(len(Data_set)))
        data_list=[]
        for i in indexes:
            data_list.append(Data_Set[i][0].numpy())
        return np.mean(np.rollaxis(np.array(data_list),2,1)[:,-1,:],axis=-1)

    alldata=getData(Data_set)
    traindata=getData(Data_set,train_index)
    plt.figure()
    sns.kdeplot(alldata, fill=True, label='Full Data')
    sns.kdeplot(traindata, fill=True, label='Initial 20% Data')
    plt.legend()
    os.makedirs("Distributions",exist_ok=True)
    plt.savefig("Distributions/"+graph_name+".jpg")


# Turns a dictionary into a class
class Dict2Class(object):
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])

def divideTheData(data,divisions=10):
    datavalues=[d[1] for d in data]
    bins=np.linspace(np.min(datavalues), np.max(datavalues)+.001, divisions+1)
    datalist=[[d for d in data if bins[i-1]<=d[1]<bins[i]] for i in range(1,len(bins))]
    return datalist

def getLatenHyperCubeSampels(data,datpercentage,divisions=10):
    datalist=divideTheData(data, divisions=divisions)
    dataNumberFromPercentage=[round(len(data)*datpercentage/100) for data in datalist]
    selecteddata=sum([random.sample(data,percen) for data,percen in zip(datalist,dataNumberFromPercentage) if percen>0],[])
    index,data=np.array([d[0] for d in selecteddata]),np.array([d[1] for d in selecteddata])
    return index,data

def get1DData(Data_set,index):
    # data = np.mean(np.stack([d[0] for d in Data_set])[:, :, -1],axis=1)
    # data = np.mean(np.stack([d[0] for d in Data_set]),axis=(1,2))
    data = np.stack([Data_set[i][0] for i in index])
    # data = data[:, :, -1]
    data = data.reshape((data.shape[0],-1))
    pca=PCA(n_components=1)
    data=pca.fit_transform(data)
    return data

def getTrainingIndicesByLatenHybercube(Data_set,index,datpercentage=20):
    # use Laten Hypercube to Selecti Inital data
    data=get1DData(Data_set,index)
    data=list(zip(range(len(data)),data))
    indexes,_=getLatenHyperCubeSampels(data,datpercentage=datpercentage)
    return np.array(index,dtype=np.int32)[indexes]


def getUsedSeedsinResultsfor(Numberofselected,NumberOfCycles,traindatapercentage,resultFolder):
    filePath=resultFolder+"/Results.csv"
    if not os.path.exists(filePath): return []
    data=pd.read_csv(filePath)
    data=data[data["Number of cycle"]==NumberOfCycles]
    data=data[data["Number of selected per cycle"]==Numberofselected]
    data=data[data["InitialData Percentage"]==traindatapercentage]
    seed=data["Seed"].values
    return seed


def getAllCombinations(withInitialData=True,execludeProcessed=False,filepath=None):
    combineations,combineationsrev=[],[]
    for i in range(1, int(800 ** 0.5) + 1):
        if 800 % i == 0:
            j = 800 // i
            combineations.append((i,j))
            combineationsrev.append((j,i))
    combineationsrev.reverse()
    combineations=combineations[1:]+combineationsrev[1:-1]
    if not withInitialData: return combineations
    combs=[]
    for indata in [10,15,20]:
        for com in combineations:
            combs.append((indata,com[0],com[1]))
    if execludeProcessed and filepath is not None and os.path.exists(filepath):
        filercombs=[]
        data = pd.read_csv(filepath)
        for initdata, NumberOfCycles, target_Samples in combs:
            datapart = data[data["Number of cycle"] == NumberOfCycles]
            datapart = datapart[datapart["Number of selected per cycle"] == target_Samples]
            datapart = datapart[datapart["InitialData Percentage"] == initdata]
            if datapart.shape[0]<10:
                filercombs.append((initdata, NumberOfCycles, target_Samples))
            combs=filercombs
    return combs


def divideCombinationBasedOnNumberOfCycles(combs,numberofprocess=11):
    combs = sorted(combs, key=lambda x: x[1])
    combperprocess = (sum([x[1] for x in combs]) // numberofprocess)
    combinations,currentcomb = [],[]
    for com in combs:
        currentcomb.append(com)
        if sum([x[1] for x in currentcomb]) > combperprocess:
            combinations.append(currentcomb)
            currentcomb = []
    if len(currentcomb) > 0: combinations.append(currentcomb)
    return combinations


def selectIndexices(dataset,sindex):
	dataset.traindata,dataset.trainlabels=dataset.traindata[sindex],dataset.trainlabels[sindex]
	dataset.Sl_Labels, dataset.NA_Labels=dataset.Sl_Labels[sindex], dataset.NA_Labels[sindex]
	dataset.postiveAug=dataset.postiveAug[sindex]


def getSdataOnly(DataDet,fixedIndex=None):
	dataset=makeACopy(DataDet)
	sindex=dataset.Sl_Labels==0
	selectIndexices(dataset, sindex)
	if fixedIndex is not None:
		rearanegIndex=dict(zip(np.arange(sindex.shape[0])[sindex],range(sindex.shape[0])))
		fixedIndex=list(map(lambda x:rearanegIndex[x],fixedIndex))
	return dataset,fixedIndex

def getLdataOnly(DataDet,fixedIndex=None):
	dataset=makeACopy(DataDet)
	sindex=dataset.Sl_Labels==1
	selectIndexices(dataset, sindex)
	if fixedIndex is not None:
		rearanegIndex=dict(zip(np.arange(sindex.shape[0])[sindex],range(sindex.shape[0])))
		fixedIndex=list(filter(lambda x:x <sindex.shape[0],fixedIndex)) # remove out of index indexices
		fixedIndex=list(map(lambda x:rearanegIndex[x],fixedIndex))
	return dataset,fixedIndex


def makeACopy(dataSet):
	dataset=copy.copy(dataSet)
	dataset.traindata,dataset.trainlabels,dataset.NA_Labels,dataset.Sl_Labels=[copy.deepcopy(d) for d in [dataset.traindata,dataset.trainlabels,dataset.NA_Labels,dataset.Sl_Labels]]
	return dataset


def Train_Test_Split(dataset,test_percentage=.2,shuffle=False):
	if shuffle:
		idx = np.random.permutation(dataset.traindata.shape[0])
		dataset.traindata, dataset.trainlabels = dataset.traindata[idx], dataset.trainlabels[idx]
		dataset.Sl_Labels, dataset.NA_Labels = dataset.Sl_Labels[idx], dataset.NA_Labels[idx]
	trainsize=dataset.traindata.shape[0]-int(dataset.traindata.shape[0]*test_percentage)
	testset=makeACopy(dataset)
	dataset.traindata, dataset.trainlabels = dataset.traindata[:trainsize], dataset.trainlabels[:trainsize]
	dataset.Sl_Labels, dataset.NA_Labels = dataset.Sl_Labels[:trainsize], dataset.NA_Labels[:trainsize]
	testset.traindata, testset.trainlabels = testset.traindata[trainsize:], testset.trainlabels[trainsize:]
	testset.Sl_Labels, testset.NA_Labels = testset.Sl_Labels[trainsize:], testset.NA_Labels[trainsize:]
	return dataset,testset

def splitValidationDate(Data_set,valType="L",test_percentage=0.2):
    train_data = makeACopy(Data_set)
    if valType=="L":
        _, val_index = train_test_split([d[-1] for d in train_data if d[4] == 1], test_size=test_percentage, shuffle=True)
    elif valType == "SL":
        valcount=int(len(Data_set)*test_percentage)
        val_index = random.choices([d[-1] for d in Data_set if d[4]==0],k=valcount//2)+random.choices([d[-1] for d in Data_set if d[4]==1],k=valcount//2)
    else:
        _, val_index = train_test_split([d[-1] for d in train_data if d[4] == 0], test_size=test_percentage, shuffle=True)
    train_index = list(filter(lambda x: x not in val_index, range(len(train_data))))
    val_data = makeACopy(train_data)
    selectIndexices(train_data, train_index)
    selectIndexices(val_data, val_index)
    return train_data,val_data


def SelectCorrectContrastive(args,Data_set, unlabeled_index,query_Index,PrevSlCount=None):
    if args.dataSelection=="SL":return query_Index
    if args.disableContrastive: opdatarange=(15,20)
    else: opdatarange=(6,12)
    queryS=[q for q in query_Index if Data_set[q][4]==0]
    queryL=[q for q in query_Index if Data_set[q][4]==1]
    unlabeledS=[q for q in unlabeled_index if Data_set[q][4]==0]
    unlabeledL=[q for q in unlabeled_index if Data_set[q][4]==1]
    if PrevSlCount is not None: SLPecent=[PrevSlCount[0]/np.sum(PrevSlCount)*100,PrevSlCount[1]/np.sum(PrevSlCount)*100]
    if args.dataSelection=="L":
        spercent=len(queryS)/len(query_Index)*100
        print("SSSSS",opdatarange[0],spercent,opdatarange[1])
        if opdatarange[0]<=spercent<=opdatarange[1]:return query_Index
        if spercent<opdatarange[0]:
            instancecount=len(query_Index)*opdatarange[0]//100
            if instancecount==0 and PrevSlCount is not None and not opdatarange[0]<=SLPecent[0]<=opdatarange[1]:
                instancecount=len(query_Index)//2
            return queryL[:len(query_Index)-instancecount]+random.choices(unlabeledS,k=instancecount)
        else:
            instancecount=len(query_Index)*opdatarange[1]//100
            print("SSS",instancecount)
            if instancecount==0 and PrevSlCount is not None and not opdatarange[0]<=SLPecent[0]>=opdatarange[1]:
                instancecount=instancecount//2
            print("SSS",instancecount)
            return queryL+random.choices(queryS,k=instancecount)+random.choices(unlabeledL,k=len(query_Index)-instancecount)
    elif args.dataSelection=="S":
        lpercent=len(queryL)/len(query_Index)*100
        if opdatarange[0]<=lpercent<=opdatarange[1]:return query_Index
        if lpercent<opdatarange[0]:
            instancecount=len(query_Index)*opdatarange[0]//100
            if instancecount==0 and PrevSlCount is not None and not opdatarange[0]<=SLPecent[1]<=opdatarange[1]:
                instancecount=len(query_Index)//2
            return queryS[:len(query_Index)-instancecount]+random.choices(unlabeledL,k=instancecount)
        else:
            instancecount=len(query_Index)*opdatarange[1]//100
            if instancecount==0 and PrevSlCount is not None and not opdatarange[0]<=SLPecent[1]<=opdatarange[1]:
                instancecount=instancecount//2
            return queryS+random.choices(queryL,k=instancecount)+random.choices(unlabeledS,k=len(query_Index)-instancecount)
    else:
        raise Exception("Check")
