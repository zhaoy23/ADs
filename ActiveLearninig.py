import datetime
import os.path,pandas as pd,sys,argparse
import time

from utils.utils import save_checkpoint,Logger,getUsedSeedsinResultsfor,getAllCombinations,saveModels
import torch,numpy as np,random
import torch.utils.data as data
from torch import optim,nn
from training.joineval import sampleSelection
from models.CustomModels import CustomModel
from utils.utils import get_label_index,Dict2Class,set_random_seed,showResults,getTrainingIndicesByLatenHybercube,divideCombinationBasedOnNumberOfCycles
from getitem import Megas
from training.utils import getOptimizer
from training.JoinTraining import TrainModel,EvaluateUncertainModel


norenew=True
Normalize=True
resultFolder="Results"
contrastivelossname="Cosine" #"Cosine" "PaireWise"
contrastiveoptimizername = 'sgd'
conepoch,uncertainepoch=2,200
traindatapercentage=20
batch_size=1024
test_batch_size=1024
NumberOfCycles = 10  # query times
target_Samples=80
contrastive_alpha=1
uncertainity_beta=1
uncertainity_learning_rate=0.002
contrastibe_learning_rate=0.1
seedValues=3
sl_target_list=[0]
plotGraphs=True
basepepoces=10

# conepoch,uncertainepoch,NumberOfCycles,basepepoces,resultFolder=1,1,1,1,"ResultsTODEL"

if not norenew:
    resultFolder+="renew"

modelDir="ModelWeights/"
logger = Logger("Join_Stratigy_logs_" + str(target_Samples) + "_" + str(traindatapercentage), ask=True)
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

criterion = nn.CrossEntropyLoss().to(device)


def Training(Data_set,uncertainty_train_loader,contrastive_train_loder, conepoch,uncertainepoch,contrastiveModel=None, uncertainityModel=None,cycle=0,uncertain_train=True,contrastive_train=True):
    # This method do the training only we pass the uncertain_train and contrastive_train True or false if want to train or not.
    if contrastive_train:
        if contrastiveModel is None:
            contrastiveModel = CustomModel(num_classes=len(Data_set.SL_uniqueLabels)).to(device)  # create a new model for each NumberOfCycles so that it can have random weights
        contrastiveoptimizer, scheduler_warmup = getOptimizer(contrastiveModel, contrastiveoptimizername,contrastibe_learning_rate)
    else:
        contrastiveModel,contrastiveoptimizer,scheduler_warmup=None,None,None
    if uncertain_train:
        if uncertainityModel is None:
            uncertainityModel = CustomModel(num_classes=len(Data_set.NA_uniqueLabels)).to(device)
        uncertainityoptimizer = optim.Adam(uncertainityModel.parameters(), lr=uncertainity_learning_rate)
    else:
        uncertainityModel,uncertainityoptimizer=None,None
    results = TrainModel(contrastiveModel, contrastiveoptimizer, scheduler_warmup, contrastivelossname,
                         uncertainityModel, uncertainityoptimizer, criterion, uncertainty_train_loader,contrastive_train_loder, conepoch,
                         uncertainepoch, cycle, logger, device)
    print("------------------Join Training Complete------------------------")
    return contrastiveModel, uncertainityModel, results

# define the Data loader for testing data
test_Data_set=Megas("../data/",window_size=44,mode="eval",Normalize=Normalize)
test_loader = data.DataLoader(test_Data_set, batch_size=batch_size)
def makeIndexFile(disableContrastive):
    if disableContrastive:
        IndexFolder="Indexices/Noncontrastive"
    else:
        IndexFolder="Indexices/Contrastive"
    os.makedirs(resultFolder+"/"+IndexFolder,exist_ok=True)
    indexfile=resultFolder+"/"+IndexFolder + "/Index_"+datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S")+".csv"
    if not os.path.exists(indexfile):
        with open(indexfile,"w") as f:
            f.write("TotalCycle,Data/Cycle,TrainPercent,CycleIndex,TrainAcc,TestAcc,Lcount,Indexics\n")
    return indexfile


def ProcessData(target_Samples,NumberOfCycles,traindatapercentage,basepepoces,conepoch,uncertainepoch,disableContrastive=False,disableUncertainiy=False,gpu=False):
    # This is the main Function Control everything
    indexfile=makeIndexFile(disableContrastive)
    args = Dict2Class(
        {"load_path": "logs/features/", "ood_samples": 2, "dataset": "Megas", "K_shift": 1, "k": 100.0, "t": 0.9,
         "target_list": sl_target_list, "target": target_Samples, "contrastive_alpha": contrastive_alpha,
         "uncertainity_beta": uncertainity_beta})
    myresultFolder=resultFolder
    if disableUncertainiy: myresultFolder = resultFolder+"_WithoutUncertianty"
    if disableContrastive: myresultFolder = resultFolder+"_WithoutContrastive"

    for i in range(0,basepepoces):
        # check and select that is not in result file for current combination
        usedSeeds = getUsedSeedsinResultsfor(target_Samples,NumberOfCycles,traindatapercentage,myresultFolder)
        seedValue = random.randint(1, 5000)
        while seedValue in usedSeeds: seedValue = random.randint(1, 50000)
        print("Using Seed", seedValue,"NumberOfCycles",NumberOfCycles,"target_Samples", target_Samples, "traindatapercentage", traindatapercentage)
        modelname="Joint_" + str(seedValue) +"_"+str(NumberOfCycles)+"_"+ str(target_Samples) + "_" + str(traindatapercentage)
        if disableUncertainiy: modelname+="_WithoutUncertianty"
        if disableContrastive: modelname+="_WithoutContrastive"
        #  set seed
        set_random_seed(seedValue)
        Data_set=Megas("../data/",window_size=44,Normalize=Normalize)
        initial_training_budget=int(len(Data_set)*traindatapercentage/100)
        print("Initial_training_budget",initial_training_budget,"Initial_test_budget",len(test_Data_set))
        print("Number of Classes ",len(Data_set.alllabels),"Contrastive Class",len(Data_set.SL_uniqueLabels),"Uncertainity Class",len(Data_set.NA_uniqueLabels))

        # ----------------set active learning dataset----------------------------------------

        # select Indexies based on latenHybercube
        train_indexes=list(getTrainingIndicesByLatenHybercube(Data_set,traindatapercentage))
        print(len(train_indexes), "Samples Selected from ",len(Data_set)," From which have",np.unique([Data_set[i][4] for i in train_indexes], return_counts=True),np.unique([d[4] for d in Data_set], return_counts=True))
        # if plotGraphs: PlotDistribution(Data_set, train_indexes, graph_name=modelname + "_") # uncomment it if you want to plot distibution
        # get the Indexies of  Negative and postive Data Samples so that we can use postive data fro train and negative in contrastive learning as negative sample
        negativeIndex=[indx for indx in train_indexes if Data_set[indx][4] not in sl_target_list]
        train_indexes=[indx for indx in train_indexes if Data_set[indx][4] in sl_target_list]
        train_loader=data.DataLoader(Data_set, batch_size=batch_size, sampler=train_indexes)
        Data_set.setNegativeIndex(negativeIndex)

        unlabeled_index=[i for i in range(len(Data_set)) if i not in train_indexes]

        print("Train Have ", len(train_indexes), "Samples")
        print("Test Have ", len(test_Data_set), "Samples")
        print("Unlabeled Have ",len(unlabeled_index),"Samples")

        trainaccuracylist, testaccuracylist,l_count_list,contrastivetrainlosslist,contrastivetestlosslist,uncertaintrainlosslist,uncertaintestlosslist=[],[],[],[],[],[],[]
        contrastiveModel, uncertainityModel=None,None
        # ----------------Start active Cycles----------------------------------------
        for i in range(NumberOfCycles):
            print("Train Have ", len(train_indexes), "Samples")
            print("UnLabelIndex have ",np.unique([Data_set[i][4] for i in unlabeled_index],return_counts=True),"total",len(unlabeled_index))
            set_random_seed(seedValue)
            # Do the Training id the norenew is true the it will train for 1st cycle only
            uncertain_train,contrastive_train= disableUncertainiy is False,disableContrastive is False #and (contrastiveModel is None or norenew is False)
            newcontrastiveModel, uncertainityModel, results=Training(Data_set,train_loader,train_loader,conepoch,uncertainepoch,contrastiveModel, uncertainityModel,cycle=i,contrastive_train=contrastive_train,uncertain_train=uncertain_train)
            if results[0] is not None: contrastivetrainlosslist.append(results[0])
            if results[1] is not None: uncertaintrainlosslist.append(results[1])
            if newcontrastiveModel is not None:
                contrastiveModel=newcontrastiveModel
            with torch.no_grad():
                # Select the sample based on contrastive and uncertainty approch
                query_L,_ = sampleSelection(args, Data_set, unlabeled_index, train_indexes,  test_batch_size, contrastiveModel, uncertainityModel,disableContrastive,disableUncertainiy)

            # add the selected Indexies to training data
            train_indexes += query_L
            negativeIndex+=query_L
            if norenew is False: # for  renewing we will not exclude l samples.
                train_indexes = [indx for indx in train_indexes if Data_set[indx][4] in sl_target_list]
                negativeIndex = [indx for indx in negativeIndex if Data_set[indx][4] not in sl_target_list]
            Data_set.setNegativeIndex(negativeIndex)
            unlabeled_index = list(np.setdiff1d(list(unlabeled_index), list(train_indexes)))
            print("Selected Indexices are ",query_L)
            print("Number of Selected samples:{}, ".format(len(query_L)))
            print("Number of unlabeled samples:", len(unlabeled_index))

            # --- Evalaution of UncertainModel ---
            if disableUncertainiy is False:
                train_acc,test_acc=EvaluateUncertainModel(uncertainityModel, train_loader,test_loader,device)
                trainaccuracylist.append(train_acc)
                testaccuracylist.append(test_acc)
            else:
                trainaccuracylist.append(0)
                testaccuracylist.append(0)
            l_count_list.append(len([i for i in query_L if Data_set[i][4]==1]))
            print("Train Acc:", trainaccuracylist[-1], "Testacc:", testaccuracylist[-1]," Lcount: ",l_count_list[-1])

            # build the loadder for selected data
            sampler_labeled = data.sampler.SubsetRandomSampler(train_indexes)  # make indices initial to the samples
            train_loader = data.DataLoader(Data_set, sampler=sampler_labeled,batch_size=batch_size)
            with open(indexfile,"a") as f:
                f.write(f"{NumberOfCycles},{target_Samples},{traindatapercentage},{i+1},{trainaccuracylist[-1]},{testaccuracylist[-1]},{l_count_list[-1]},{','.join([str(t) for t in train_indexes]) }\n")
            print("############################################")
            print(f"{NumberOfCycles},{target_Samples},{traindatapercentage},{i+1},{trainaccuracylist[-1]},{testaccuracylist[-1]},{l_count_list[-1]}")
            time.sleep(5)
        # Save results and model in file
        results={"trainaccuracylist":trainaccuracylist, "testaccuracylist":testaccuracylist,"l_count_list":l_count_list,
                 "contrastivetrainlosslist":np.array(contrastivetrainlosslist),"uncertaintrainlosslist":np.array(uncertaintrainlosslist)}
                 # ,"contrastivetestlosslist":np.array(contrastivetestlosslist),"uncertaintestlosslist":np.array(uncertaintestlosslist),}
        P={"seed":seedValue,"target_Samples":target_Samples,"NumberOfCycles":NumberOfCycles,"traindatapercentage":traindatapercentage}
        saveModels(contrastiveModel, uncertainityModel, modelname, modelDir)
        showResults(P,modelname,results,plotGraphs,resultFolder=myresultFolder,disableContrastive=disableContrastive,disableUncertainiy=disableUncertainiy)
    print("indexfile",indexfile)
    return indexfile

def getArguments(args):
    # Parse the command line arguments
    sys.argv.extend(args)
    parser = argparse.ArgumentParser(
                        prog='Query Sampling',
                        description='It Do the Active Learning based on contrastive and uncertianity')

    parser.add_argument('-id', '--initdata',default=[20],nargs="+",type=int,help="Percentage of Inital Data")
    parser.add_argument('-cy', '--cycle',default=[8],nargs="+",type=int,help="Number Of Cycles")
    parser.add_argument('-tp', '--totalpoints',default=400,type=int,help="Number Of Total Point to select")
    parser.add_argument('-dc', '--disableContrastive', action="store_true",help="Whether You want to disable the contrastivelearning for experiment")
    parser.add_argument('-ep', '--baseepoches',default=1,type=int,help="Number Of Base Epoces with different seeds")
    parser.add_argument('-ce', '--contrastiveEpoches',default=conepoch,type=int,help="Number Of Epoces For Contrastive Lerarning")
    parser.add_argument('-ue', '--uncertainityEpoches',default=uncertainepoch,type=int,help="Number Of Epoces For Uncertainity Lerarning")
    args = parser.parse_args()
    print(args)
    initdatalist,cyclelist, totalpoints, baseepoches,contrastiveEpochesin,uncertainityEpoches,disableContrastive=args.initdata,args.cycle,args.totalpoints,args.baseepoches,args.contrastiveEpoches,args.uncertainityEpoches,args.disableContrastive
    assert all([totalpoints%cycle==0 for cycle in cyclelist])," Cycle Should be divisible by total point"
    selectPointlist=[totalpoints//cycle for cycle in cyclelist]
    return initdatalist,cyclelist, selectPointlist, baseepoches,contrastiveEpochesin,uncertainityEpoches,disableContrastive


def DoActiveLearning(args=[]):
    initdatalist,cyclelist, selectPointlist, baseepoches,contrastiveEpochesin,uncertainityEpoches,disableContrastive=getArguments(args)
    indexfilelist=[]
    for initdata in initdatalist:
        for cycle in cyclelist:
            for tsample in selectPointlist:
                indexfile=ProcessData(tsample,cycle,initdata,baseepoches,contrastiveEpochesin,uncertainityEpoches,disableContrastive)
                indexfilelist.append(indexfile)
    # print(indexfilelist)
    return indexfilelist[0]

if __name__ == '__main__':

    indexfile=DoActiveLearning()
    print("Index file",indexfile)

# export CUDA_VISIBLE_DEVICES=1