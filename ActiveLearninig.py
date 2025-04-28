import datetime,warnings
import os.path, sys,argparse
import time

from utils.utils import Logger,getUsedSeedsinResultsfor, saveModels
import torch,numpy as np,random
import torch.utils.data as data
from torch import nn
from training.joineval import sampleSelection
from models.CustomModels import CustomModel
from utils.utils import (Dict2Class,set_random_seed,showResults,getTrainingIndicesByLatenHybercube,
                         getLdataOnly,getSdataOnly,splitValidationDate,SelectCorrectContrastive)
from Data.getitem import Megas
from Data.SimulationDataset import DistDataSet
from training.utils import getOptimizer
from training.JoinTraining import TrainModel,EvaluateUncertainModel
from WKA.DataPattrens import getDataDetails
warnings.filterwarnings('ignore')


# DatasetName="Random"
DatasetName="MegasFiles"
norenew=True
Normalize=True
# resultFolder="Results/Results_Normal_Laplace"
resultFolder="Results/Results_Megas"
contrastivelossname="Cosine" #Cosine Euclidean  PairWise"
contrastiveoptimizername = 'sgd'
uncertinityoptimizername = 'adam'
conepoch,uncertainepoch=200,200
traindatapercentage=20
batch_size=64
test_batch_size=1024
NumberOfCycles = 5  # query times
target_Samples=80
contrastive_alpha=1
uncertainity_beta=1
contrastive_learning_rate=0.001
uncertainity_learning_rate=0.001
seedValues=3
plotGraphs=True
basepepoces=1
contrastivconvlayers=1
uncertainityconvlayers=2
sim_lambda ,dis_lambda= 1.0,1.0
contrastiveactivation1="tanh"
contrastiveactivation2="relu6"
uncertainityactivation1="relu"
unceratinityactivation2="relu"
disableContrastive=False

# conepoch,uncertainepoch,NumberOfCycles,basepepoces,resultFolder=1,1,1,1,"ResultsTODEL"
if not norenew:
    resultFolder+="renew"

modelDir="ModelWeights/"
# logger = Logger("Join_Stratigy_logs_" + str(target_Samples) + "_" + str(traindatapercentage), ask=True)
logger = None
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss().to(device)


def Training(args,seed,Data_set,uncertainty_train_loader,contrastive_train_loder,val_loader, conepoch,uncertainepoch,contrastiveModel=None, uncertainityModel=None,cycle=0,uncertain_train=True,contrastive_train=True):
    # This method do the training only we pass the uncertain_train and contrastive_train True or false if want to train or not.
    if contrastive_train:
        if contrastiveModel is None:
            set_random_seed(seed)
            contrastiveModel = CustomModel(num_classes=len(Data_set.SL_uniqueLabels),nconv=args.contrastivconvlayers,activatation1=contrastiveactivation1,activatation2=contrastiveactivation2).to(device)  # create a new model for each NumberOfCycles so that it can have random weights
        contrastiveoptimizer, scheduler_warmup = getOptimizer(contrastiveModel, args.contrastiveoptimizer,args.contrastivelr,addschedular=True)
    else:
        contrastiveModel,contrastiveoptimizer,scheduler_warmup=None,None,None
    if uncertain_train:
        if uncertainityModel is None:
            set_random_seed(seed)
            uncertainityModel = CustomModel(num_classes=len(Data_set.NA_uniqueLabels),nconv=args.uncertainityconvlayers,activatation1=uncertainityactivation1,activatation2=unceratinityactivation2).to(device)
        uncertainityoptimizer, _ = getOptimizer(uncertainityModel, args.uncertainityoptimizer,args.uncertainitylr,addschedular=False)
    else:
        uncertainityModel,uncertainityoptimizer=None,None
    results = TrainModel(contrastiveModel, contrastiveoptimizer, scheduler_warmup, args.contrastivloss,uncertainityModel, uncertainityoptimizer, criterion, uncertainty_train_loader,
                         contrastive_train_loder,val_loader, conepoch,uncertainepoch, cycle, logger, device,sim_lambda = args.sim_lambda,dis_lambda=args.dis_lambda,modelbasepath="logs/models/"+args.modelname+"/",savemodel=True)
    print("------------------Join Training Complete------------------------")
    return contrastiveModel, uncertainityModel, results

def makeIndexFile(args,ExpName,disableContrastive,resultFolder=resultFolder):
    # THis Method write the Index of data being used in different Cycles
    if disableContrastive:
        IndexFolder="Indexices/Noncontrastive"
    else:
        IndexFolder="Indexices/Contrastive"
    os.makedirs(resultFolder+"/"+IndexFolder,exist_ok=True)
    indexfile=resultFolder+"/"+IndexFolder + "/Index_"+ExpName+".csv"
    if not os.path.exists(indexfile):
        with open(indexfile,"w") as f:
            f.write("EpochExp,TotalCycle,Data/Cycle,TrainPercent,CycleIndex,TrainAcc,STestAcc,LTestAcc,SCount,LCount,Indexics\n")
    return indexfile


def selectInitalData(args,Data_set):
    # This method select the Inital Data to used in First Cycle
    # Data selection S where we only want to select S as Intial Data L where we only want to L as inital data and SL where we want to selct both S and L
    if args.dataSelection == "S":
        SN_indexes = list(getTrainingIndicesByLatenHybercube(Data_set, [d[-1] for d in Data_set if d[3] == 0], traindatapercentage))
        SAN_indexes = list(getTrainingIndicesByLatenHybercube(Data_set, [d[-1] for d in Data_set if d[3] == 1], traindatapercentage))
        train_indexes = SN_indexes + SAN_indexes
        if 2 in Data_set.alllabels:
            LN_indexes = list(getTrainingIndicesByLatenHybercube(Data_set, [d[-1] for d in Data_set if d[3] == 2],traindatapercentage))
            LAN_indexes = list(getTrainingIndicesByLatenHybercube(Data_set, [d[-1] for d in Data_set if d[3] == 3],traindatapercentage))
            NegtiveIndex = LN_indexes + LAN_indexes
        else:
            NegtiveIndex = list(getTrainingIndicesByLatenHybercube(Data_set, [d[-1] for d in Data_set if d[3] == 3],traindatapercentage))
    elif args.dataSelection == "L":
        LN_indexes = list(getTrainingIndicesByLatenHybercube(Data_set, [d[-1] for d in Data_set if d[3] == 2], traindatapercentage))
        LAN_indexes = list(getTrainingIndicesByLatenHybercube(Data_set, [d[-1] for d in Data_set if d[3] == 3], traindatapercentage))
        train_indexes = LN_indexes + LAN_indexes
        if 2 in Data_set.alllabels:
            SN_indexes = list(getTrainingIndicesByLatenHybercube(Data_set, [d[-1] for d in Data_set if d[3] == 0],traindatapercentage))
            SAN_indexes = list(getTrainingIndicesByLatenHybercube(Data_set, [d[-1] for d in Data_set if d[3] == 1],traindatapercentage))
            NegtiveIndex = SN_indexes + SAN_indexes
        else:
            NegtiveIndex = list(getTrainingIndicesByLatenHybercube(Data_set, [d[-1] for d in Data_set if d[3] == 0],traindatapercentage))
    elif args.dataSelection == "SL":
        SN_indexes = list(getTrainingIndicesByLatenHybercube(Data_set, [d[-1] for d in Data_set if d[3] == 0], traindatapercentage))
        SAN_indexes = list(getTrainingIndicesByLatenHybercube(Data_set, [d[-1] for d in Data_set if d[3] == 1], traindatapercentage))
        LN_indexes = list(getTrainingIndicesByLatenHybercube(Data_set, [d[-1] for d in Data_set if d[3] == 2], traindatapercentage))
        LAN_indexes = list(getTrainingIndicesByLatenHybercube(Data_set, [d[-1] for d in Data_set if d[3] == 3], traindatapercentage))
        train_indexes = SN_indexes + SAN_indexes+LN_indexes+LAN_indexes
        NegtiveIndex=None

    random.shuffle(train_indexes)
    if NegtiveIndex is not None:random.shuffle(NegtiveIndex)
    return train_indexes,NegtiveIndex


def ProcessData(ExpName,args,target_Samples,NumberOfCycles,traindatapercentage,resultFolder=resultFolder):
    # This is the main Function Control everything
    # define the Data loader for testing data
    set_random_seed(args.seed)
    sl_target_list=[1] if args.dataSelection=="S" else [0]
    if args.DatasetName=="Random":
        # In case of Simulation data. Generate data from distribitions. There parameter written in getDataDetails method
        distargs, otherdataargs, _, _ = getDataDetails(args.DataIndex)
        lpercent, normalize, datasize, seed, LNASLA = otherdataargs
        Data_set = DistDataSet(mode="Ads",requiredSize=datasize, train=True, distargs=distargs, lpercent=lpercent,normalize=normalize, seed=seed,LNASLA=LNASLA)
        test_Data_set = DistDataSet(mode="Ads",requiredSize=datasize, train=False, distargs=distargs,lpercent=lpercent, normalizer=Data_set.normalizer,normalize=normalize, seed=seed)
        if "RandomAds" not in  args.DataName:
            args.DataName="RandomAds_"+args.DataName
    else:
        # In case of Megad data. Read from FIles
        args.DataName="MegasFiles"
        Data_set=Megas("../DATA/data/",args.DataIndex,dataSelection=args.dataSelection,window_size=44,Normalize=Normalize)
        test_Data_set = Megas("../DATA/data/",args.DataIndex,dataSelection=args.dataSelection, window_size=44, mode="eval", Normalize=Normalize)

    train_data, val_data = splitValidationDate(Data_set, valType=args.dataSelection)
    s_test_data,_=getSdataOnly(test_Data_set)
    l_test_data,_=getLdataOnly(test_Data_set)
    val_loader = data.DataLoader(val_data, batch_size=args.batch_size)
    s_test_loader = data.DataLoader(s_test_data, batch_size=args.batch_size)
    l_test_loader = data.DataLoader(l_test_data, batch_size=args.batch_size)
    myresultFolder=resultFolder
    if args.disableUncertainiy: myresultFolder = resultFolder + "_WithoutUncertianty"
    if args.disableContrastive: myresultFolder = resultFolder + "_WithoutContrastive"
    indexfile=makeIndexFile(args,ExpName,args.disableContrastive,resultFolder=myresultFolder)
    cargs = Dict2Class(
        {"load_path": "logs/features/", "ood_samples": 2, "dataset": "Megas", "K_shift": 1, "k": 100.0, "t": 0.9,
         "target_list": sl_target_list, "target": target_Samples, "contrastive_alpha": contrastive_alpha,
         "uncertainity_beta": uncertainity_beta})

    # if we want to repeat this experiment multiple time
    for be in range(0,args.baseepoches):
        EpochExperiment=datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S")
        # check and select that is not in result file for current combination
        usedSeeds = getUsedSeedsinResultsfor(target_Samples,NumberOfCycles,traindatapercentage,myresultFolder)
        seedValue = args.seed
        while seedValue in usedSeeds: seedValue = random.randint(1, 50000)
        print("Using Seed", seedValue,"NumberOfCycles",NumberOfCycles,"target_Samples", target_Samples, "traindatapercentage", traindatapercentage)
        modelname=f"Joint_{ExpName}_{EpochExperiment}"
        if args.disableUncertainiy: modelname+="_WithoutUncertianty"
        if args.disableContrastive: modelname+="_WithoutContrastive"
        args.modelname=modelname
        #  set seed
        set_random_seed(seedValue)
        initial_training_budget=int(len(Data_set)*traindatapercentage/100)
        print("Initial_training_budget",initial_training_budget,"Initial_test_budget",len(test_Data_set))
        print("Number of Classes ",len(Data_set.alllabels),"Contrastive Class",len(Data_set.SL_uniqueLabels),"Uncertainity Class",len(Data_set.NA_uniqueLabels))

        # ----------------set active learning dataset----------------------------------------

        # select Indexies based on latenHybercube
        train_indexes, NegtiveIndex = selectInitalData(args, Data_set)
        print(len(train_indexes), "Samples Selected from ",len(Data_set)," From which have",np.unique([Data_set[i][3] for i in train_indexes], return_counts=True), "from total",np.unique([d[3] for d in Data_set], return_counts=True))
        # if plotGraphs: PlotDistribution(Data_set, train_indexes, graph_name=modelname + "_") # uncomment it if you want to plot distibution
        # get the Indexies of  Negative and SummaryWriterd postive Data Samples so that we can use postive data fro train and negative in contrastive learning as negative sample
        with open(indexfile, "a") as f:
            f.write(f"{EpochExperiment},{NumberOfCycles},{target_Samples},{traindatapercentage},{0},{-1},{-1},{-1},{-1},{-1},{','.join([str(t) for t in train_indexes])}\n")
        train_loader=data.DataLoader(Data_set, batch_size=args.batch_size, sampler=train_indexes)
        Data_set.setNegativeIndex(NegtiveIndex)
        unlabeled_index=[i for i in range(len(Data_set)) if i not in train_indexes]
        print("Train Have ", len(train_indexes), "Samples")
        print("Test Have ", len(test_Data_set), "Samples")
        print("Unlabeled Have ",len(unlabeled_index),"Samples")
        lpercent=sum([d[4]==1 for d in train_data])/len(train_data)*100 if args.dataSelection=="S" else sum([d[4]==0 for d in train_data])/len(train_data)*100 if args.dataSelection=="L" else 0
        SL_count_list,contrastivetrainlosslist,contrastivetestlosslist,uncertaintrainlosslist,uncertaintestlosslist,TrainDatCount=[],[],[],[],[],[]
        trainresultslist,stestresultslist,ltestresultslist=[],[],[]
        contrastiveModel, uncertainityModel=None,None
        sumslcount=None
        # ----------------Start active Cycles-----------------------------------a-----
        # This is Cycle loop
        for i in range(NumberOfCycles):
            set_random_seed(seedValue)
            uncertain_train,contrastive_train= args.disableUncertainiy is False,args.disableContrastive is False #and (contrastiveModel is None or norenew is False)
            contrastiveModel, uncertainityModel=None,None
            newcontrastiveModel, uncertainityModel, results=Training(args,seedValue,Data_set,train_loader,train_loader,val_loader,args.contrastiveEpoches,args.uncertainityEpoches,contrastiveModel, uncertainityModel,cycle=i,contrastive_train=contrastive_train,uncertain_train=uncertain_train,)
            if results[0] is not None: contrastivetrainlosslist.append(results[0])
            if results[1] is not None: uncertaintrainlosslist.append(results[1])
            if newcontrastiveModel is not None:
                contrastiveModel=newcontrastiveModel
            with torch.no_grad():
                # Select the sample based on contrastive and uncertainty approch
                query_L,_ = sampleSelection(cargs, Data_set, unlabeled_index, train_indexes,  test_batch_size, contrastiveModel, uncertainityModel,args.disableContrastive,args.disableUncertainiy)
            query_L=SelectCorrectContrastive(args,Data_set, unlabeled_index,query_L,sumslcount)
            # print("Query Indesx",len(query_L),np.unique([Data_set[i][4] for i in query_L],return_counts=True))
            if len(query_L)==0: exit()
            if args.disableUncertainiy is False:
                #acc,f1,roc
                trainresults,stestresults,ltestresults=EvaluateUncertainModel(uncertainityModel, [train_loader,s_test_loader,l_test_loader],device)
                trainresultslist.append(trainresults)
                stestresultslist.append(stestresults)
                ltestresultslist.append(ltestresults)
            else:
                trainresultslist.append([0,0,0])
                stestresultslist.append([0,0,0])
                ltestresultslist.append([0,0,0])

            SLCount={k:v for k,v in zip(*np.unique([Data_set[i][4] for i in query_L],return_counts=True))}
            if 0 not in SLCount: SLCount[0]=0
            if 1 not in SLCount: SLCount[1]=0
            SL_count_list.append([SLCount[0],SLCount[1]])
            sumslcount=np.sum(np.array(SL_count_list),axis=0)
            print("Cycle .............",i,"IndexFile",indexfile)
            print("Train Acc:", trainresultslist[-1][0], "STestacc:", stestresultslist[-1][0],f"In which we had {SLCount[0]} S and {SLCount[1]} L ")
            print("Scount: ",SLCount[0],"Lcount: ",SLCount[1],"Out of",len(query_L),"Total Scount: ",sumslcount[0],"Total Lcount: ",sumslcount[1],"Out of",len(query_L)*(i+1))
            print("SLcount: ",np.unique(np.concatenate([d[4] for d in train_loader]),return_counts=True),np.unique([d[4] for d in Data_set],return_counts=True))
            train_indexes += query_L
            # build the loadder for selected data
            with open(indexfile,"a") as f:
                f.write(f"{EpochExperiment},{NumberOfCycles},{target_Samples},{traindatapercentage},{i+1},{trainresultslist[-1][0]},{stestresultslist[-1][0]},{ltestresultslist[-1][0]},{SLCount[0]},{SLCount[1]},{','.join([str(t) for t in train_indexes]) }\n")
            random.shuffle(train_indexes)
            if NegtiveIndex is not None: random.shuffle(NegtiveIndex)
            sampler_labeled = data.sampler.SubsetRandomSampler(train_indexes)  # make indices initial to the samples
            train_loader = data.DataLoader(Data_set, sampler=sampler_labeled,batch_size=args.batch_size)
            TrainDatCount.append(len(train_indexes))
            Data_set.setNegativeIndex(NegtiveIndex)
            unlabeled_index = list(np.setdiff1d(list(unlabeled_index), list(train_indexes)))
            if (len(train_indexes)*100//len(Data_set))>(100-lpercent):
                print("Maximum S Points Reterived")
                break
            time.sleep(1)
        # Save results and model in file
        trainresultsnp,stestresultsnp,ltestresultsnp,SL_count_np=np.array(trainresultslist),np.array(stestresultslist),np.array(ltestresultslist),np.array(SL_count_list)
        results={"TrainingAcc":trainresultsnp[:,0], "STestAcc":stestresultsnp[:,0], "LTestAcc":ltestresultsnp[:,0],
                 "TrainF1":trainresultsnp[:,1],"STestF1":stestresultsnp[:,1],"LTestF1":ltestresultsnp[:,1],
                 "TrainRocAuc":trainresultsnp[:,2],"STestRocAuc":stestresultsnp[:,2],"LTestRocAuc":ltestresultsnp[:,2],
                 "SCountList":SL_count_np[:,0],"LCountList":SL_count_np[:,1],
                 "TrainDataCount":TrainDatCount,"contrastivetrainlosslist":np.array(contrastivetrainlosslist),"uncertaintrainlosslist":np.array(uncertaintrainlosslist)}
                 # ,"contrastivetestlosslist":np.array(contrastivetestlosslist),"uncertaintestlosslist":np.array(uncertaintestlosslist),}
        P={"seed":seedValue,"DataSelection":args.dataSelection,"target_Samples":target_Samples,"NumberOfCycles":NumberOfCycles,"traindatapercentage":traindatapercentage,"ContrastiveEpoches":conepoch,"Uncertainepoch":uncertainepoch}
        P.update({"ContrastivConvLayers":args.contrastivconvlayers,"uncertainityConvLayers":args.uncertainityconvlayers,"UncertainityLR":args.uncertainitylr,"ContrastiveLR":args.contrastivelr})
        P.update({"ContrastiveOptimizer":args.contrastiveoptimizer,"UncertinityOptimizer":args.uncertainityoptimizer,"BatchSize":args.batch_size,"Contrastiveloss":args.contrastivloss})
        P.update({"ContrastiveActivation1":contrastiveactivation1,"ContrastiveActivation2":contrastiveactivation2,"UncertainityActivation1":uncertainityactivation1,"UncertainityActivation2":unceratinityactivation2})
        P.update({"SimLambda":args.sim_lambda, "dis_lambda":args.dis_lambda,"contrastive_alpha":contrastive_alpha,"uncertainity_beta":uncertainity_beta})
        SLclasses,SLCount=np.unique(np.concatenate([d[4] for d in train_loader]), return_counts=True)
        print("SLcount: ",[f"{cl}->{count}" for cl,count in zip(SLclasses,SLCount)] ,np.unique([d[4] for d in Data_set], return_counts=True),"Total Selected",len(train_indexes)/len(Data_set)*100,"% ")
        saveModels(EpochExperiment,contrastiveModel, uncertainityModel, modelname, modelDir+args.DataName+"_"+args.dataSelection+"/")
        showResults(ExpName,EpochExperiment,P,modelname,args.DataName,args.dataSelection,results,plotGraphs,resultFolder=myresultFolder,disableContrastive=args.disableContrastive,disableUncertainiy=args.disableUncertainiy)
    # print("indexfile",indexfile)

    return indexfile

def getArguments(args):
    sys.argv.extend(args)
    parser = argparse.ArgumentParser(
                        prog='Query Sampling',
                        description='It Do the Active Learning based on contrastive and uncertianity')
    parser.add_argument('-dn', '--DatasetName', default=DatasetName, choices=["MegasFiles", "Random"], type=str, help="DataSet To Use")
    parser.add_argument('-di', '--DataIndex', default=2, type=int, help="DataSet To Use")
    # parser.add_argument('-d', '--datapath', default="cauchy__laplace_T_expon_normal_data", type=str, help="DataSet To Use")
    parser.add_argument('-ds', '--dataSelection', default="S", choices=["S", "L","SL"], type=str, help="DataSet To Select")
    parser.add_argument('-id', '--initdata',default=[5],nargs="+",type=int,help="Percentage of Inital Data")
    parser.add_argument('-cy', '--cycle',default=[NumberOfCycles],nargs="+",type=int,help="Number Of Cycles")
    parser.add_argument('-tp', '--totalpoints',default=800,type=int,help="Number Of Total Point to select")
    parser.add_argument('-dc', '--disableContrastive', action="store_true",default=disableContrastive,help="Whether You want to disable the contrastivelearning for experiment")
    parser.add_argument('-ep', '--baseepoches',default=basepepoces,type=int,help="Number Of Base Epoces with different seeds")
    parser.add_argument('-ce', '--contrastiveEpoches',default=conepoch,type=int,help="Number Of Epoces For Contrastive Lerarning")
    parser.add_argument('-ue', '--uncertainityEpoches',default=uncertainepoch,type=int,help="Number Of Epoces For Uncertainity Lerarning")
    parser.add_argument('-cls', '--contrastivloss',default=contrastivelossname,type=str,help="Contrastive Loss")
    parser.add_argument('-cl', '--contrastivconvlayers',default=contrastivconvlayers,type=int,help="Number Of Contrastiv convolution layers")
    parser.add_argument('-ul', '--uncertainityconvlayers',default=uncertainityconvlayers,type=int,help="Number Of uncertinity Uncertainity Lauer")
    parser.add_argument('-co', '--contrastiveoptimizer',default=contrastiveoptimizername,type=str,help="Optimizer for contrastive part")
    parser.add_argument('-uo', '--uncertainityoptimizer',default=uncertinityoptimizername,type=str,help="Optimizer for uncertinity part")
    parser.add_argument('-cr', '--contrastivelr',default=contrastive_learning_rate,type=float,help="Learning Rate for contrastive part")
    parser.add_argument('-ur', '--uncertainitylr',default=uncertainity_learning_rate,type=float,help="Learning Rate for uncertinity part")
    parser.add_argument('-b', '--batch_size',default=batch_size,type=int,help="Batch Size")
    parser.add_argument('-sl', '--sim_lambda',default=sim_lambda,type=float,help="Similarity Lambda")
    parser.add_argument('-dl', '--dis_lambda',default=dis_lambda,type=float,help="Dissimilarity Lambda")
    args = parser.parse_args()
    args.disableUncertainiy = False # We always use Uncertainity
    args.seed = np.random.randint(0, 1000000000)
    args.cyclelist=[cycle for cycle in args.cycle if args.totalpoints%cycle==0]
    assert len(args.cyclelist)>0," No Cyc divisible by total point"
    args.selectPointlist=[args.totalpoints//cycle for cycle in args.cyclelist]
    return args



def DoActiveLearning(args=[]):
    args=getArguments(args)
    ExpName=datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S")
    if args.DatasetName=="Random":
        _,_,dataname,_=getDataDetails(args.DataIndex)
        print("Running for Data",args.DataIndex,dataname)
        args.DataName=dataname
    elif args.DatasetName=="MegasFiles":
        args.DataName = "Megas"
    resultFolder = "Results/Result_"+args.DataName+"_"+args.dataSelection
    if not norenew:resultFolder += "renew"
    indexfilelist=[]
    for initdata in args.initdata:
        for cycle,tsample in zip(args.cyclelist,args.selectPointlist):
            # # indexfile=ProcessData(ExpName,DataPath,tsample,cycle,initdata,baseepoches,contrastiveEpochesin,uncertainityEpoches,disableContrastive,resultFolder=resultFolder)
            indexfile=ProcessData(ExpName,args,tsample,cycle,initdata,resultFolder=resultFolder)
            indexfilelist.append(indexfile)
    return indexfilelist[0]

if __name__ == '__main__':
    indexfile=DoActiveLearning()
    print("Index file",indexfile)


