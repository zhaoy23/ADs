# this File Include all Nessasry Informartion FOr supericed method. like Random, RandomS, RandomSL
import warnings
import datetime,argparse,os.path
import random,copy,torch,numpy as np
import sys
import time
from pathlib import Path

from sklearn.model_selection import train_test_split
from torch import nn,optim
from Data.getitem import Megas
from Data.SimulationDataset import DistDataSet,DistribuitonList
from models.CustomModels import CustomModel
import torch.utils.data as data
from sklearn.metrics import f1_score,roc_auc_score
from utils.utils import (Dict2Class,set_random_seed,showResults,getTrainingIndicesByLatenHybercube,
                         getLdataOnly,getSdataOnly,selectIndexices,makeACopy,Train_Test_Split,splitValidationDate)
batch_size=128
# Settings the warnings to be ignored
warnings.filterwarnings('ignore')

def getDatasets(args,DataPath="data"):
	if "Megas" in args.DatasetName:
		Data_set=Megas(DataPath,window_size=44,Normalize=True,dataSelection=args.dataSelection)
		test_Data_set=Megas(DataPath,window_size=44,mode="eval",Normalize=True,dataSelection=args.dataSelection)
	else:
		distibutionSmall, distibutionLarge,distlist,lpercent, normalize, requiredSize, seed=DistribuitonList(int(args.DataName.split("_")[-1]))
		distargs = [distibutionSmall, distibutionLarge]+distlist
		Data_set = DistDataSet(mode="RandomAds", requiredSize=[requiredSize[0]], train=True, window_size=44, shuffle=True,
		                       normalize=normalize, normalizer=None, lpercent=lpercent, distargs=distargs, seed=seed,LNASLA=False)
		test_Data_set = DistDataSet(mode="RandomAds", requiredSize=[requiredSize[1]], train=True, window_size=44, shuffle=True,
		                       normalize=normalize, normalizer=None, lpercent=lpercent, distargs=distargs, seed=seed,LNASLA=False)
	return Data_set,test_Data_set

device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")




def evaluation(model,loader,criterion,labeltype="na"):
	loss_list, pred_list, label_list = [], [], []
	model=model.eval()
	with torch.no_grad():
		for n, (anchor, postive, negative, labels, salabel, nalabel, index) in enumerate(loader):
			anchor = anchor.to(device)
			labels = nalabel.to(device) if labeltype == "na" else salabel.to(device)
			predictions = model(anchor)
			loss = criterion(predictions, labels)
			pred_list.append(predictions)
			label_list.append(labels)
			loss_list.append(loss.detach())
	loss, prob, label = torch.mean(torch.tensor(loss_list)).numpy(), torch.cat(pred_list,dim=0), torch.cat(label_list)
	pred=torch.argmax(prob, dim=1)
	acc = float((torch.sum(pred == label) / pred.shape[0]).cpu().numpy())
	f1 = f1_score(label.cpu(), pred.cpu())
	rocauc=roc_auc_score(label.cpu(),prob[:,1].cpu())
	return acc,f1,rocauc,loss,pred.shape[0]

def Training(args,model,trainloader,valloader=None,testloaders=None,dataname="data",indexname="No",labeltype="na",outfile="Res",outFolder="Results/Supervised/",SaveModel=False,printEpoch=True):
	# Supervised Training
	Path(outFolder).mkdir(exist_ok=True,parents=True)
	difffile=datetime.datetime.now().strftime('%y_%m_%d_%H_%M_%S')+str(random.randint(0,5000))
	outFile=f"{outFolder}{outfile}_{difffile}.csv"
	print("Results will save in ",outFile)
	os.makedirs(f"ModelWeights/{dataname}",exist_ok=True)
	modelpath=f"ModelWeights/{dataname}/BestSupervised_{difffile}.pt"
	with open(outFile,"w") as f:
		f.write("Epoch,TrainCount,ValCount,IndexFilename,TrainAcc,ValAcc,TrainF1,ValF1,TrainRocAuc,ValRocAuc,TrainLoss,ValLoss\n")

	model.train()
	criterion = nn.CrossEntropyLoss().to(device)
	optimizer=optim.Adam(model.parameters(), lr=args.learningrate)
	BestResults=[0]
	for epoch in range(0, args.supervisedepochs):
		trainloss, pred_list,label_list = [],[],[]
		model=model.train()
		# print(model.state_dict())
		# print(optimizer.state_dict())
		# input()
		for n, (anchor, postive, negative, labels, salabel, nalabel, index) in enumerate(trainloader):
			anchor = anchor.to(device)
			labels = nalabel.to(device) if labeltype=="na" else salabel.to(device)
			optimizer.zero_grad()
			predictions = model(anchor)
			loss = criterion(predictions, labels)
			loss.backward()
			optimizer.step()
			pred_list.append(predictions)
			label_list.append(labels)
			trainloss.append(loss.detach())

		trainloss, trainpredprob, train_label=torch.mean(torch.tensor(trainloss)).numpy(),torch.cat(pred_list,dim=0),torch.cat(label_list)
		trainpred=torch.argmax(trainpredprob,dim=1)
		train_Acc=(torch.sum(trainpred==train_label)/trainpred.shape[0]).cpu().numpy()
		trainf1=f1_score(train_label.cpu(),trainpred.cpu())
		trainrocauc=roc_auc_score(train_label.cpu(),trainpredprob.detach().cpu()[:,1])
		if valloader is not None:
			valacc,valf1,valrocauc,valloss,valcount=evaluation(model, valloader, criterion, labeltype=labeltype)
		else:
			valacc,valf1,valrocauc,valcount="N/A","N/A","N/A",0
		# print(f"Epoch {epoch+1},Training auccuracy is {train_Acc} Train F1 is {trainf1}, Val Accuracy is {valacc}, Val F1 {valf1}, Training Loss is {trainloss}, Val Loss is {valloss}")
		if printEpoch:
			print(f"Epoch {epoch+1},Training auccuracy is {train_Acc}  Val Accuracy is {valacc}")
		with open(outFile,"a") as f:
			f.write(f"{epoch+1},{trainpred.shape[0]},{valcount},{indexname},{train_Acc},{valacc},{trainf1},{valf1},{trainrocauc},{valrocauc},{trainloss},{trainloss}\n")
		if valloader is not None:
			if valacc>BestResults[0]:
				BestResults = [valacc, train_Acc, trainloss, trainloss,trainf1,valf1, epoch]
				if SaveModel:
					print("Saving New Best Model")
					torch.save(model,modelpath)
		else:
			torch.save(model, modelpath)
	# if valacc==1.0: break
	print("---------------Training Complete-----------------------")
	# print(f"Best Results are at epoch {BestResults[-1]}, Train Acc {BestResults[1]}, Vall Acc {BestResults[0]}, Train Loss {BestResults[2]}, Vall Loss {BestResults[3]}")
	if testloaders is None: return outFile,BestResults[1],BestResults[0],train_Acc,valacc,BestResults[4],BestResults[5],trainf1,valf1
	if SaveModel:
		model = torch.load(modelpath)
	resultlist=[]
	for testloader in testloaders:
		testacc,testf1,testrocauc, testloss,testcount = evaluation(model, testloader, criterion, labeltype=labeltype)
		resultlist.append([testacc,testf1,testrocauc,testcount])
		with open(outFile, "a") as f:
			f.write(f"Test Accuracy on best model, {testacc},F1,{testf1},RocAuc,{testrocauc},Test Count,{testcount}\n")
		print(f"Test acc {testacc}, F1 {testf1}")
		print("Results saved At ", outFile)
		# return outFile,BestResults[1],BestResults[0],train_Acc,valacc,BestResults[4],BestResults[5],trainf1,valf1
	return outFile,resultlist,difffile


def Supervisedvalidation(args,DataPath,dataname="Megas_S"):
	#Supervised Training on S Data
	set_random_seed(args.seed)
	Data_set,test_Data_set=getDatasets(args,DataPath=DataPath)
	if args.dataSelection=="S":
		train_data_set,_=getSdataOnly(Data_set)
	else:
		train_data_set, _ = getLdataOnly(Data_set)
	S_test_data_set,_=getSdataOnly(test_Data_set)
	L_test_data_set, _ = getLdataOnly(test_Data_set)

	train_Data_set,val_Data_set=Train_Test_Split(train_data_set,test_percentage=.2)
	print("Train len: ",len(train_Data_set),"Val len",len(val_Data_set),"S Test len:",len(S_test_data_set),"L Test len:",len(L_test_data_set))
	train_loader = data.DataLoader(train_Data_set, batch_size=batch_size)
	val_loader = data.DataLoader(val_Data_set, batch_size=batch_size)
	s_test_loader = data.DataLoader(S_test_data_set, batch_size=batch_size)
	l_test_loader = data.DataLoader(L_test_data_set, batch_size=batch_size)

	naModel = CustomModel(num_classes=len(train_data_set.NA_uniqueLabels),nconv=args.numberconvolution).to(device)
	outFolder = f"Results/Supervised/"
	_,results,EXPName=Training(args,naModel,train_loader,val_loader,[train_loader,s_test_loader,l_test_loader],dataname=DataPath,labeltype="na",outfile="Full_"+args.dataSelection,outFolder=outFolder)
	print("Train len: ",len(train_Data_set),"Val len",len(val_Data_set),"S Test len:",len(S_test_data_set),"L Test len:",len(L_test_data_set))
	[Trainacc,Trainf1,Trainrocauc,Traincount],[s_testacc,s_testf1,s_testrocauc,s_testcount],[l_testacc,l_testf1,l_testrocauc,l_testcount]=results
	SLCOUNT={k:v for k,v in list(zip(*np.unique(train_data_set.Sl_Labels,return_counts=True)))}
	FileName=f"Results/Result_{dataname}/ALL_Supervised_{args.dataSelection}_Scores.csv"
	if not os.path.exists(FileName):
		with open(FileName,"w") as f:
			f.write(f"Method,DataSelection,InitialData,QueryData,EXPName,SEED,Supervisedepochs,NumberConvolution,BatchSize,LearningRate,TotolTrainCount,TrainCountUsed,STestCount,LTestCount,TrainAcc,TrainF1,TrainRocAuc,S_TestAcc,S_TestF1,S_TestRocAuc,L_TestAcc,L_TestF1,L_TestRocAuc,TrainSCount,TrainLCount\n")
	with open(FileName,"a") as f:
		f.write(f"Supervised,{args.dataSelection},100,100,{EXPName},{args.seed},{args.supervisedepochs},{args.numberconvolution},{args.batch_size},{args.learningrate},{Traincount},{Traincount},{s_testcount},{l_testcount},{Trainacc},{Trainf1},{Trainrocauc},{s_testacc},{s_testf1},{s_testrocauc},{l_testacc},{l_testf1},{l_testrocauc},{SLCOUNT.get(0,0)},{SLCOUNT.get(1,0)}\n")
	print("Results Saved at",FileName)



def ADSIndexices(args,DataPath,file=None,epochs=200,adstype="Contrastive"):
	Data_set, test_Data_set = getDatasets(DataPath=DataPath)
	print("ADSIndexices",file)
	# Do Supervised Lerning on Data selected By Active Learning
	test_data_set,_=getSdataOnly(test_Data_set)
	indexFolder="Results/Indexices/"+adstype+"/"
	if file is None:
		IndexFile=indexFolder+os.listdir(indexFolder)[-1]
	else:
		IndexFile = file
	with open(IndexFile) as f:
		dataindex=f.readlines()[1:]
	Lcount=0
	indexname=IndexFile.split(".")[0].split("/")[-1][6:]
	for d in dataindex:
		Lcount+=int(d.split(",")[6])
		if int(d.split(",")[0]) !=int(d.split(",")[3]): continue
		indexies=[int(d) for d in  d.split(",")[7:]]
		train_data=makeACopy(Data_set)
		selectIndexices(train_data, indexies)
		val_data,train_data=Train_Test_Split(train_data, test_percentage=.8)
		print("From Full Data",len(Data_set),"Train have", len(train_data),"Val have", len(val_data), "Test have ", len(test_data_set), "instances")
		train_loader = data.DataLoader(train_data, batch_size=batch_size)
		val_loader = data.DataLoader(val_data, batch_size=batch_size)
		test_loader = data.DataLoader(test_data_set, batch_size=batch_size)
		naModel = CustomModel(num_classes=len(train_data.NA_uniqueLabels),nconv=args.numberconvolution).to(device)
		outfile="ADS_Con" if adstype=="Contrastive" else "ADS_NonCon"
		outFolder = f"Results/Result_{DataPath}/Supervised/"
		Training(naModel, train_loader,val_loader, test_loader,dataname=DataPath,indexname=indexname, labeltype="na", epochs=epochs, learning_rate=0.001, outfile=outfile,outFolder=outFolder)
		print("LCount is ",Lcount)
		# print("Train Test acc",d.split(",")[4:6])
		print("From Fulldata",len(Data_set),"Train have", len(train_data),"Val have", len(val_data), "Test have ", len(test_data_set), "instances")

def getFixNIndex(n=684,dataset="data"):
	ADSFolder = f"Results/Result_{dataset}/Indexices/Contrastive/"
	filepath=ADSFolder+sorted(os.listdir(ADSFolder))[-1]
	with open(filepath) as f:
		data=f.readlines()[:2]
	data=data[1].split(",")[len(data[0].split(","))-1:]
	return list(map(int,data[:n]))

def RandomIndexices(args,DataPath,dataname="Megas",ltype="S"):
	# Select Random Indexices for the Random Experiment
	set_random_seed(args.seed)
	Data_set,test_Data_set=getDatasets(args,DataPath=DataPath)
	# fixedIndex = getFixNIndex(n=fixsamples,dataset=dataname)
	if ltype=="S":
		train_data,_=getSdataOnly(Data_set)
		train_data,val_data=Train_Test_Split( train_data, test_percentage=.2,shuffle=True)
		fixedIndex = list(getTrainingIndicesByLatenHybercube(train_data, list(range(len(train_data))), args.initdata))
	elif ltype == "L":
		train_data, _ = getLdataOnly(Data_set)
		train_data,val_data=Train_Test_Split( train_data, test_percentage=.2,shuffle=True)
		fixedIndex = list(getTrainingIndicesByLatenHybercube(train_data, list(range(len(train_data))), args.initdata))
	elif ltype=="SL_L":
		train_data,val_data=splitValidationDate(Data_set,valType="L")
		fixedIndex = list(getTrainingIndicesByLatenHybercube(train_data, list(range(len(train_data))), args.initdata))
	else:
		train_data=makeACopy(Data_set)
		train_data,val_data=Train_Test_Split( train_data, test_percentage=.2,shuffle=True)
		fixedIndex = list(getTrainingIndicesByLatenHybercube(train_data, list(range(len(train_data))), args.initdata))

	print(ltype,np.unique(val_data.Sl_Labels,return_counts=True))
	S_test_data_set,_=getSdataOnly(test_Data_set)
	L_test_data_set,_=getLdataOnly(test_Data_set)
	randomindexies=random.sample([d for d in range(len(train_data)) if d not in fixedIndex],args.totalpoints)
	indexies=fixedIndex+randomindexies
	TotolTrainCount=len(train_data)
	selectIndexices(train_data,indexies)
	print("Train len: ",len(train_data),"Val len",len(val_data),"S Test len:",len(S_test_data_set),"L Test len:",len(L_test_data_set))
	train_loader = data.DataLoader(train_data, batch_size=args.batch_size,  num_workers=4)
	val_loader = data.DataLoader(val_data, batch_size=args.batch_size,  num_workers=4)
	s_test_loader = data.DataLoader(S_test_data_set, batch_size=args.batch_size,  num_workers=4)
	l_test_loader = data.DataLoader(L_test_data_set, batch_size=args.batch_size,  num_workers=4)
	naModel = CustomModel(num_classes=len(train_data.NA_uniqueLabels),nconv=args.numberconvolution).to(device)
	outFolder = f"Results/Result_{dataname}/Supervised_{ltype}/"
	outfile="Random_OnlyS" if ltype=="S" else "Random_OnlyL"  if ltype=="L" else "Random_SL_L" if ltype=="SL_L" else "Random_SL_S"
	outFile,results,EXPName=Training(args,naModel, train_loader, val_loader,[train_loader,s_test_loader,l_test_loader], dataname=dataname,labeltype="na",outfile=outfile,outFolder=outFolder)
	print("Train len: ",len(train_data),"Val len",len(val_data),"S Test len:",len(S_test_data_set),"L Test len:",len(L_test_data_set))
	[Trainacc,Trainf1,Trainrocauc,Traincount],[s_testacc,s_testf1,s_testrocauc,s_testcount],[l_testacc,l_testf1,l_testrocauc,l_testcount]=results
	SLCOUNT={k:v for k,v in list(zip(*np.unique(train_data.Sl_Labels,return_counts=True)))}
	FileName=f"Results/Result_{dataname}/ALL_Supervised_{args.dataSelection}_Scores.csv"
	if not os.path.exists(FileName):
		with open(FileName,"w") as f:
			f.write(f"Method,DataSelection,InitialData,QueryData,EXPName,SEED,Supervisedepochs,NumberConvolution,BatchSize,LearningRate,TotolTrainCount,TrainCountUsed,STestCount,LTestCount,TrainAcc,TrainF1,TrainRocAuc,S_TestAcc,S_TestF1,S_TestRocAuc,L_TestAcc,L_TestF1,L_TestRocAuc,TrainSCount,TrainLCount\n")
	with open(FileName,"a") as f:
		f.write(f"Random_{ltype},{args.dataSelection},{args.initdata},{args.totalpoints},{EXPName},{args.seed},{args.supervisedepochs},{args.numberconvolution},{args.batch_size},{args.learningrate},{TotolTrainCount},{Traincount},{s_testcount},{l_testcount},{Trainacc},{Trainf1},{Trainrocauc},{s_testacc},{s_testf1},{s_testrocauc},{l_testacc},{l_testf1},{l_testrocauc},{SLCOUNT.get(0,0)},{SLCOUNT.get(1,0)}\n")
	print("Results Saved at",FileName)
	train_lcount = np.sum([d[4] for d in train_data])
	val_lcount = np.sum([d[4] for d in val_data])
	with open(outFile, "a") as f:
		f.write(f"Lcount  in Train , {train_lcount},Val,{val_lcount},Test,{SLCOUNT.get(1,0)}\n")
		f.write(f"Scount  in Train , {len(train_data)-train_lcount},Val,{len(val_data)-val_lcount},Test,{SLCOUNT.get(0, 0)}\n")
	print(F"L count in train {train_lcount}, val {val_lcount}, Test LCount {SLCOUNT.get(1,0)}")


def showTestAccuracyFor(prefix,folder="Results/Supervised/"):
	files=[folder+f for f in os.listdir(folder) if f.startswith(prefix)]
	for file in files:
		with open(file) as f:
			data=f.readlines()[-2]
		print(data[:-1],file.split("/")[-1])


def MakeMannualContrastiveSelection(args,DataPath="../DATA/data/",InitalData=10,QueryData=800,dataSelection="L",dtype="With CL"):
	set_random_seed(args.seed)
	dataname = "Megas_"+dataSelection+"_ConSEL"
	outFolder = f"Results/Result_{dataname}/ConSelected_{dataSelection}_{dtype.replace(" ","_")}/"
	Data_set=Megas(DataPath,window_size=44,Normalize=True,dataSelection=dataSelection)
	test_Data_setL=Megas(DataPath,window_size=44,mode="eval",Normalize=True,dataSelection="L")
	test_Data_setS=Megas(DataPath,window_size=44,mode="eval",Normalize=True,dataSelection="S")
	LN_indexes = list(getTrainingIndicesByLatenHybercube(Data_set, [d[-1] for d in Data_set if d[3] == 2], InitalData))
	LAN_indexes = list(getTrainingIndicesByLatenHybercube(Data_set, [d[-1] for d in Data_set if d[3] == 3], InitalData))
	train_indexes = LN_indexes + LAN_indexes
	random.shuffle(train_indexes)
	unlabeled_index = [i for i in range(len(Data_set)) if i not in train_indexes]
	opdatapercent=(5,12) if dtype=="With CL" else (18,22)
	if dataSelection=="L":
		opdatacount=random.randint(*opdatapercent)*QueryData//100
		SelectedIndex=random.choices([d for d in unlabeled_index if Data_set[d][4] == 0],k=opdatacount)+random.choices([d for d in unlabeled_index if Data_set[d][4] == 1],k=QueryData-opdatacount)
	elif dataSelection=="S":
		opdatacount=random.randint(*opdatapercent)*QueryData//100
		SelectedIndex=random.choices([d for d in unlabeled_index if Data_set[d][4] == 1],k=opdatacount)+random.choices([d for d in unlabeled_index if Data_set[d][4] == 0],k=QueryData-opdatacount)
	train_indexes=train_indexes+SelectedIndex
	train_loader = data.DataLoader(Data_set, batch_size=args.batch_size, sampler=train_indexes)
	test_loaderL = data.DataLoader(test_Data_setL, batch_size=args.batch_size )
	test_loaderS = data.DataLoader(test_Data_setS, batch_size=args.batch_size)
	naModel = CustomModel(num_classes=2,nconv=args.numberconvolution).to(device)
	outfile = "Results"
	outFile,results,expname=Training(args,naModel, train_loader, None,[test_loaderL,test_loaderS], dataname=dataname+dtype.replace(' ','_'),labeltype="na",outfile=outfile,outFolder=outFolder)
	opCount=sum([Data_set[ind][4]==0 for ind in train_indexes]) if dataSelection=="L" else sum([Data_set[ind][4]==1 for ind in train_indexes])
	[Trainacc, Trainf1, Trainrocauc],[Ltestacc, Ltestf1, Ltestrocauc ],[Stestacc, Stestf1, Stestrocauc]=results
	FileName=f"Results/Result_{dataname}/ConTrastiveSelection_Count.csv"
	if not os.path.exists(FileName):
		with open(FileName,"w") as f:
			f.write(f"DataSelection,Dtype,InitialData,QueryData,EXPName,SEED,Supervisedepochs,NumberConvolution,BatchSize,LearningRate,TrainAcc,TrainF1,TrainRocAuc,S_TestAcc,S_TestF1,S_TestRocAuc,L_TestAcc,L_TestF1,L_TestRocAuc,OP_Count,TotolTrain\n")
	with open(FileName,"a") as f:
		f.write(f"{dataSelection},{dtype},{InitalData},{QueryData},{expname},{args.seed},{args.supervisedepochs},{args.numberconvolution},{args.batch_size},{args.learningrate},{Trainacc},{Trainf1},{Trainrocauc},{Stestacc},{Stestf1},{Stestrocauc},{Ltestacc},{Ltestf1},{Ltestrocauc},{opCount},{len(train_indexes)}\n")
	print("Data Written at",FileName)



def argParser():
	parser = argparse.ArgumentParser(prog='Supervised Learning Contrastive',description='It Do the Active Learning based on contrastive and uncertianity')
	parser.add_argument('-con', '--contrastive', action='store_true',help="Contrastive")
	parser.add_argument('-ds', '--dataSelection',default="L",type=str,help="Data Selection")
	parser.add_argument('-qd', '--QueryData',default=800,type=int,help="QueryData")
	parser.add_argument('-id', '--initalData',default=10,type=float,help="Intial Data")
	parser.add_argument('-lr', '--learningrate',default=0.0001,type=float,help="Learning Rate")
	parser.add_argument('-e', '--supervisedepochs',default=4,type=int,help="Epoches for Supervised")
	parser.add_argument('-bs', '--batch_size',default=127,type=int,help="BatchSize")
	parser.add_argument('-nc', '--numberconvolution', default=5, type=int, help="Number of Convoluations")
	args = parser.parse_args()
	args.seed=random.randint(0,100000000)

	return args

