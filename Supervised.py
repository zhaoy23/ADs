import datetime,argparse,os.path
import random,copy,torch,numpy as np
import sys
import time
from pathlib import Path

import pandas as pd
from torch import nn,optim
from getitem import Megas
from models.CustomModels import CustomModel
import torch.utils.data as data
from sklearn.metrics import f1_score
batch_size=128
Data_set=Megas("../data/",window_size=44,Normalize=True)
test_Data_set=Megas("../data/",window_size=44,mode="eval",Normalize=True)

device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

def selectIndexices(dataset,sindex):
	dataset.traindata,dataset.trainlabels=dataset.traindata[sindex],dataset.trainlabels[sindex]
	dataset.Sl_Labels, dataset.NA_Labels=dataset.Sl_Labels[sindex], dataset.NA_Labels[sindex]


def getSdataOnly(DataDet,fixedIndex=None):
	dataset=makeACopy(DataDet)
	sindex=dataset.Sl_Labels==0
	selectIndexices(dataset, sindex)
	if fixedIndex is not None:
		rearanegIndex=dict(zip(np.arange(sindex.shape[0])[sindex],range(sindex.shape[0])))
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


def evaluation(model,loader,criterion,labeltype="na"):
	loss_list, pred_list, label_list = [], [], []
	model=model.eval()
	with torch.no_grad():
		for n, (anchor, postive, negative, labels, salabel, nalabel, index) in enumerate(loader):
			anchor = anchor.to(device)
			labels = nalabel.to(device) if labeltype == "na" else salabel.to(device)
			predictions = model(anchor)
			loss = criterion(predictions, labels)
			pred_list.append(torch.argmax(predictions, dim=1))
			label_list.append(labels)
			loss_list.append(loss.detach())
	loss, pred, label = torch.mean(torch.tensor(loss_list)).numpy(), torch.cat(pred_list), torch.cat(label_list)
	acc = (torch.sum(pred == label) / pred.shape[0]).cpu().numpy()
	f1 = f1_score(label.cpu(), pred.cpu())
	return acc,f1,loss,pred.shape[0]

def Training(model,trainloader,valloader,testloader=None,indexname="No",labeltype="na",epochs=150,learning_rate=0.001,outfile="Res",outFolder="Results/Supervised/"):
	# Supervised Training
	Path(outFolder).mkdir(exist_ok=True,parents=True)
	difffile=datetime.datetime.now().strftime('%y_%m_%d_%H_%M_%S')+str(random.randint(0,5000))
	outFile=f"{outFolder}{outfile}_{difffile}.csv"
	print("Results will save in ",outFile)
	modelpath=f"ModelWeights/BestSupervised_{difffile}.pt"
	with open(outFile,"w") as f:
		f.write("Epoch,TrainCount,ValCount,IndexFilename,TrainAcc,ValAcc,TrainF1,ValF1,TrainLoss,ValLoss\n")

	model.train()
	criterion = nn.CrossEntropyLoss().to(device)
	optimizer=optim.Adam(model.parameters(), lr=learning_rate)
	BestResults=[0]
	for epoch in range(0, epochs):
		trainloss, pred_list,label_list = [],[],[]
		model=model.train()
		for n, (anchor, postive, negative, labels, salabel, nalabel, index) in enumerate(trainloader):
			anchor = anchor.to(device)
			labels = nalabel.to(device) if labeltype=="na" else salabel.to(device)
			optimizer.zero_grad()
			predictions = model(anchor)
			loss = criterion(predictions, labels)
			loss.backward()
			optimizer.step()
			pred_list.append(torch.argmax(predictions,dim=1))
			label_list.append(labels)
			trainloss.append(loss.detach())
		trainloss, trainpred, train_label=torch.mean(torch.tensor(trainloss)).numpy(),torch.cat(pred_list),torch.cat(label_list)
		train_Acc=(torch.sum(trainpred==train_label)/trainpred.shape[0]).cpu().numpy()
		trainf1=f1_score(train_label.cpu(),trainpred.cpu())
		valacc,valf1,valloss,valcount=evaluation(model, valloader, criterion, labeltype=labeltype)
		print(f"Epoch {epoch+1},Training auccuracy is {train_Acc} Train F1 is {trainf1}, Val Accuracy is {valacc}, Val F1 {valf1}, Training Loss is {trainloss}, Val Loss is {valloss}")
		with open(outFile,"a") as f:
			f.write(f"{epoch+1},{trainpred.shape[0]},{valcount},{indexname},{train_Acc},{valacc},{trainf1},{valf1},{trainloss},{trainloss}\n")
		if valacc>BestResults[0]:
			print("Saving New Best Model")
			torch.save(model,modelpath)
			BestResults=[valacc,train_Acc,trainloss,trainloss,epoch]
	print("---------------Training Complete-----------------------")
	print(f"Best Results are at epoch {BestResults[-1]}, Train Acc {BestResults[1]}, Vall Acc {BestResults[0]}, Train Loss {BestResults[2]}, Vall Loss {BestResults[3]}")
	if testloader is None: return BestResults[1],BestResults[0]
	model = torch.load(modelpath)
	testacc,testf1, testloss,testcount = evaluation(model, testloader, criterion, labeltype=labeltype)
	with open(outFile, "a") as f:
		f.write(f"Test Accuracy on best model, {testacc},F1,{testf1},Test Count,{testcount}\n")
	print(f"Test acc {testacc}, F1 {testf1}")
	print("Results saved At ", outFile)
	return outFile


def Supervisedvalidation(epochs=200):
	#Supervised Training on S Data
	train_data_set,_=getSdataOnly(Data_set)
	test_data_set,_=getSdataOnly(test_Data_set)
	train_Data_set,val_Data_set=Train_Test_Split(train_data_set,test_percentage=.2)
	print("Train len: ",len(train_Data_set),"Val len",len(val_Data_set),"Test len:",len(test_data_set))

	train_loader = data.DataLoader(train_Data_set, batch_size=batch_size)
	val_loader = data.DataLoader(val_Data_set, batch_size=batch_size)
	test_loader = data.DataLoader(test_data_set, batch_size=batch_size)

	naModel = CustomModel(num_classes=len(train_data_set.NA_uniqueLabels)).to(device)
	Training(naModel,train_loader,val_loader,test_loader,labeltype="na",epochs=epochs,learning_rate=0.001,outfile="Full")
	print("Train len: ",len(train_Data_set),"Val len",len(val_Data_set),"Test len:",len(test_data_set))



def ADSIndexices(file=None,epochs=200,adstype="Contrastive"):
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
		naModel = CustomModel(num_classes=len(train_data.NA_uniqueLabels)).to(device)
		outfile="ADS_Con" if adstype=="Contrastive" else "ADS_NonCon"
		Training(naModel, train_loader,val_loader, test_loader,indexname=indexname, labeltype="na", epochs=epochs, learning_rate=0.001, outfile=outfile)
		print("LCount is ",Lcount)
		# print("Train Test acc",d.split(",")[4:6])
		print("From Fulldata",len(Data_set),"Train have", len(train_data),"Val have", len(val_data), "Test have ", len(test_data_set), "instances")

def getFixNIndex(n=684,ADSFolder="Results/Indexices/Contrastive/"):
	filepath=ADSFolder+os.listdir(ADSFolder)[0]
	with open(filepath) as f:
		data=f.readlines()[:2]
	data=data[1].split(",")[len(data[0].split(","))-1:]
	return list(map(int,data[:n]))

def RandomIndexices(ltype="S",fixsamples=684,randomsample=400,epochs=200):
	fixedIndex = getFixNIndex(n=fixsamples)
	if ltype=="S":
		train_data,fixedIndex=getSdataOnly(Data_set,fixedIndex)
	else:
		train_data=makeACopy(Data_set)
	test_data_set,_=getSdataOnly(test_Data_set)
	randomindexies=random.sample(range(len(train_data)),k=fixsamples+randomsample)
	indexies=fixedIndex+list(filter(lambda x:x not in fixedIndex,randomindexies))[:randomsample]
	selectIndexices(train_data,indexies)
	train_data,val_data=Train_Test_Split(train_data, test_percentage=.2,shuffle=True)
	print("Train have",len(train_data),"Val have",len(val_data),"Test have ",len(test_data_set),"instances")
	train_loader = data.DataLoader(train_data, batch_size=batch_size,  num_workers=4)
	val_loader = data.DataLoader(val_data, batch_size=batch_size,  num_workers=4)
	test_loader = data.DataLoader(test_data_set, batch_size=batch_size,  num_workers=4)
	naModel = CustomModel(num_classes=len(train_data.NA_uniqueLabels)).to(device)
	outFile=Training(naModel, train_loader, val_loader,test_loader, labeltype="na", epochs=epochs, learning_rate=0.001,outfile="Random_OnlyS" if ltype=="S" else "Random_SL")
	print("Train have",len(train_data),"Val have",len(val_data),"Test have ",len(test_data_set),"instances")
	train_lcount = np.sum([d[4] for d in train_data])
	val_lcount = np.sum([d[4] for d in val_data])
	test_lcount = np.sum([d[4] for d in test_data_set])
	with open(outFile, "a") as f:
		f.write(f"Lcount  in Train , {train_lcount},Val,{val_lcount},Test,{test_lcount}\n")
	print(F"L count in train {train_lcount}, val {val_lcount}, test {test_lcount}")
	time.sleep(3)


def getArguments():
	# Parse the command line arguments
	spmethods=["Validation","ADSCon","ADSNonCon","RandomS","RandomSL"]
	# sys.argv.extend("--method ADSCon".split())
	parser = argparse.ArgumentParser(prog='Supervised Learning',description='It Do the Active Learning based on contrastive and uncertianity')
	parser.add_argument('-a', '--ActiveLearningOnly',default="ADSCon",choices=spmethods,type=str,help="Method to use")
	parser.add_argument('-m', '--method',default="ADS",choices=spmethods,type=str,help="Method to use")
	parser.add_argument('-f', '--indexfile',default=None,type=str,help="index File")
	parser.add_argument('-e', '--epochs',default=400,type=int,help="Epoches for Supervised")
	args = parser.parse_args()
	print(args)
	return args,spmethods

def showTestAccuracyFor(prefix,folder="Results/Supervised/"):
	files=[folder+f for f in os.listdir(folder) if f.startswith(prefix)]
	for file in files:
		with open(file) as f:
			data=f.readlines()[-2]
		print(data[:-1],file.split("/")[-1])



