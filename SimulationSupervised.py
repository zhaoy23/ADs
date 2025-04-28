import os.path,random
import torch,numpy as np
from GenerateRandomData import generateData
from Supervised import Training,evaluation
from models.CustomModels import CustomModel
from training.JoinTraining import trainContrastive
import time
from torch.utils.data import Dataset
import pandas as pd,numpy as np
import torch.utils.data as data
from utils.utils import set_random_seed
from RandomDataset import DistDataSet
from scipy.stats import entropy
from utils.utils import set_random_seed,getTrainingIndicesByLatenHybercube
from training.utils import getOptimizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")




distibutionSmall, distibutionLarge = ["normal"], ["laplace"]
normallistsmall, abnormallistsmall = [(2, 5), (50, 5), (100, 5)], [(5, 10), (55, 10), (105, 20)]
normallistlarge, abnormallistlarge = [(2, 5), (50, 5), (100, 5)], [(2, 5), (50, 5), (100, 5)]
distargs=[distibutionSmall,distibutionLarge,normallistsmall,abnormallistsmall,normallistlarge,abnormallistlarge]

def getClasses(requiredSize):
	return np.concatenate((np.zeros((requiredSize//2,)),np.ones((requiredSize//2,))),axis=0).astype(int)




def ifexperimentExists(expname,resultFile,workingonFile):
	if os.path.exists(resultFile):
		with open(resultFile) as f:
			for line in f.readlines():
				if expname in line:
					return True
	if os.path.exists(workingonFile):
		with open(workingonFile) as f:
			for line in f.readlines():
				if expname in line:
					return True
	return False

def SaveGoodModels(results,models,experimentname,resdiff=0.15,):
	if results[0][3] > 0.80 and results[1][3] > 0.61 and results[0][1] - results[1][1] > resdiff and results[0][3] - results[1][3] > resdiff:
		os.makedirs("DistModelWeights",exist_ok=True)
		modelpath="DistModelWeights/Model_"+experimentname
		print("Saving Model at",modelpath)
		torch.save(models[0], modelpath+"S.pt")
		torch.save(models[0], modelpath+"SL.pt")

def calcualteUncertainity(model,Data):
	loader = data.DataLoader(Data, batch_size=64)
	predictions,labellist=[],[]
	model.eval()
	with torch.no_grad():
		for anchor, postive, negative, labels, sllabel, nalabel, index in loader:
			predictions.append(torch.nn.Sigmoid()(model(anchor.to(device))))
			labellist.append(sllabel)
	predictions=torch.cat(predictions,dim=0)
	labels=torch.cat(labellist)
	uncertaintyScore = entropy(predictions.T.cpu())
	Suncertianity=[u for u,l in zip(uncertaintyScore,labels) if l==0]
	Luncertianity=[u for u,l in zip(uncertaintyScore,labels) if l==1]
	return torch.mean(torch.Tensor(Suncertianity)),torch.mean(torch.Tensor(Luncertianity))


def CalcualteLatentAccuracy(dataloader,model,Label=""):
	latentspaces,labelslist=[],[]
	with torch.no_grad():
		for anchor, postive, negative, labels, sllabel, nalabel, index in dataloader:
			out,outputs_aux=model(anchor.to(device), simclr=True, penultimate=True)
			latentspaces.append(outputs_aux["simclr"])
			# out=model(anchor.to(device))
			# latentspaces.append(out)
			labelslist.append(sllabel)
	latentspace=torch.cat(latentspaces).cpu().numpy()
	labels=torch.cat(labelslist).numpy()
	lc=LogisticRegression()
	lc.fit(latentspace,labels)
	pred=lc.predict(latentspace)
	return accuracy_score(labels,pred)

def getContrastiveAccuracies(seed,Data_set,traindatapercentage=20,batch_size=64,contrastiveoptimizer="sgd",contrastivelr=0.01,lossname="Cosine",conepochs=100):
	set_random_seed(seed)
	SN_indexes = list(getTrainingIndicesByLatenHybercube(Data_set, [d[-1] for d in Data_set if d[3] == 0], traindatapercentage))
	SAN_indexes = list(getTrainingIndicesByLatenHybercube(Data_set, [d[-1] for d in Data_set if d[3] == 1], traindatapercentage))
	train_indexes = SN_indexes + SAN_indexes
	if 2 in Data_set.alllabels:
		LN_indexes = list(getTrainingIndicesByLatenHybercube(Data_set, [d[-1] for d in Data_set if d[3] == 2], traindatapercentage))
		LAN_indexes = list(getTrainingIndicesByLatenHybercube(Data_set, [d[-1] for d in Data_set if d[3] == 3], traindatapercentage))
		NegtiveIndex = LN_indexes + LAN_indexes
	else:
		NegtiveIndex = list(getTrainingIndicesByLatenHybercube(Data_set, [d[-1] for d in Data_set if d[3] == 3], traindatapercentage))
	Data_set.setNegativeIndex(NegtiveIndex)
	unlabeldIndex = [i for i in range(len(Data_set)) if i not in train_indexes and i not in NegtiveIndex]
	FullDataLoader = data.DataLoader(Data_set, batch_size=batch_size)
	contrastive_train_loder = data.DataLoader(Data_set, batch_size=batch_size, sampler=train_indexes)
	unlabeled_loader = data.DataLoader(Data_set, batch_size=batch_size, sampler=unlabeldIndex)
	set_random_seed(seed)
	contrastiveModel = CustomModel(num_classes=len(Data_set.SL_uniqueLabels)).to(device)  # create a new model for each NumberOfCycles so that it can have random weights
	contrastiveoptimizer, scheduler_warmup = getOptimizer(contrastiveModel, contrastiveoptimizer, contrastivelr,addschedular=True)
	bflacc=CalcualteLatentAccuracy(FullDataLoader, contrastiveModel, "Before Full Data")
	buacc=CalcualteLatentAccuracy(unlabeled_loader, contrastiveModel, "Before Unlabeled Data")
	# beforeweights = contrastiveModel.state_dict()["conv1.weight"].clone()
	trainContrastive(conepochs, 0, contrastiveModel, contrastiveoptimizer, scheduler_warmup, lossname,
					 contrastive_train_loder,  device=device, savemodel=False)
	# afterwerights = contrastiveModel.state_dict()["conv1.weight"]
	afacc=CalcualteLatentAccuracy(FullDataLoader, contrastiveModel, "After Full Data")
	auacc=CalcualteLatentAccuracy(unlabeled_loader, contrastiveModel, "After Unlabeled Data")
	return bflacc,buacc,afacc,auacc

def RandomDistributionTraining(lpercent,epoches,distargs,requiredSize=(7000,1000),SaveDistribution=False,normalize=False,seed=np.random.randint(1,1e+8),resultFile="DISTResults.csv"):
	traindatapercentage,batch_size,contrastiveoptimizer,contrastivelr,lossname,conepochs = 20,64,"sgd",0.01,"Cosine",100
	LNASLA = True
	distibutionSmall, distibutionLarge, normallistsmall, abnormallistsmall, normallistlarge, abnormallistlarge=distargs
	normallistsmallstr = "__".join(["_".join([str(n) for n in ns]) for ns in normallistsmall])
	abnormallistsmallstr = "__".join(["_".join([str(n) for n in ns]) for ns in abnormallistsmall])
	normallistlargestr = "__".join(["_".join([str(n) for n in ns]) for ns in normallistlarge])
	abnormallistlargestr = "__".join(["_".join([str(n) for n in ns]) for ns in abnormallistlarge])
	requiredSizeStr='_'.join([str(s) for s in requiredSize])
	experimentname=f"{epoches},{lpercent},{requiredSizeStr},{normalize},{'_'.join(distibutionSmall)},{'_'.join(distibutionLarge)},{normallistsmallstr},{abnormallistsmallstr},{normallistlargestr},{abnormallistlargestr},{seed},{traindatapercentage},{batch_size},{contrastiveoptimizer},{contrastivelr},{lossname},{conepochs},{LNASLA}"
	if SaveDistribution:
		if ifexperimentExists(experimentname, resultFile,"CSV/Workingon.csv"):
			print(experimentname ,"experiment exisits")
			return
		with open("CSV/Workingon.csv","a") as f:
			f.write(experimentname+"\n")
	print("Selected Experiment",experimentname)
	unresults,models=[],[]
	set_random_seed(seed)
	trainData = DistDataSet(mode="RandomAds",requiredSize=requiredSize, train=True, distargs=distargs, lpercent=lpercent,normalize=normalize, seed=seed,LNASLA=LNASLA)
	testdata = DistDataSet(requiredSize=requiredSize, train=False, distargs=distargs, normalizer=trainData.normalizer,normalize=normalize, seed=seed)
	for i in range(2):
		if i==0:
			# S Data Only
			trainIndex=[d[-1] for d in trainData if d[4]==0]
			train_loader = data.DataLoader(trainData, batch_size=64,sampler=trainIndex)
		else:
			# S And L Data
			train_loader = data.DataLoader(trainData, batch_size=64)
		test_loader = data.DataLoader(testdata, batch_size=64)
		set_random_seed(seed)
		uncertinitymodel = CustomModel(num_classes=2).to(device)
		_,besttrain,bestval,lasttrain,lastval=Training(uncertinitymodel, train_loader, test_loader, testloader=None, dataname="TEMP", indexname="No", labeltype="na",epochs=epoches, learning_rate=0.001, outfile="Res", outFolder="Results/TEMP/",printEpoch=False)
		models.append(uncertinitymodel)
		unresults.append([besttrain,bestval,lasttrain,lastval])

	conresults=getContrastiveAccuracies(seed,trainData,traindatapercentage=traindatapercentage,batch_size=batch_size,contrastiveoptimizer=contrastiveoptimizer,contrastivelr=contrastivelr,lossname=lossname,conepochs=conepochs )
	SSuncertinyt,SLuncerity=calcualteUncertainity(models[0],trainData)
	SLSuncertinyt,SLLuncerity=calcualteUncertainity(models[1],trainData)
	print(f"S  Best Train Accuracy {unresults[0][0]:.3} Test Accuracy {unresults[0][1]:.3} Last Train Accuracy {unresults[0][2]:.3} Test Accuracy {unresults[0][3]:.3} SUncertinty {SSuncertinyt:.3} LUncertinity {SLuncerity:.3}")
	print(f"SL Best Train Accuracy {unresults[1][0]:.3} Test Accuracy {unresults[1][1]:.3} Last Train Accuracy {unresults[1][2]:.3} Test Accuracy {unresults[1][3]:.3} SUncertinty {SLSuncertinyt:.3} LUncertinity {SLLuncerity:.3}")
	print(f"Con Results Before Full {conresults[0]:.3} Before Train Unlabels {conresults[1]:.3} After Train Full  {conresults[2]:.3} After Unlabeled Acc {conresults[3]:.3}")
	# print(CheckThreeAccuracies(models[0],requiredSize,distargs,lpercent,normalize,seed))
	if SaveDistribution:
		SaveGoodModels(unresults, models,experimentname)
		with open(resultFile, "a") as f:
			f.write(f"{experimentname},{unresults[0][0]},{unresults[0][1]},{unresults[0][2]},{unresults[0][3]},{unresults[1][0]},{unresults[1][1]},{unresults[1][2]},{unresults[1][3]},{SSuncertinyt},{SLuncerity},{SLSuncertinyt},{SLLuncerity},{conresults[0]},{conresults[1]},{conresults[2]},{conresults[3]}\n")
	print("************************* Experiment Completed ***************************")
	return unresults,[SSuncertinyt,SLuncerity,SLSuncertinyt,SLLuncerity]


def tryDifferentPattrens(epoches=200,filename="CSV/DISTConResultsLN.csv"):
	os.makedirs("CSV",exist_ok=True)
	if not os.path.exists(filename):
		with open(filename,"w") as f:
			f.write("Epoches,Lpercnet,FullDataSize,Normalizee,Dist1,Dist2,Dist1Normal,Dist1Abnormal,Dist2Normal,Dist2Anormal,SEED,Contraindatapercentage,ConBatch_size,contrastiveoptimizer,contrastivelr,Conlossname,conepochs,ChoseLN_AS_LA,SBestTrainAcc,SBestVallAcc,SLastTrainAcc,SLastValAcc,SLBestTrainAcc,SLBestVallAcc,SLLastTrainAcc,SLLastValAcc,SModleSUncertainity,SModelLUncertainity,SLModleSUncertainity,SLModelLUncertainity,Berofe Training Full Acc,Before Training Unlabeld Acc,After Training Full Acc,After Training Unlabeld Acc\n")
	alldists=["normal","laplace","expon","cauchy","logistic","gamma","T"]
	while True:
		distibutionSmall=random.sample(alldists,1)
		otherdist=[d for d in alldists if d not in distibutionSmall]
		distibutionLarge=random.sample(otherdist,random.randint(2,4))
		normallistsmall=[(random.randint(2,150),random.randint(2,100)) for i in range(3)]
		abnormallistsmall=[(random.randint(2,150),random.randint(2,100)) for i in range(3)]
		normallistlarge=[(random.randint(2,150),random.randint(2,100)) for i in range(3)]
		abnormallistlarge=[(random.randint(2,150),random.randint(2,100)) for i in range(3)]
		distargs = [distibutionSmall, distibutionLarge, normallistsmall, abnormallistsmall, normallistlarge, abnormallistlarge]
		for i in range(20):
			normalize = random.random() > .5
			lpercent=random.choice([20,30,40,50,60,70,80,90])
			seed=np.random.randint(1, 1e+8)
			trainsize = random.choice([7000,10000,70000, 100000, 120000, 150000])
			requiredSize = [trainsize, max(1000,trainsize//10)]
			print("Choosed",normalize,lpercent,seed,trainsize,requiredSize)
			RandomDistributionTraining(lpercent, epoches, distargs,requiredSize=requiredSize, SaveDistribution=True,normalize=normalize,seed=seed,resultFile=filename)



def RunPattrenWithMegas(datapath="cauchy__laplace_T_expon_normal_data",DataIndex=5,epoches=200):
	# distibutionSmall, distibutionLarge = ["logistic"], "cauchy_gamma_laplace_expon".split("_")
	# distlist="98_92__105_49__30_63	145_90__132_27__2_81 36_19__55_3__140_72  147_16__92_21__61_42".split()
	# lpercent,normalize,requiredSize,seed=140,False,(7000,1000),26337414
	distibutionSmall, distibutionLarge = ['cauchy'], 'gamma_normal'.split('_')
	distlist = '40_59__100_42__127_8	97_64__38_52__60_4 47_36__45_44__14_26  58_58__145_74__35_88'.split()
	lpercent, normalize, requiredSize, seed = 50, True, (7000, 1000), 7587294
	dist1normallist = [[int(d) for d in dt.split("_")] for dt in distlist[0].split("__")]
	dist1abnormallist = [[int(d) for d in dt.split("_")] for dt in distlist[1].split("__")]
	dist2normallist = [[int(d) for d in dt.split("_")] for dt in distlist[2].split("__")]
	dist2abnormallist = [[int(d) for d in dt.split("_")] for dt in distlist[3].split("__")]
	distargs = [distibutionSmall, distibutionLarge, dist1normallist, dist1abnormallist, dist2normallist, dist2abnormallist]

	trainData = DistDataSet(requiredSize=requiredSize, train=True, distargs=distargs, lpercent=lpercent,normalize=normalize, seed=seed)
	testdata = DistDataSet(requiredSize=requiredSize, train=False, distargs=distargs,normalizer=trainData.normalizer, normalize=normalize, seed=seed)

	# trainData = Megas(datapath, DataIndex, window_size=44,Normalize=True)
	# testdata = Megas(datapath, DataIndex, window_size=44, mode="eval", Normalize=True)
	# print(len(trainData),len(testdata))
	Results=[]
	for i in range(2):
		if i==0:
			# for S  Data only
			train_indexes=[d[-1] for d in trainData if d[4]==0]  #For S Data
			train_loader = data.DataLoader(trainData, batch_size=64,sampler=train_indexes)
		else:
			# for S and L Data
			train_loader = data.DataLoader(trainData, batch_size=64)
		test_loader = data.DataLoader(testdata, batch_size=64)
		set_random_seed(seed)
		model = CustomModel(num_classes=2).to(device)
		_, besttrain, bestval, lasttrain, lastval = Training(model, train_loader, test_loader, testloader=None,
															 dataname="TEMP", indexname="No", labeltype="na",
															 epochs=epoches, learning_rate=0.001, outfile="Res",
															 outFolder="Results/TEMP/")
		print(np.unique(np.concatenate([t[3].numpy() for t in train_loader], axis=0), return_counts=True))
		print(np.unique(np.concatenate([t[3].numpy() for t in test_loader], axis=0), return_counts=True))
		Results.append([besttrain,bestval,lasttrain,lastval])
	for i,(besttrain,bestval,lasttrain,lastval) in enumerate(Results):
		print(f"{'S ' if i==0 else 'SL'} Accuracy besttrain: {besttrain} bestval: {bestval} lasttrain:{lasttrain} lastval {lastval}")


def RunPattren(epoches=200):
	distibutionSmall, distibutionLarge = ['cauchy'], 'laplace_logistic'.split('_')
	distlist = '126_87__29_77__65_43	49_99__120_58__48_3 45_25__13_81__71_19  6_14__132_88__55_9'.split()
	lpercent, normalize, requiredSize, seed = 50, True, (7000, 1000), 88879653
	dist1normallist = [[int(d) for d in dt.split("_")] for dt in distlist[0].split("__")]
	dist1abnormallist = [[int(d) for d in dt.split("_")] for dt in distlist[1].split("__")]
	dist2normallist = [[int(d) for d in dt.split("_")] for dt in distlist[2].split("__")]
	dist2abnormallist = [[int(d) for d in dt.split("_")] for dt in distlist[3].split("__")]
	distargs = [distibutionSmall, distibutionLarge, dist1normallist, dist1abnormallist, dist2normallist, dist2abnormallist]
	RandomDistributionTraining(lpercent, epoches, distargs,requiredSize=requiredSize, SaveDistribution=False,normalize=normalize,seed=seed)


def checkFilterPattrensUncertinity(epoches=200):
	df=pd.read_csv("ServerResultes/DifferentDistribution/CSV/DISTFullResults1.csv")
	i=1
	for inf,row in df.iterrows():
		if inf<43708: continue
		# minSAcc=max(row["SLastValAcc"],row["SBestVallAcc"])
		# besSLAcc=max(row["SLLastValAcc"],row["SLBestVallAcc"])
		if row["Lpercnet"]>85: continue
		# if len(row["Dist2"].split("_"))>2: continue
		# if  row["SLastValAcc"]>0.80 and row["SLLastValAcc"]>0.61 and minSAcc-besSLAcc>0.15:
		# if not row['Normalizee']: continue
		resdiff=.10
		# print(df.columns)
		if row["SLastValAcc"] > 0.80 and row["SLLastValAcc"] > 0.60 and row["SBestVallAcc"] - row["SLBestVallAcc"] > resdiff  and row["SLastValAcc"] - row["SLLastValAcc"] > resdiff:
			ds=row["FullDataSize"].split("_")
			dist1normallist = [[int(d) for d in dt.split("_")] for dt in  row['Dist1Normal'].split("__")]
			dist1abnormallist = [[int(d) for d in dt.split("_")] for dt in row['Dist1Abnormal'].split("__")]
			dist2normallist = [[int(d) for d in dt.split("_")] for dt in row['Dist2Normal'].split("__")]
			dist2abnormallist = [[int(d) for d in dt.split("_")] for dt in row['Dist2Anormal'].split("__")]
			distargs = [row['Dist1'].split('_'), row['Dist2'].split("_"),dist1normallist, dist1abnormallist, dist2normallist,dist2abnormallist]
			print("*#"*20,inf,"*#"*20)
			resuls,uncertinity=RandomDistributionTraining(row['Lpercnet'], epoches, distargs, requiredSize=(int(ds[0]),int(ds[1])), SaveDistribution=False,normalize=row['Normalizee'], seed=row['SEED'])
			print("**************************",f"{row['Lpercnet']}  {row['SBestVallAcc']} > {row['SLBestVallAcc']}    {row['SLastValAcc']} > {row['SLLastValAcc']}")
			# print("**************************",f"{row['SModleSUncertainity']} < {row['SModelLUncertainity']}    {row['SLModleSUncertainity']} < {row['SLModelLUncertainity']}")
			# print(row)
			print(f"elif DataIndex == {i}:")
			print(f"	distibutionSmall, distibutionLarge = ['{row['Dist1']}'], '{row['Dist2']}'.split('_')")
			print(f"	distlist='{row['Dist1Normal']}	{row['Dist1Abnormal']} {row['Dist2Normal']}  {row['Dist2Anormal']}'.split()")
			print(f"	lpercent,normalize,requiredSize,seed={row['Lpercnet']},{row['Normalizee']},({ds[0]},{ds[1]}),{row['SEED']}")
			if uncertinity[0]<uncertinity[1] and uncertinity[2]<uncertinity[3]:
				input(f"Checked {inf} ?")


def filterPatrnes():
	# df=pd.read_csv("ServerResultes/DifferentDistribution/CSV/DISTResults.csv")
	# df=pd.read_csv("ServerResultes/DifferentDistribution/CSV/DISTFullResults.csv")
	df=pd.read_csv("ServerResultes/IronamanDist/DISTConResults.csv")
	i=1
	filtedResults=[]
	for inf,row in df.iterrows():
		# if row["Lpercnet"]>85: continue
		resdiff=.10
		# if row["SLastValAcc"] < 0.80 or row["SLLastValAcc"] < 0.60: continue
		if row["SLBestVallAcc"] < 0.55 or row["SLLastValAcc"] < 0.55: continue
		if row["SBestVallAcc"] - row["SLBestVallAcc"] < resdiff  or row["SLastValAcc"] - row["SLLastValAcc"] < resdiff: continue
		if row['SModleSUncertainity'] >= row['SModelLUncertainity'] or row['SLModleSUncertainity'] >= row['SLModelLUncertainity']: continue
		if row['Berofe Training Full Acc'] >= row['After Training Full Acc'] or row['Before Training Unlabeld Acc'] >= row['After Training Unlabeld Acc']: continue
		print("**************************",f"{row['Lpercnet']}  {row['SBestVallAcc']} > {row['SLBestVallAcc']}    {row['SLastValAcc']} > {row['SLLastValAcc']}")
		print("**************************",f"{row['SModleSUncertainity']} < {row['SModelLUncertainity']}    {row['SLModleSUncertainity']} < {row['SLModelLUncertainity']}")
		# print(f"elif DataIndex == {i}:")
		ds=row["FullDataSize"].split("_")
		print(f"	distibutionSmall, distibutionLarge = ['{row['Dist1']}'], '{row['Dist2']}'.split('_')")
		print(f"	distlist='{row['Dist1Normal']}	{row['Dist1Abnormal']} {row['Dist2Normal']}  {row['Dist2Anormal']}'.split()")
		print(f"	lpercent,normalize,requiredSize,seed={row['Lpercnet']},{row['Normalizee']},({ds[0]},{ds[1]}),{row['SEED']}")
		filtedResults.append(row)
		i+=1
	# pd.DataFrame(filtedResults).to_csv("ServerResultes/DifferentDistribution/CSV/Filtered.csv",index=None)
	# row1=df[df["SEED"]=="55376545"]
	# row2=df[df["SEED"]=="91087894"]
	# df=pd.concat((row1,row2))
	# df.to_csv("Seed.csv")


if __name__ == '__main__':
	tryDifferentPattrens()
	# RunPattrenwithDifferenSeeds()
	# RunPattrenWithMegas()
	# RunPattren()
	# checkFilterPattrensUncertinity()
	# filterPatrnes()

