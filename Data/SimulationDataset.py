import random,torch
# from Data.GenerateRandomData import generateData
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
import pandas as pd,numpy as np
from WKA.TimeStampAugmentation import Augmenetor
from scipy.stats import norm, laplace,expon,cauchy,logistic,t,gamma

Double=False
df=10
a=2
disdiff, nadiff = 0, 0

def AddTimeSeries(data,frequency=80000):
	t = np.linspace(data.min(), data.max(), data.shape[0], endpoint=False)
	# Generate the sine wave
	timesireis = 0.5 * np.sin(2 * np.pi * frequency * t)
	data+=timesireis.reshape(-1,1)
	return data


def generateDatafordistibution(size,distibution="normal",distibutionargs=(0,5)):
	if distibution=="normal":
		return norm.rvs(size=size, loc=distibutionargs[0], scale=distibutionargs[1])+distibutionargs[1]
	if distibution=="laplace":
		return laplace.rvs(size=size, loc=distibutionargs[0], scale=distibutionargs[1])+distibutionargs[1]
	if distibution=="expon":
		return expon.rvs(size=size, loc=distibutionargs[0], scale=distibutionargs[1])+distibutionargs[1]
	if distibution=="cauchy":
		return cauchy.rvs(size=size, loc=distibutionargs[0], scale=distibutionargs[1])+distibutionargs[1]
	if distibution=="logistic":
		return logistic.rvs(size=size, loc=distibutionargs[0], scale=distibutionargs[1])+distibutionargs[1]
	if distibution=="T":
		return t.rvs(size=size,  df=df, loc=distibutionargs[0], scale=distibutionargs[1])+distibutionargs[1]
	if distibution=="gamma":
		return gamma.rvs(size=size,a=a, loc=distibutionargs[0], scale=distibutionargs[1])+distibutionargs[1]




def generateData(folder,requiredSize,distibutionSmall,distibutionLarge,normallistsmall, abnormallistsmall, normallistlarge, abnormallistlarge,addtimeseires=True,seed=None):
	# half in case not double so both labale can have 50%
	if seed is None: seed=np.random.randint(0,1e+8)
	np.random.seed(seed)
	requiredSize = (requiredSize+1) // 2
	if "small" in folder:
		distibution = distibutionSmall
		normallist, abnormallist = normallistsmall, abnormallistsmall
	else:
		distibution = distibutionLarge
		normallist, abnormallist = normallistlarge, abnormallistlarge
	if len(normallist) == 3 and len(abnormallist) == 3:
		normalData = np.concatenate([np.concatenate([generateDatafordistibution((requiredSize, 1), distibution=d, distibutionargs=n) for n in normallist], axis=1) for d in distibution], axis=0)
		abnormalData = np.concatenate([np.concatenate([generateDatafordistibution((requiredSize, 1), distibution=d, distibutionargs=n) for n in abnormallist], axis=1) for d in distibution], axis=0)
	else:
		normalData = np.concatenate([generateDatafordistibution(requiredSize, distibution=d, distibutionargs=n) for d in distibution for n in normallist],axis=0)
		abnormalData = np.concatenate([generateDatafordistibution(requiredSize, distibution=d, distibutionargs=n) for d in distibution for n in abnormallist],axis=0)
		np.random.shuffle(normalData)
		np.random.shuffle(abnormalData)
	normalData, abnormalData = normalData[:requiredSize], abnormalData[:requiredSize]
	if "small" in folder:
		normalData[:, 0] += disdiff
		normalData[:, 2] += disdiff
		abnormalData[:, 0] += disdiff
		abnormalData[:, 2] += disdiff
		abnormalData[:, 1] += nadiff
		if addtimeseires:
			normalData = AddTimeSeries(normalData, frequency=8000)
			abnormalData = AddTimeSeries(abnormalData, frequency=8000)
	else:
		abnormalData[:, 1] += nadiff
		if addtimeseires:
			normalData = AddTimeSeries(normalData, frequency=10000)
			abnormalData = AddTimeSeries(abnormalData, frequency=10000)
	generatedData = np.concatenate([normalData, abnormalData], axis=0)
	return generatedData,normalData, abnormalData



def getClasses(requiredSize):
	return np.concatenate((np.zeros((requiredSize//2,)),np.ones((requiredSize//2,))),axis=0).astype(int)

class DistDataSet(Dataset):
	def __init__(self,mode="Random",requiredSize=7000,train=True,window_size=44,shuffle=True,normalize=True,normalizer=None,lpercent=100,distargs=None,seed=np.random.randint(1,1e+8),transform=torch.Tensor,LNASLA=False):
		self.mode=mode
		distibutionSmall,distibutionLarge,normallistsmall,abnormallistsmall,normallistlarge,abnormallistlarge=distargs
		normallistsmall, abnormallistsmall, normallistlarge, abnormallistlarge=self.splitDataCounts([normallistsmall,abnormallistsmall,normallistlarge,abnormallistlarge])
		self.normalize,self.normalizer=normalize,normalizer
		if mode=="Ads":
			normallistsmallname,abnormallistsmallname="__".join(["_".join([str(n1) for n1 in n]) for n in normallistsmall]),"__".join(["_".join([str(n1) for n1 in n]) for n in abnormallistsmall])
			normallistlargename,abnormallistlargename="__".join(["_".join([str(n1) for n1 in n]) for n in normallistlarge]),"__".join(["_".join([str(n1) for n1 in n]) for n in abnormallistlarge])
			distlist="__".join([normallistsmallname,abnormallistsmallname,normallistlargename,abnormallistlargename])
			modelname = f"WKA_{'_'.join(distibutionSmall)}___{'_'.join(distibutionLarge)}___{distlist}___{lpercent}_{normalize}_{seed}_{requiredSize[0]}"
			self.augmentor = Augmenetor(modelname=f"{modelname}")
		requiredSize=[r+window_size*4 for r in requiredSize]
		if train:
			# Sdata,_,_ = generateData("small", requiredSize[0], distibutionSmall, distibutionLarge, normallistsmall,abnormallistsmall, normallistlarge, abnormallistlarge,seed=seed)
			# Ldata,_,_ = generateData("large", int(requiredSize[0]*lpercent//100), distibutionSmall, distibutionLarge, normallistsmall,abnormallistsmall, normallistlarge, abnormallistlarge,seed=seed)
			SSize,Lsize=int(requiredSize[0]*(100-lpercent)/100),int(requiredSize[0]*lpercent/100)
			Sdata,_,_ = generateData("small", SSize, distibutionSmall, distibutionLarge, normallistsmall,abnormallistsmall, normallistlarge, abnormallistlarge,seed=seed)
			Ldata,_,_ = generateData("large", Lsize, distibutionSmall, distibutionLarge, normallistsmall,abnormallistsmall, normallistlarge, abnormallistlarge,seed=seed)
			self.fitnormalizer([Sdata,Ldata])
			Sdata,Ldata=self.transnormalizer([Sdata,Ldata])
			Snormaldata, Sabnormaldata = Sdata[:Sdata.shape[0]//2], Sdata[Sdata.shape[0]//2:]
			Lnormaldata, Labnormaldata = Ldata[:Ldata.shape[0]//2], Ldata[Ldata.shape[0]//2:]
			Snormaldata=np.stack([Snormaldata[i-window_size:i] for i in range(window_size,Snormaldata.shape[0])])
			sabnormaldata=np.stack([Sabnormaldata[i-window_size:i] for i in range(window_size,Sabnormaldata.shape[0])])
			Lnormaldata=np.stack([Lnormaldata[i-window_size:i] for i in range(window_size,Lnormaldata.shape[0])])
			Labnormaldata=np.stack([Labnormaldata[i-window_size:i] for i in range(window_size,Labnormaldata.shape[0])])
			Sdata=np.concatenate((Snormaldata,sabnormaldata),axis=0)
			Ldata=np.concatenate((Lnormaldata,Labnormaldata),axis=0)
			Sclasses = getClasses(Sdata.shape[0])
			if LNASLA:
				#SELECT AL L As ABNORMRAl Including L Normal
				Lclasses = np.ones((Ldata.shape[0],)).astype(int)
			else:
				Lclasses = getClasses(Ldata.shape[0])
			self.X = np.concatenate((Sdata, Ldata), axis=0)
			self.NA_Labels = np.concatenate((Sclasses, Lclasses), axis=0)
			self.Sl_Labels=np.concatenate((np.zeros(Sdata.shape[0]), np.ones(Ldata.shape[0])), axis=0).astype(int)
		else:
			data,_,_ = generateData("small", requiredSize[1], distibutionSmall, distibutionLarge, normallistsmall,abnormallistsmall, normallistlarge, abnormallistlarge,seed=seed)
			self.fitnormalizer([data])
			data=self.transnormalizer([data])[0]
			normaldata, abnormaldata = data[:data.shape[0]//2], data[data.shape[0]//2:]
			normaldata=np.stack([normaldata[i-window_size:i] for i in range(window_size,normaldata.shape[0])])
			abnormaldata=np.stack([abnormaldata[i-window_size:i] for i in range(window_size,abnormaldata.shape[0])])
			self.X=np.concatenate((normaldata,abnormaldata),axis=0)
			self.NA_Labels = getClasses(self.X.shape[0])
			self.Sl_Labels = np.zeros(self.X.shape[0]).astype(int)
		if shuffle:
			self.X,self.NA_Labels,self.Sl_Labels=self.shuffle((self.X,self.NA_Labels,self.Sl_Labels))
		self.traindata=np.rollaxis(self.X,2,1).astype(np.float32)
		self.trainlabels=np.copy(self.NA_Labels)
		self.trainlabels[self.Sl_Labels==1]+=2
		if self.mode=="Ads":
			x1, x2 = self.augmentor.getPositiveAugmentedDataBatch(self.X)
			self.x1,self.x2 = np.rollaxis(x1, 2, 1).astype(np.float32), np.rollaxis(x2, 2, 1).astype(np.float32)
			self.postiveAug=np.array([random.choice((xx1, xx2)) for xx1, xx2 in zip(self.x1, self.x2)])
			if transform is not None:
				self.postiveAug=transform(self.postiveAug)
				self.postiveAug12 = [transform(self.x1), transform(self.x2)]
			else:
				self.postiveAug12 = [self.x1, self.x2]
		negativeAug = [self.traindata[self.Sl_Labels == l] for l in np.unique(self.Sl_Labels)]
		for na in negativeAug: np.random.shuffle(na)
		# negative sample can be any class expect of that class
		negativeAug = {i: [negativeAug[j] for j in range(len(negativeAug)) if i != j] for i in range(len(np.unique(self.Sl_Labels)))}
		self.negativeAug = {i: np.concatenate(na) if len(na) > 0 else [] for i, na in negativeAug.items()}

		if transform is not None:
			self.traindata = transform(self.traindata)
		self.negativeIndexies = None
		self.alllabels = np.unique(self.trainlabels)
		self.SL_uniqueLabels = np.unique(self.Sl_Labels)
		self.NA_uniqueLabels = np.unique(self.NA_Labels)

	def splitDataCounts(self,distlist):
		if isinstance(distlist,str):
			return [[[int(d) for d in dt.split("_")] for dt in dl.split("__")] for dl in distlist]
		return distlist

	def setNegativeIndex(self, negativeIndexies):
		if negativeIndexies is not None:
			self.negativeIndexies = negativeIndexies

	def fitnormalizer(self,datalist):
		if self.normalize:
			if self.normalizer == None:
				self.normalizer =MinMaxScaler()
				data = np.concatenate(datalist)
				self.normalizer.fit(data)

	def transnormalizer(self,datalist):
		if self.normalize:
			return [self.normalizer.transform(data) for data in datalist]
		return datalist

	def shuffle(self,arraylist):
		p = np.random.permutation(arraylist[0].shape[0])
		return [a[p] for a in arraylist]

	def __len__(self):
		return self.NA_Labels.shape[0]

	def __getitem__(self, index,mode="train",s=0):
		if self.mode=="Ads":
			if mode == "train":
				if self.negativeIndexies is None:
					if len(self.negativeAug[self.Sl_Labels[index]]) > 0:
						negaugs = self.negativeAug[self.Sl_Labels[index]]
						negativeSample = negaugs[random.randint(0, negaugs.shape[0]) - 1]
					else:
						negativeSample = []
				else:
					negativeSample = self.traindata[random.choice(self.negativeIndexies)]
				return self.traindata[index], self.postiveAug[index], negativeSample, self.trainlabels[index], self.Sl_Labels[index], self.NA_Labels[index], index
			else:
				return self.traindata[index], self.postiveAug12[s][index], self.Sl_Labels[index], index
		elif self.mode=="RandomAds":
			if mode == "train":
				if self.negativeIndexies is None:
					if len(self.negativeAug[self.Sl_Labels[index]]) > 0:
						negaugs = self.negativeAug[self.Sl_Labels[index]]
						negativeSample = negaugs[random.randint(0, negaugs.shape[0]) - 1]
					else:
						negativeSample = []
				else:
					negativeSample = self.traindata[random.choice(self.negativeIndexies)]
				postiveSample=random.choice(self.traindata)
				return self.traindata[index], postiveSample, negativeSample, self.trainlabels[index], self.Sl_Labels[index], self.NA_Labels[index], index
			else:
				return self.traindata[index], self.traindata[index], self.Sl_Labels[index], index
		return self.traindata[index],0,0,self.trainlabels[index],self.Sl_Labels[index],self.NA_Labels[index],index


def DistribuitonList(Index=1):
	if Index==1:
		distibutionSmall, distibutionLarge = ['expon'], 'laplace_logistic'.split('_')
		distlist = '126_87__29_77__65_43 49_99__120_58__48_3 45_25__13_81__71_19  6_14__132_88__55_9'.split()
		lpercent, normalize, requiredSize, seed = 50, True, (5000, 1000), 88879653
	elif Index==10:
		distibutionSmall, distibutionLarge = ['expon'], 'laplace_logistic'.split('_')
		distlist = '126_87__29_77__65_43 49_99__120_58__48_3 45_25__13_81__71_19  6_14__132_88__55_9'.split()
		lpercent, normalize, requiredSize, seed = 80, True, (7000, 1000), 88879653
	return distibutionSmall, distibutionLarge,distlist,lpercent, normalize, requiredSize, seed
