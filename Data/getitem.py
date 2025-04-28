import time
from torch.utils.data import Dataset
import pandas as pd,numpy as np
import os,random,torch
from utils.utils import set_random_seed
from WKA.TimeStampAugmentation import Augmenetor

class Megas(Dataset):
    # it load the dataset from Files
    def __init__(self, path,dataindex="",window_size=44,trans=torch.Tensor,dataSelection="S",collectDataPercent=1,mode="train",Normalize=False,labelProbabaility=False,seed=None):
        self.normalrange,self.abnormalrange = (5, 35),(60, 90)
        self.mode=mode
        self.dataSelection=dataSelection
        self.labelProbabaility=labelProbabaility
        if seed is not None: set_random_seed(seed)
        if Normalize: path=path.replace("data","data_normalized")
        # self.augmentor=Augmenetor(modelname=f"WTK_Autoencoder_{path}{dataindex}")
        self.augmentor=Augmenetor(modelname=f"WTK_Autoencoder_normalized")
        self.path=path
        if "/" not in path:path=f"../DATA/Data{dataindex}/{path}/"
        self.loadData(path,window_size=window_size,transform=trans,collectDataPercent=collectDataPercent)
        self.alllabels=np.unique(self.trainlabels)
        self.SL_uniqueLabels=np.unique(self.Sl_Labels)
        self.NA_uniqueLabels=np.unique(self.NA_Labels)
        self.Normalize=Normalize
        self.negativeIndexies=None

    def setNegativeIndex(self,negativeIndexies):
        self.negativeIndexies=negativeIndexies

    def loadData(self, DataPath, window_size, transform, collectDataPercent):
        self.loadDataFormat(DataPath, window_size, transform, collectDataPercent)

    def loadFile(self,filepath,window_size,label=None,collectDataPercent=1):
        #  if label is None label differently for normal and abnormal
        # read the file and get the data from normal and abnormal ranges
        df=pd.read_csv(filepath, names=['X', 'Y', 'Z'])
        r, b = pd.qcut(df.index, np.arange(0., 1.01, 0.05), retbins=True)
        sequence = {int(value * 100): int(key) for key, value in zip(b.round(), np.arange(0., 1.05, 0.05))}
        labels = np.zeros(df.shape[0]) + 2
        labels[sequence[self.normalrange[0]]:sequence[self.normalrange[1]]]=0
        labels[sequence[self.abnormalrange[0]]:sequence[self.abnormalrange[1]]]=1
        labels[list(range(sequence[self.normalrange[0]], sequence[self.normalrange[1]]))] = 0
        labels[list(range(sequence[self.abnormalrange[0]], sequence[self.abnormalrange[1]]))] = 1
        feturelen = df.shape[1]
        df["label"] = labels
        traindata,trainlabel = [],[]
        for lab in [0,1]:
            dflab=df[df["label"]==lab]
            for index in range(window_size, int(dflab.shape[0])):
                traindata.append(dflab.iloc[index - window_size:index,:feturelen].values)
                trainlabel.append(lab)
        traindata,trainlabel=np.array(traindata),np.array(trainlabel)
        if label is not None:
            trainlabel=np.ones(traindata.shape[0],dtype=int) * label
        if collectDataPercent<1:
            idx = np.random.permutation(traindata.shape[0])
            traindata, trainlabel = traindata[idx], trainlabel[idx]
            index=int(traindata.shape[0]*collectDataPercent)
            traindata,trainlabel=traindata[:index],trainlabel[:index]
        return traindata,trainlabel

    def loadDataFormat(self, DataPath, window_size, transform, collectDataPercent):
        # S1 + S2 + Normal, S1 + S2 + ABNormal, L + Normal, L + ABNormal,
        traindata, trainlabels,postiveAug,postiveAug1,postiveAug2,negativeAug=[],[],[],[],[], [[],[]]
        v=None
        datacount={"megas1":[],"megas2":[],"largeL ":[]}
        for dtype in ["megas1","megas2"]:
            if self.mode=="train":
                datafiles=[DataPath+"megas_small/"+f for f in os.listdir(DataPath + "megas_small") if dtype in f][:4]
            else:
                datafiles = [DataPath + "megas_small/" + f for f in os.listdir(DataPath + "megas_small") if dtype in f][4:5]
                # if self.dataSelection == "S":
                #     datafiles=[DataPath + "megas_small/" + f for f in os.listdir(DataPath + "megas_small") if dtype in f][4:5]
                # else:
                #     datafiles = []
            for f in datafiles:
                td,tl= self.loadFile(f,window_size,v,collectDataPercent)
                traindata.append(td)
                trainlabels.append(tl)
                x1,x2=self.augmentor.getPositiveAugmentedDataBatch(td)
                postiveAug.append(np.array([random.choice((xx1,xx2)) for xx1,xx2 in zip(x1,x2)]))
                postiveAug1.append(x1)
                postiveAug2.append(x2)
                datacount[dtype].append(np.unique(tl,return_counts=True)[1])

        # n_classes = np.unique(np.concatenate(trainlabels)).shape[0]
        n_classes=2
        if self.mode == "train":
            datafiles = [DataPath + "megax_large/"+f for f in os.listdir(DataPath + "megax_large")[:3]]
        else:
            datafiles = [DataPath + "megax_large/" + f for f in os.listdir(DataPath + "megax_large")[3:4]]
            # if self.dataSelection == "S":
            #     datafiles=[]
            # else:
            #     datafiles = [DataPath + "megax_large/" + f for f in os.listdir(DataPath + "megax_large")[3:4]]
            # # datafiles = [DataPath + "megax_large/"+f for f in os.listdir(DataPath + "megax_large")[3:4]]
        for f in datafiles:
            td, tl = self.loadFile(f, window_size, v, collectDataPercent)
            traindata.append(td)
            trainlabels.append(tl+n_classes)
            x1,x2=self.augmentor.getPositiveAugmentedDataBatch(td)
            postiveAug.append(np.array([random.choice((xx1,xx2)) for xx1,xx2 in zip(x1,x2)]))
            postiveAug1.append(x1)
            postiveAug2.append(x2)
            datacount["largeL "].append(np.unique(tl, return_counts=True)[1])

        datacount={k:np.sum(np.vstack(v),axis=0) for k,v in datacount.items() if len(v)>0}
        for k,vl in datacount.items():
            for i,v in enumerate(vl):
                print(f"{self.mode} {k[-2:]} {['Normal','abnormal'][i]} {v}")
        # concat data
        negativeAug=[np.concatenate(na) for na in negativeAug if len(na)>0]
        traindata, trainlabels,postiveAug=np.concatenate(traindata), np.concatenate(trainlabels),np.concatenate(postiveAug)
        postiveAug1,postiveAug2=np.concatenate(postiveAug1),np.concatenate(postiveAug2)
        Sl_Labels,NA_Labels=np.copy(trainlabels),np.copy(trainlabels)
        Sl_Labels[Sl_Labels == 1] = 0
        Sl_Labels[Sl_Labels == 2] = 1
        Sl_Labels[Sl_Labels == 3] = 1
        NA_Labels[NA_Labels == 2] = 0
        NA_Labels[NA_Labels == 3] = 1
        # change shape by moving axis
        traindata,postiveAug,negativeAug=np.rollaxis(traindata,2,1),np.rollaxis(postiveAug,2,1),[np.rollaxis(na,2,1) for na in negativeAug]
        postiveAug1, postiveAug2=np.rollaxis(postiveAug1,2,1),np.rollaxis(postiveAug2,2,1)
        # print(traindata.shape,postiveAug.shape,trainlabels.shape,)
        negativeAug=[traindata[Sl_Labels==l] for l in range(2)]+negativeAug
        # negativeAug=[traindata[Sl_Labels==l] for l in np.unique(Sl_Labels)]+negativeAug
        # shuffle
        idx = np.random.permutation(traindata.shape[0])
        traindata,postiveAug,postiveAug1,postiveAug2=traindata[idx],postiveAug[idx],postiveAug1[idx],postiveAug2[idx]
        trainlabels,Sl_Labels,NA_Labels=trainlabels[idx],Sl_Labels[idx],NA_Labels[idx]
        for na in negativeAug: np.random.shuffle(na)
        negativeAug = {i: [negativeAug[j] for j in range(len(negativeAug)) if i != j] for i in range(2)}
        negativeAug = {i: np.concatenate(na) if len(na)>0 else [] for i,na in negativeAug.items()}


        # Transformation If required
        if transform is not None:
            self.traindata, self.trainlabels=transform(traindata),  trainlabels
            self.postiveAug,self.negativeAug=transform(postiveAug),{k:transform(na) for k,na in  negativeAug.items()}
            self.postiveAug12=[transform(postiveAug1),transform(postiveAug2)]
            self.Sl_Labels,self.NA_Labels=Sl_Labels,NA_Labels
        else:
            self.traindata, self.trainlabels=traindata,  trainlabels
            self.postiveAug,self.negativeAug=postiveAug,negativeAug
            self.postiveAug12=[postiveAug1,postiveAug2]
            self.Sl_Labels,self.NA_Labels=Sl_Labels,NA_Labels
        print(self.traindata.shape, self.trainlabels.shape,self.postiveAug.shape,{k:v.shape for k,v in self.negativeAug.items()})
        Labels=["S Normal","S Abnormal","L Normal","L AbNormal"]
        countlabel=list(zip(*np.unique(self.trainlabels, return_counts=True)))
        print(self.path.replace("_normalized","")," have Total ",self.trainlabels.shape[0],*[(Labels[di],c) for di,c in countlabel])


    def __getitem__(self, index,mode="train",s=0):
        if isinstance(index, np.float64):
            index = index.astype(np.int64)
        if mode=="train":
            if self.negativeIndexies is None:
                if len(self.negativeAug[self.Sl_Labels[index]]) > 0:
                    negaugs=self.negativeAug[self.Sl_Labels[index]]
                    negativeSample=negaugs[random.randint(0,negaugs.shape[0])-1]
                else: negativeSample=[]
            else:
                negativeSample=self.traindata[random.choice(self.negativeIndexies)]
            return self.traindata[index], self.postiveAug[index],negativeSample,self.trainlabels[index],self.Sl_Labels[index],self.NA_Labels[index],index
        else:
            return self.traindata[index], self.postiveAug12[s][index],  self.Sl_Labels[index], index


    def __len__(self):
        return self.traindata.shape[0]

