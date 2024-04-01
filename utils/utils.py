import os,pickle,random,shutil,sys,stat #seaborn as sns
from datetime import datetime
import os.path,pandas as pd
import numpy as np
import torch
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter
from sklearn.decomposition import PCA

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

            self.set_dir(logdir)

    def _make_dir(self, fn):
        today = datetime.today().strftime("%y%m%d")
        logdir = 'logs/' + fn
        return logdir

    def set_dir(self, logdir, log_fn='log.txt'):
        self.logdir = logdir
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        self.writer = SummaryWriter(logdir)
        self.log_file = open(os.path.join(logdir, log_fn), 'a')

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


def save_checkpoint(epoch, model_state, optim_state, logdir,modelinfo=""):
    last_model = os.path.join(logdir, 'last_'+modelinfo+"_"+str(epoch)+'.model')
    last_optim = os.path.join(logdir, 'last_'+modelinfo+"_"+str(epoch)+'.optim')
    last_config = os.path.join(logdir, 'last_'+modelinfo+"_"+str(epoch)+'.config')
    opt = {'epoch': epoch}
    torch.save(model_state, last_model)
    torch.save(optim_state, last_optim)
    with open(last_config, 'wb') as handle:
        pickle.dump(opt, handle, protocol=pickle.HIGHEST_PROTOCOL)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
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

def saveResult(P,modelname,results,resultFolder,disableContrastive=False,disableUncertainiy=False):
    os.makedirs(resultFolder,exist_ok=True)
    OutFile=resultFolder+"Results_ALL.csv"
    if not os.path.exists(OutFile):
        with open(OutFile,"w") as f:
            f.write("Number of cycle,Number of selected per cycle,InitialData Percentage,Seed,disableContrastive,disableUncertainiy,ResultType,Results\n")
    result=[[P["NumberOfCycles"],P["target_Samples"],P["traindatapercentage"],P["seed"],str(disableContrastive),str(disableUncertainiy),"TrainingAcc"]+["%.2f"%(r) for r in results["trainaccuracylist"]],
            [P["NumberOfCycles"],P["target_Samples"],P["traindatapercentage"],P["seed"],str(disableContrastive),str(disableUncertainiy),"TestAcc"]+["%.2f"%(r) for r in results["testaccuracylist"]],
            [P["NumberOfCycles"],P["target_Samples"],P["traindatapercentage"],P["seed"],str(disableContrastive),str(disableUncertainiy),"L_Count_List"]+["%.2f"%(r) for r in results["l_count_list"]]]
    with open(OutFile,"a") as f:
        for res in result:
            f.write(",".join([str(r) for r in res])+"\n")

    OutFile=resultFolder+"Results.csv"
    if not os.path.exists(OutFile):
        with open(OutFile,"w") as f:
            f.write("Number of cycle,Number of selected per cycle,InitialData Percentage,Seed,disableContrastive,disableUncertainiy,TrainAcc,TestACC,Lcount\n")
    result=[P["NumberOfCycles"],P["target_Samples"],P["traindatapercentage"],P["seed"],str(disableContrastive),str(disableUncertainiy)]+["%.2f"%(results[lss][-1]) for lss in ["trainaccuracylist","testaccuracylist"]]+["%.2f"%(np.sum(results["l_count_list"]))]
    with open(OutFile,"a") as f:
        f.write(",".join([str(r) for r in result])+"\n")



def showResults(P,modelname,results,plotGraphs=False,resultFolder="Results",disableContrastive=False,disableUncertainiy=False):
    print("Saving results at",resultFolder)
    resultFolder+="/"
    saveResult(P,modelname,  results, resultFolder,disableContrastive,disableUncertainiy)
    if plotGraphs:
        os.makedirs(resultFolder+"Images",exist_ok=True)
        trainaccuracylist, testaccuracylist, l_count_list = [results[m] for m in ["trainaccuracylist", "testaccuracylist", "l_count_list"]]
        PlotGraph([trainaccuracylist,testaccuracylist], ["Train","Test"], ["Number of Al Cycle","Accuaracy(%)"], resultFolder+"Images/"+modelname+"_Accuracy.jpg")
        PlotGraph([l_count_list], ["Train"], ["Number of Al Cycle","number of selected L in each cycle"], resultFolder+"Images/"+modelname+"_L_Count.jpg")
        PlotErrorGraph(trainaccuracylist, testaccuracylist, resultFolder+"Images/"+modelname+"_Accuracy_Mean_Std.jpg")
        # PlotGraph([contrastivetrainloss], ["Train Loss"], ["Number of Iteration","Contrastive Loss"], resultFolder+"Images/"+modelname+"_Constrastive_Loss.jpg")
        # plt.show()


def saveModels(contrastiveModel,UncertainModel,model_name,modeldir):
    os.makedirs(modeldir, exist_ok=True)
    torch.save(contrastiveModel, modeldir+"Contrastive_"+model_name+".pth")
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

def get1DData(Data_set):
    # data = np.mean(np.stack([d[0] for d in Data_set])[:, :, -1],axis=1)
    # data = np.mean(np.stack([d[0] for d in Data_set]),axis=(1,2))
    data = np.stack([d[0] for d in Data_set])
    # data = data[:, :, -1]
    data = data.reshape((data.shape[0],-1))
    pca=PCA(n_components=1)
    data=pca.fit_transform(data)
    return data

def getTrainingIndicesByLatenHybercube(Data_set,datpercentage=20):
    data=get1DData(Data_set)
    data=list(zip(range(len(data)),data))
    indexes,_=getLatenHyperCubeSampels(data,datpercentage=datpercentage)
    return indexes.astype(np.int32)


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
