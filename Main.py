import argparse,sys
import random
import numpy as np
from Supervised import Supervisedvalidation,ADSIndexices,RandomIndexices
from ActiveLearninig import ProcessData,norenew


def getArguments():
	conepoch, uncertainepoch = 2, 200
	DataNameList=["Megas","Megas_L","Random_1"]
	spmethods=["Supervised","RandomS","RandomL","RandomSL_S","RandomSL_L","ADS","ADSNonCL"]
	parser = argparse.ArgumentParser(prog='Active Learning Supervised Learning',description='It Do the Active Learning based on contrastive and uncertianity')
	parser.add_argument('-dn', '--DatasetName',default="Megas",type=str,choices=DataNameList,help="DataSet Name")
	parser.add_argument('-di', '--DataIndex', default=3, type=int, help="DataSet To Use")
	parser.add_argument('-d', '--datapath',default="../DATA/data/",type=str,help="DataSet To Use")
	parser.add_argument('-ds', '--dataSelection',default="S",choices=["S","L"],type=str,help="DataSet To Select")
	parser.add_argument('-m', '--method',default="Supervised",choices=spmethods,type=str,help="Method to use")
	parser.add_argument('-id', '--initdata', default=20, nargs="+", type=int, help="Percentage of Inital Data")
	parser.add_argument('-cy', '--cycle', default=[8], nargs="+", type=int, help="Number Of Cycles")
	parser.add_argument('-tp', '--totalpoints', default=800, type=int, help="Number Of Total Point to select")
	parser.add_argument('-dc', '--disableContrastive', action="store_true",help="Whether You want to disable the contrastivelearning for experiment")
	parser.add_argument('-se', '--supervisedepochs',default=200,type=int,help="Epoches for Supervised")
	parser.add_argument('-ep', '--baseepoches', default=1, type=int, help="Number Of Base Epoces with different seeds")
	parser.add_argument('-lr', '--learningrate', default=0.001, type=float, help="Learning rate to use")
	parser.add_argument('-bs', '--batch_size', default=128, type=int, help="Batch Size")
	parser.add_argument('-nc', '--numberconvolution', default=1, type=int, help="Number of convolution layers")
	parser.add_argument('-ce', '--contrastiveEpoches', default=conepoch, type=int,help="Number Of Epoces For Contrastive Lerarning")
	parser.add_argument('-ue', '--uncertainityEpoches', default=uncertainepoch, type=int,help="Number Of Epoces For Uncertainity Lerarning")
	args = parser.parse_args()
	args.seed=np.random.randint(0,1000000000)
	print(args)
	return args,spmethods

def DoActiveLearning(args):
	initdatalist, cyclelist, totalpoints, baseepoches, contrastiveEpochesin, uncertainityEpoches, disableContrastive,datapath = args.initdata, args.cycle, args.totalpoints, args.baseepoches, args.contrastiveEpoches, args.uncertainityEpoches, args.disableContrastive,args.datapath
	resultFolder = "Results/Result_" + datapath.split("/")[-2]
	if not norenew: resultFolder += "renew"
	assert all([totalpoints % cycle == 0 for cycle in cyclelist]), " Cycle Should be divisible by toral point"
	selectPointlist = [totalpoints // cycle for cycle in cyclelist]
	return ProcessData(datapath,selectPointlist[0], cyclelist[0], initdatalist[0], baseepoches, contrastiveEpochesin,uncertainityEpoches, disableContrastive,resultFolder=resultFolder)


if __name__ == '__main__':
	args, methods = getArguments()
	if args.method == "Supervised":
		args.dataSelection="L"
		print("Running Supervised code with following arguments ",args)
		Supervisedvalidation(args, DataPath=args.datapath)
	elif args.method == "ADS": #"ADSCon":
		DoActiveLearning(args)
	elif args.method == "ADSNonCL":
		args.disableContrastive = True
		DoActiveLearning(args)
	elif args.method == "RandomS":
		RandomIndexices(args,DataPath=args.datapath,ltype="S", )
	elif args.method == "RandomSL_S":
		print("Running RandomSL_S code with following arguments ",args)
		RandomIndexices(args,DataPath=args.datapath, ltype="SL", )
	elif args.method == "RandomL":
		print("Running RandomL code with following arguments ", args)
		args.dataSelection = "L"
		for e in range(args.baseepoches):
			RandomIndexices(args, DataPath=args.datapath, dataname="Megas_L", ltype="L", )
	elif args.method == "RandomSL_L":
		print("Running RandomSL_L code with following arguments ",args)
		args.dataSelection="L"
		RandomIndexices(args,DataPath=args.datapath,dataname="Megas_L",ltype="SL_L", )
	else:
		print("Allowed methods are ", *methods)


