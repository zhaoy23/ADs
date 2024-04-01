import argparse,sys
from Supervised import Supervisedvalidation,ADSIndexices,RandomIndexices
from ActiveLearninig import ProcessData


def getArguments():
	conepoch, uncertainepoch = 2, 200
	# Parse the command line arguments
	spmethods=["Supervised","RandomS","RandomSL","ADS","ADSNonCL"]
	parser = argparse.ArgumentParser(prog='Active Learning Supervised Learning',description='It Do the Active Learning based on contrastive and uncertianity')
	parser.add_argument('-m', '--method',default="Supervised",choices=spmethods,type=str,help="Method to use")
	parser.add_argument('-id', '--initdata', default=[20], nargs="+", type=int, help="Percentage of Inital Data")
	parser.add_argument('-cy', '--cycle', default=[8], nargs="+", type=int, help="Number Of Cycles")
	parser.add_argument('-tp', '--totalpoints', default=400, type=int, help="Number Of Total Point to select")
	parser.add_argument('-dc', '--disableContrastive', action="store_true",help="Whether You want to disable the contrastivelearning for experiment")
	parser.add_argument('-se', '--supervisedepochs',default=400,type=int,help="Epoches for Supervised")
	parser.add_argument('-ep', '--baseepoches', default=1, type=int, help="Number Of Base Epoces with different seeds")
	parser.add_argument('-ce', '--contrastiveEpoches', default=conepoch, type=int,help="Number Of Epoces For Contrastive Lerarning")
	parser.add_argument('-ue', '--uncertainityEpoches', default=uncertainepoch, type=int,help="Number Of Epoces For Uncertainity Lerarning")
	args = parser.parse_args()
	print(args)
	return args,spmethods

def DoActiveLearning(args):
	initdatalist, cyclelist, totalpoints, baseepoches, contrastiveEpochesin, uncertainityEpoches, disableContrastive = args.initdata, args.cycle, args.totalpoints, args.baseepoches, args.contrastiveEpoches, args.uncertainityEpoches, args.disableContrastive
	assert all([totalpoints % cycle == 0 for cycle in cyclelist]), " Cycle Should be divisible by toral point"
	selectPointlist = [totalpoints // cycle for cycle in cyclelist]
	return ProcessData(selectPointlist[0], cyclelist[0], initdatalist[0], baseepoches, contrastiveEpochesin,uncertainityEpoches, disableContrastive)


if __name__ == '__main__':
	args, methods = getArguments()
	if args.method == "Supervised":
		Supervisedvalidation(epochs=args.supervisedepochs)
	elif args.method == "ADS": #"ADSCon":
		DoActiveLearning(args)
	elif args.method == "ADSNonCL":
		args.disableContrastive = True
		DoActiveLearning(args)
	elif args.method == "RandomS":
		RandomIndexices(ltype="S", epochs=args.supervisedepochs)
	elif args.method == "RandomSL":
		RandomIndexices(ltype="SL", epochs=args.supervisedepochs)
	else:
		print("Allowed methods are ", *methods)


