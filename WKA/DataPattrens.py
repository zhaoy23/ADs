from utils.utils import set_random_seed

def getPattren(DataIndex):
	# Dataset Arguments
	if DataIndex==1:
		distibutionSmall, distibutionLarge = ['laplace'], 'logistic_expon'.split('_')
		distlist = '133_87__37_88__106_61	68_91__54_40__54_62 31_91__75_21__75_77  61_94__39_71__14_65'.split()
		lpercent, normalize, requiredSize, seed,LNASLA = 70, True, (12000, 1000), 37107087,False
	if DataIndex==2:
		distibutionSmall, distibutionLarge = ['laplace'], 'logistic_expon'.split('_')
		distlist = '133_87__37_88__106_61	68_91__54_40__54_62 31_91__75_21__75_77  61_94__39_71__14_65'.split()
		lpercent, normalize, requiredSize, seed,LNASLA = 70, True, (10000, 1000), 65307236,False
	if DataIndex==3:
		distibutionSmall, distibutionLarge = ['laplace'], 'logistic_expon'.split('_')
		distlist = '133_87__37_88__106_61	68_91__54_40__54_62 31_91__75_21__75_77  61_94__39_71__14_65'.split()
		lpercent, normalize, requiredSize, seed,LNASLA = 50, True, (10000, 1000), 65307236,False
	if DataIndex==4:
		distibutionSmall, distibutionLarge = ['laplace'], 'logistic_expon'.split('_')
		distlist = '133_87__37_88__106_61	68_91__54_40__54_62 31_91__75_21__75_77  61_94__39_71__14_65'.split()
		lpercent, normalize, requiredSize, seed,LNASLA = 50, True, (10000, 1000), 37107087,False

	if DataIndex==10:
		distibutionSmall, distibutionLarge = ['cauchy'], 'logistic_gamma_T_expon'.split('_')
		distlist = '116_97__128_50__87_97	60_42__114_21__24_34 107_54__114_95__71_77  127_91__56_45__131_34'.split()
		lpercent, normalize, requiredSize, seed, LNASLA = 40, False, (7000, 1000), 26007043, True
	return distibutionSmall, distibutionLarge,distlist,lpercent, normalize, requiredSize, seed,LNASLA

def getDataDetails(DataIndex=1):
	distibutionSmall, distibutionLarge, distlist, lpercent, normalize, requiredSize, seed,LNASLA=getPattren(DataIndex)
	dist1normallist = [[int(d) for d in dt.split("_")] for dt in distlist[0].split("__")]
	dist1abnormallist = [[int(d) for d in dt.split("_")] for dt in distlist[1].split("__")]
	dist2normallist = [[int(d) for d in dt.split("_")] for dt in distlist[2].split("__")]
	dist2abnormallist = [[int(d) for d in dt.split("_")] for dt in distlist[3].split("__")]
	distargs = [distibutionSmall, distibutionLarge, dist1normallist, dist1abnormallist, dist2normallist, dist2abnormallist]
	other= lpercent, normalize, requiredSize, seed,LNASLA
	dataname=f"{'_'.join(distibutionSmall)}__{'_'.join(distibutionLarge)}__{'___'.join(distlist)}__{normalize}__{lpercent}__{requiredSize[0]}_{requiredSize[1]}_{seed}_{LNASLA}"
	metadata=f"Dist1_Dist2_DistArguments_Normalize_Lpercent_DataSize_Seed_LSASNA"
	set_random_seed(seed)
	return distargs,other,dataname,metadata


