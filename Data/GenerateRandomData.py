import json
import os,sys,scipy
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import norm, laplace,expon,cauchy,logistic,t,gamma,gaussian_kde
from sklearn.preprocessing import StandardScaler,MinMaxScaler
# ,"normal_gamma_data","gamma_laplace_data"]
distibutionSmall="normal"
distibutionLarge="laplace"
# distibutionSmall="gamma"
# distibutionLarge="laplace"
np.random.seed(142)


Double=False
df=10
a=2
disdiff, nadiff = 0, 0
# dataInfo={"DataName":DataName,"Diff":{"disdiff":disdiff,"nadiff":nadiff}, "Double": Double, "df": df, "a": a, "seed": seed,"datasize":datasize, "Dist1NormalList": dist1normallist, "Dist1AbnormalList": dist1abnormallist,"Dist2NormalList": dist2normallist, "Dist2AbnormalList": dist2abnormallist,"distlist":distlist}

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

