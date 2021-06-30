
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import os

# --------------------------------------------------------------------------------
# Set program parameters
# --------------------------------------------------------------------------------


# --------------------------------------------------------------------------------
# Set path for data and logs
# --------------------------------------------------------------------------------
pathData = "/gpfs/home/kw468/airlines_jmp/"
pathOutput = "/gpfs/home/kw468/airlines_jmp/output/"


os.chdir(pathData)
from estim_markets import *



X = []
for m in mkts:
	print(m)
	tmp   	= pd.read_parquet(pathData  + "estimation/" + m + "/" + m + "stochLim_counterfactuals.parquet")
	tmp 	= tmp.loc[tmp.initC != 0]
	tmp 	= tmp.groupby(["initC", "fl"]).salesD.sum().reset_index()
	tmp 	= tmp.groupby("initC").salesD.mean()
	pctchange 	= tmp.pct_change()
	thres   = np.argmax((np.abs(pctchange) < .005) & (pctchange > 0))
	if thres == 0:
		thres = len(pctchange)
	data 	= pd.read_csv(pathData  + "estimation/" + m + "/" + m + ".csv")
	data 	= data.loc[data.tdate == 0]
	X.extend([(data.seats > thres).values])


np.sum([x.sum() for x in X]) / np.sum([len(x)for x in X])