"""
    This script calculates the incentives for passengers to wait to purchase in
    "The Welfare Effects of Dynamic Pricing:Evidence from Airline Markets"
--------------------------------------------------------------------------------
change log:
    v0.0.1  Mon 14 Jun 2021
-------------------------------------------------------------------------------
notes:
    
--------------------------------------------------------------------------------
contributors:
    Kevin:
        name:       Kevin Williams
        email:      kevin.williams@yale.edu
--------------------------------------------------------------------------------
Copyright 2021 Yale University
"""


# -------------------------------------------------------------------------------
# IMPORT REQUIRED PACKAGES
# -------------------------------------------------------------------------------


import numpy as np
import csv
import pandas as pd
import sys
import glob
import re 
import os
from estim_markets import *

# -------------------------------------------------------------------------------
# DEFINE PATHS
# -------------------------------------------------------------------------------

INPUT = "../../data"

os.chdir(INPUT)


routeDirs = glob.glob(INPUT + "/estimation/*_*")
routes = [re.split("/", f)[-1] for f in routeDirs]

routes = [r for r in routes if r in mkts]

paramFiles = [f + "_robust_params.csv" for f in routes]
dataFiles = [f + ".csv" for f in routes]
priceFiles = [f + "_prices.csv" for f in routes]
T = 60


# -------------------------------------------------------------------------------
# DEFINE FUNCTIONS
# -------------------------------------------------------------------------------


def processRoute(num):
	VAR = np.genfromtxt(INPUT + "/estimation/" + routes[num] + "/robust_estim/" + paramFiles[num])
	beta = np.array(VAR[0:7])
	bL = np.minimum(VAR[7], VAR[8])
	bB = np.maximum(VAR[7], VAR[8])
	gamma = 1 / (np.exp(
		-VAR[9] - np.arange(0, 60) * VAR[10] - (np.arange(0, 60) ** 2) * VAR[11]
	) + 1)
	# equivalent to jnp.array([1/(1 + jnp.exp(-g[0] + -t*g[1] - t**2*g[2])) for t in range(0,60)])
	# range(int(min(Tdata)),int(max(Tdata)+1))])
	muT = np.array(
		[VAR[12]] * (T - 20) + [VAR[13]] * 7 + [VAR[14]] * 7 + [VAR[15]] * 6
	)
	muD = np.append(np.array([1]), np.array(VAR[16:22]))
	mu = muT[:, None] * muD[None, :]
	sig = VAR[-1]
	A1 = mu * gamma[:, None]
	A2 = mu * (1 - gamma[:, None])
	return -bL, routes[num], bL / bB, A1.sum() / (A2.sum() + A1.sum()), mu.mean()

X = []
for it in range(len(routes)):
    X.append(list(processRoute(it)))

estims = pd.DataFrame(X)
estims.rename(
	columns = {1: "route", 0: "alpha", 2: "ratio", 3: "Aratio", 4: "mumean"},
	inplace = True
)

## so that AB, BA both become A-B
def determine_OD_Pair(D,O):
    return "_".join(sorted([D,O]))

# -------------------------------------------------------------------------------
# SIZE OF CHOICE SETS
# -------------------------------------------------------------------------------
Y = []
for it in range(len(routes)):
	lenn = pd.read_csv(
		INPUT + "/estimation/" + routes[it] + "/" + priceFiles[it]
	).shape[0]
	Y.extend([lenn])

min(Y)
max(Y)
#5, 11

estims["ratio"].mean()
estims["ratio"].median()
estims.Aratio.mean()

# >>> estims["ratio"].mean()
# 3.337667633207444
# >>> estims["ratio"].median()
# 2.2530864840364693
# >>> estims.Aratio.mean()
# 0.23320206718309763

# -------------------------------------------------------------------------------
# OPEN THE DATA
# -------------------------------------------------------------------------------


df = pd.read_parquet(INPUT + "efdata_clean.parquet")
df_n = pd.read_parquet(INPUT + "asdata_clean.parquet")

df_n["lf"] = (df_n.capY - df_n.sY) / df_n.capY # adjust to look at coach only

df = df[["origin", "dest", "flightNum" ,"tdate", "ddate", "fare", "seats", "lf"]]
df_n["seats"] = df_n.sY # adjust to look at coach only
df = df.append(
	df_n[["origin", "dest", "flightNum", "tdate", "ddate", "fare", "seats", "lf"]]
)

# replace time until departure variable from -60,0 to 0,60
df["ttdate"] = -df["tdate"] + 60

cols = ["origin", "dest", "ddate", "flightNum", "tdate"]
df["ddate"] = df["ddate"].astype("category").cat.codes

df = df.sort_values(cols, ascending = False).reset_index(drop = True)

cols = ["origin", "dest", "flightNum", "ddate"]
df["difS"] = df.groupby(cols).seats.shift(-1) - df.seats
df["difP"] = df.groupby(cols).fare.shift(-1) - df.fare


df["ind"] = 0
df.loc[df.difP < 0, "ind"] = 1
df["declines"] = df.difP
df.loc[df.difP >= 0, "declines"] = np.NaN

df["route"] = np.vectorize(determine_OD_Pair)(df["origin"], df["dest"])

# -------------------------------------------------------------------------------
# CALCULATE WAITING COSTS
# -------------------------------------------------------------------------------


#
df = df.merge(estims, on = ["route"], how = "inner")

#
df["declines"] = df.declines * df.alpha
#
small = df.groupby("tdate")[["ind","declines"]].mean()
small["phi"] = -small.ind * small.declines
small = small.reset_index(drop = False)
small .phi.fillna(0, inplace = True)
#
print("mean trans cost")
print(small.phi.mean())
print("median transaction cost")
print(small.phi.median())
print("describe")
print(small.phi.describe())


# >>> print("mean trans cost")
# mean trans cost
# >>> print(small.phi.mean())
# 5.748373203718175
# >>> print("median transaction cost")
# median transaction cost
# >>> print(small.phi.median())
# 5.8481116226918415
# >>> print("describe")
# describe
# >>> print(small.phi.describe())
# count    61.000000
# mean      5.748373
# std       1.545926
# min       0.000000
# 25%       4.813686
# 50%       5.848112
# 75%       6.697160
# max       9.159233
# Name: phi, dtype: float64
# >>> 
# >>> 
