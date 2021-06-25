"""
    This script plots the estimated arrival processes in 
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
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob
import re
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import os

pathData                = "/home/kw468/Projects/airlines_jmp/"
pathIn                  = "/home/kw468/Projects/airlines_jmp/estimation/"
pathOutput              = "/home/kw468/Projects/airlines_jmp/output/"

os.chdir(pathData)
from estim_markets import *



# -------------------------------------------------------------------------------
# DEFINE FUNCTIONS FOR DEMAND
# -------------------------------------------------------------------------------

routeDirs 	= glob.glob(pathIn + "*_*")
routes 		= [re.split("/",f)[-1] for f in routeDirs]

routes 		= [r for r in routes if r in mkts]

paramFiles 	= [f + "_robust_params.csv" for f in routes]
dataFiles 	= [f + ".csv" for f in routes]
priceFiles 	= [f + "_prices.csv" for f in routes]
T = 60




# gather the parameters for a given market
def processRoute(num):
    VAR = np.genfromtxt(pathIn + "/" + routes[num] + "/robust_estim/" + paramFiles[num])
    df                  = 		pd.read_csv(pathIn + routes[num]  + "/" + routes[num] + ".csv", index_col=0)
    prices              = 		np.genfromtxt(pathIn + routes[num]  + "/" + routes[num] + "_prices.csv")
    beta                =       np.array(VAR[0:7])
    bL                  =       np.minimum(VAR[7], VAR[8])
    bB                  =       np.maximum(VAR[7], VAR[8])
    gamma               =       1/(np.exp(-VAR[9] - np.arange(0,60)*VAR[10] - (np.arange(0,60)**2)*VAR[11]) + 1)
    # equivalent to jnp.array([1/(1 + jnp.exp(-g[0] + -t*g[1] - t**2*g[2])) for t in range(0,60)])#range(int(min(Tdata)),int(max(Tdata)+1))])
    muT                 =       np.array([VAR[12]]*(T-20)+[VAR[13]]*7 + [VAR[14]]*7 + [VAR[15]]*6 )
    muD                 =       np.append(np.array([1]),np.array(VAR[16:22]))
    mu                  =       muT[:,None]*muD[None,:]
    sig                 =       VAR[-1]
    sl                  = np.exp(beta[df.dd_dow.values] + bL * prices[df.fareI.values]) / (1 + np.exp(beta[df.dd_dow.values] + bL * prices[df.fareI.values]) )
    sb                  = np.exp(beta[df.dd_dow.values] + bB * prices[df.fareI.values]) / (1 + np.exp(beta[df.dd_dow.values] + bB * prices[df.fareI.values]) )
    elas                = (bL * prices[df.fareI.values] * (1-sl))*(1-gamma[df.tdate.values]) + (bB * prices[df.fareI.values] * (1-sb))*gamma[df.tdate.values] 
    return [beta, bL, bB, mu, gamma]




# stack the parameters into df
X = []
for it in range(len(routes)):
    X.append(processRoute(it))

X = np.array(X)


def meanSummaryStats(X):
#
X1 = [z[0] for z in X]
X2 = [z[1] for z in X]
X3 = [z[2] for z in X]
X4 = [z[3] for z in X]
X5 = [z[4] for z in X]
#
X1 = np.array(X1)
X2 = np.array(X2)
X3 = np.array(X3)
X4 = np.array(X4)
X5 = np.array(X5)
#
a = X1.mean(), X1.std(), np.quantile(X1, .5), np.quantile(X1, .25), np.quantile(X1, .75)
b = X2.mean(), X2.std(), np.quantile(X2, .5), np.quantile(X2, .25), np.quantile(X2, .75)
c = X3.mean(), X3.std(), np.quantile(X3, .5), np.quantile(X3, .25), np.quantile(X3, .75)
d = X4.mean(), X4.std(), np.quantile(X4, .5), np.quantile(X4, .25), np.quantile(X4, .75)
e = X5.mean(), X5.std(), np.quantile(X5, .5), np.quantile(X5, .25), np.quantile(X5, .75)
#
p1 = "DoW Preferences" + " ".join(["&" + str("{0:.2f}".format(f)) for f in a]) +  "\\\\[.5ex]"
p2 = "Leisure Price Sensitivity" + " ".join(["&" + str("{0:.2f}".format(f)) for f in b]) +  "\\\\[.5ex]"
p3 = "Business Price Sensitivity" + " ".join(["&" + str("{0:.2f}".format(f)) for f in c]) +  "\\\\[.5ex]"
p4 = "Prob(Business)" + " ".join(["&" + str("{0:.2f}".format(f)) for f in e]) +  "\\\\[.5ex]"
p5 = "DoW Arrival Rates" + " ".join(["&" + str("{0:.2f}".format(f)) for f in d]) +  "\\\\[.5ex]"
    
with open(pathOutput + 'demand_sum.txt', 'w') as f:
    f.writelines([line + "\n" for line in [p1,p2,p3,p4,p5]])






# gather the parameters for a given market
def calcElas(num):
    VAR = np.genfromtxt(pathIn + "/" + routes[num] + "/robust_estim/" + paramFiles[num])
    df                  =       pd.read_csv(pathIn + routes[num]  + "/" + routes[num] + ".csv", index_col=0)
    prices              =       np.genfromtxt(pathIn + routes[num]  + "/" + routes[num] + "_prices.csv")
    beta                =       np.array(VAR[0:7])
    bL                  =       np.minimum(VAR[7], VAR[8])
    bB                  =       np.maximum(VAR[7], VAR[8])
    gamma               =       1/(np.exp(-VAR[9] - np.arange(0,60)*VAR[10] - (np.arange(0,60)**2)*VAR[11]) + 1)
    # equivalent to jnp.array([1/(1 + jnp.exp(-g[0] + -t*g[1] - t**2*g[2])) for t in range(0,60)])#range(int(min(Tdata)),int(max(Tdata)+1))])
    muT                 =       np.array([VAR[12]]*(T-20)+[VAR[13]]*7 + [VAR[14]]*7 + [VAR[15]]*6 )
    muD                 =       np.append(np.array([1]),np.array(VAR[16:22]))
    mu                  =       muT[:,None]*muD[None,:]
    sig                 =       VAR[-1]
    sl                  = np.exp(beta[df.dd_dow.values] + bL * prices[df.fareI.values]) / (1 + np.exp(beta[df.dd_dow.values] + bL * prices[df.fareI.values]) )
    sb                  = np.exp(beta[df.dd_dow.values] + bB * prices[df.fareI.values]) / (1 + np.exp(beta[df.dd_dow.values] + bB * prices[df.fareI.values]) )
    elas                = (bL * prices[df.fareI.values] * (1-sl))*(1-gamma[df.tdate.values]) + (bB * prices[df.fareI.values] * (1-sb))*gamma[df.tdate.values] 
    result              = pd.DataFrame({"elas" : elas})
    result["t"]         = df.tdate.values
    result["p"]         = prices[df.fareI.values]
    result["market"]    = num
    return result


# stack the parameters into df
Y = []
for it in range(len(routes)):
    Y.append(calcElas(it))


df = pd.concat(Y)
df1 = df.groupby(["t", "market"]).elas.mean().reset_index()
df1.elas.describe()



# gather the parameters for a given market
def processArrival(num):
    VAR = np.genfromtxt(pathIn + "/" + routes[num] + "/robust_estim/" + paramFiles[num])
    df                  =       pd.read_csv(pathIn + routes[num]  + "/" + routes[num] + ".csv", index_col=0)
    prices              =       np.genfromtxt(pathIn + routes[num]  + "/" + routes[num] + "_prices.csv")
    beta                =       np.array(VAR[0:7])
    bL                  =       np.minimum(VAR[7], VAR[8])
    bB                  =       np.maximum(VAR[7], VAR[8])
    gamma               =       1/(np.exp(-VAR[9] - np.arange(0,60)*VAR[10] - (np.arange(0,60)**2)*VAR[11]) + 1)
    # equivalent to jnp.array([1/(1 + jnp.exp(-g[0] + -t*g[1] - t**2*g[2])) for t in range(0,60)])#range(int(min(Tdata)),int(max(Tdata)+1))])
    muT                 =       np.array([VAR[12]]*(T-20)+[VAR[13]]*7 + [VAR[14]]*7 + [VAR[15]]*6 )
    muD                 =       np.append(np.array([1]),np.array(VAR[16:22]))
    mu                  =       muT[:,None]*muD[None,:]
    sig                 =       VAR[-1]
    return     (gamma[:,None]*mu).mean(1),     ((1-gamma[:,None])*mu).mean(1), mu


# stack the parameters into df
Z = []
for it in range(len(routes)):
    Z.append(processArrival(it))

#prob bus
np.array([z[0] for z in Z]).sum() / np.array([z[1] for z in Z]).sum()


np.array([z[2] for z in Z]).mean()



# gather the parameters for a given market
def wtp(num):
    VAR = np.genfromtxt(pathIn + "/" + routes[num] + "/robust_estim/" + paramFiles[num])
    df                  =       pd.read_csv(pathIn + routes[num]  + "/" + routes[num] + ".csv", index_col=0)
    prices              =       np.genfromtxt(pathIn + routes[num]  + "/" + routes[num] + "_prices.csv")
    beta                =       np.array(VAR[0:7])
    bL                  =       np.minimum(VAR[7], VAR[8])
    bB                  =       np.maximum(VAR[7], VAR[8])
    gamma               =       1/(np.exp(-VAR[9] - np.arange(0,60)*VAR[10] - (np.arange(0,60)**2)*VAR[11]) + 1)
    # equivalent to jnp.array([1/(1 + jnp.exp(-g[0] + -t*g[1] - t**2*g[2])) for t in range(0,60)])#range(int(min(Tdata)),int(max(Tdata)+1))])
    muT                 =       np.array([VAR[12]]*(T-20)+[VAR[13]]*7 + [VAR[14]]*7 + [VAR[15]]*6 )
    muD                 =       np.append(np.array([1]),np.array(VAR[16:22]))
    mu                  =       muT[:,None]*muD[None,:]
    sig                 =       VAR[-1]
    return bL, bB


# stack the parameters into df
W = []
for it in range(len(routes)):
    W.append(wtp(it))


np.quantile(( np.array(W)[:,0] - np.array(W)[:,1]   )   / np.array(W)[:,1], .5)


df = pd.concat(W)

df.wtp_B / df.wtp_L

df1 = df.groupby(["t", "market"]).elas.mean().reset_index()
df1.elas.describe()
