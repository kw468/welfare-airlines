"""
    This code solves estimates the model parameters in Williams (2021)
    Demand:
        * Discrete type random coefficients model
        * Consumers choose to buy or no
        * Consumers arrive according to a Poisson distribution
        * Hence, demand is Poisson.
        * Demand may be censored; random rationing is assumed
    Firm:
        * Firm knows the demand process and solves a dynamic logit model
        * The states are seats remaining and time left to sell
    LLN:
        * The log-likelihood is constructed based on demand transitions
          and the conditional choice probabilities
--------------------------------------------------------------------------------
last edit:
    Thurs 10 Feb 2021
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


# --------------------------------------------------------------------------------
# Import Required Packages 
# --------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import os

pathData                = "/home/kw468/Projects/airlines_jmp/"
pathOutput              = "/home/kw468/Projects/airlines_jmp/output/"

os.chdir(pathData)
from estim_markets import *


# --------------------------------------------------------------------------------
# Set path for data and logs
# --------------------------------------------------------------------------------
def transformNum(a):
    if np.abs(a) > .001 :
        ans = "{0:.3f}".format(a)
        ans = "$" + ans + "$"  
    else:
        ans = "{:.1E}".format(a)
    return ans


def stars(a,b):
    num = np.abs(a/b)
    if np.abs(b) >= .001:
        if num > 2.58 :
            ans = "{0:.3f}".format(b) + "^{***}"
        elif (num < 2.58 ) & (num > 1.96):
            ans = "{0:.3f}".format(b)  + "^{**}" 
        elif (num < 1.96) & (num > 1.65 ):
            ans = "{0:.3f}".format(b)  + "^{*}" 
        elif num < 1.65 :
            ans = "{0:.3f}".format(b)
        ans     = "$" + ans + "$" 
    else:
        if num > 2.58 :
            ans = "{:.1E}".format(b) + "$^{***}$"
        elif (num < 2.58 ) & (num > 1.96):
            ans = "{:.1E}".format(b)  + "$^{**}$" 
        elif (num < 1.96) & (num > 1.65 ):
            ans = "{:.1E}".format(b) + "$^{*}$" 
        elif num < 1.65 :
            ans = "{:.1E}".format(b)       
    return ans


def gatherMarketTable(market):
    pathIn                  = pathData + "estimation/" + market + "/"
    pathSE                  = pathData + "estimation/" + market + "/se/"
    df_route                = pd.read_csv(pathIn + market + ".csv", index_col=0)
    df_route_pt             = pd.read_csv(pathIn + market + "_Pt.csv", header = None)
    prices                  = np.array(np.genfromtxt(pathIn + market + "_prices.csv"))
    info                    = np.genfromtxt(pathIn + market + "_info.csv")
    Pt                      = df_route_pt.values
    qBar                    = int(np.max(df_route.seats))+1
    T                       = len(np.unique(df_route.tdate))
    numP                    = len(prices)
    obs                     = len(df_route.tdate)
    betaHat                 = np.genfromtxt(pathIn + "/robust_estim/" + market + "_robust_params.csv")
    flip = False
    if betaHat[7] > betaHat[8]:
        flip = True
    SEs                     = np.genfromtxt(pathSE + market + "_se.csv")
    tStat                   = np.genfromtxt(pathSE + market + "_tstat.csv")
    if flip == True:
        betaHat[7], betaHat[8] = betaHat[8], betaHat[7]
        SEs[7], SEs[8]         = SEs[8], SEs[7]
        tStat[7], tStat[8]     = tStat[8], tStat[7]
    n                       = [transformNum(a) for a in betaHat]
    d                       = [stars(a,b) for (a,b) in zip(betaHat,SEs)]
    result                  = [None]*(len(n)+len(d))
    result[::2]             = n
    result[1::2]            = d
    result                  = result + list(info)
    return result


def f1(x):
    return f'{x:,}'


def writeTable(markets,table,name):
    Z = [
    "\\begin{tabular}{l" + "".join(["r"]*(len(markets) + 1)) + "}",
    "\\toprule",
    "Variable &&" + "&".join([s.replace("_", "") for s in markets]) + "\\\\",
    "\\midrule",
    "\\underline{Logit Demand}&\\\\",
    "DoW Prefs& $\\beta^0$&" + "&".join(table[:,0]) + "\\\\",
    "&&"                    + "&".join(table[:,1]) + "\\\\",
    "&          $\\beta^1$&" + "&".join(table[:,2]) + "\\\\",
    "&&"                    + "&".join(table[:,3]) + "\\\\",
    "&          $\\beta^2$&" + "&".join(table[:,4]) + "\\\\",
    "&&"                    + "&".join(table[:,5]) + "\\\\",    
    "&          $\\beta^3$&" + "&".join(table[:,6]) + "\\\\",
    "&&"                    + "&".join(table[:,7]) + "\\\\",    
    "&          $\\beta^4$&" + "&".join(table[:,8]) + "\\\\",
    "&&"                    + "&".join(table[:,9]) + "\\\\",    
    "&          $\\beta^5$&" + "&".join(table[:,10]) + "\\\\",
    "&&"                    + "&".join(table[:,11]) + "\\\\",    
    "&          $\\beta^6$&" + "&".join(table[:,12]) + "\\\\",
    "&&"                    + "&".join(table[:,13]) + "\\\\",
    "Leis. Price Sens.& $\\alpha_L$&" + "&".join(table[:,14]) + "\\\\",
    "&&"                    + "&".join(table[:,15]) + "\\\\",
    "Bus. Price Sens.& $\\alpha_B$&" + "&".join(table[:,16]) + "\\\\",
    "&&"                    + "&".join(table[:,17]) + "\\\\", 
    "Pr(Bus.) Cons.& $\\gamma_0$&" + "&".join(table[:,18]) + "\\\\",
    "&&"                    + "&".join(table[:,19]) + "\\\\",
    "Pr(Bus.) Linear& $\\gamma_1$&" + "&".join(table[:,20]) + "\\\\",
    "&&"                    + "&".join(table[:,21]) + "\\\\",    
    "Pr(Bus.) Quad.& $\\gamma_2$&" + "&".join(table[:,22]) + "\\\\",
    "&&"                    + "&".join(table[:,23]) + "\\\\",
    "\\underline{Poisson Rates}&\\\\",
    "$>$ 21 Days& $\\mu_1$&" + "&".join(table[:,24]) + "\\\\",
    "&&"                    + "&".join(table[:,25]) + "\\\\",
    "14 to 21 days& $\\mu_2$&" + "&".join(table[:,26]) + "\\\\",
    "&&"                    + "&".join(table[:,27]) + "\\\\", 
    "7 to 14 days& $\\mu_3$&" + "&".join(table[:,28]) + "\\\\",
    "&&"                    + "&".join(table[:,29]) + "\\\\",
    "$<$ 7 days& $\\mu_4$&" + "&".join(table[:,30]) + "\\\\",
    "&&"                    + "&".join(table[:,31]) + "\\\\",
    "DoW Effect &$\\mu^1$&" + "&".join(table[:,32]) + "\\\\",
    "&&"                    + "&".join(table[:,33]) + "\\\\",
    "&           $\\mu^2$&" + "&".join(table[:,34]) + "\\\\",
    "&&"                    + "&".join(table[:,35]) + "\\\\",
    "&           $\\mu^3$&" + "&".join(table[:,36]) + "\\\\",
    "&&"                    + "&".join(table[:,37]) + "\\\\",
    "&           $\\mu^4$&" + "&".join(table[:,38]) + "\\\\",
    "&&"                    + "&".join(table[:,39]) + "\\\\",
    "&           $\\mu^5$&" + "&".join(table[:,40]) + "\\\\",
    "&&"                    + "&".join(table[:,41]) + "\\\\",
    "&           $\\mu^6$&" + "&".join(table[:,42]) + "\\\\",
    "&&"                    + "&".join(table[:,43]) + "\\\\",
    "\\underline{Firm Shock}&\\\\",
    "& $\\sigma$&" + "&".join(table[:,44]) + "\\\\",
    "&&"                    + "&".join(table[:,45]) + "\\\\",
    "\\midrule",            
    "LogLike &&"            + "&".join([f1(round(float(a))) for a in table[:,46]]) + "\\\\",
    "\\multicolumn{2}{l}{Number of Flights} &"       + "&".join([f1(round(float(a))) for a in table[:,47]])+ "\\\\",
    "\\multicolumn{2}{l}{Number of Dep. Dates} &"    + "&".join([f1(round(float(a))) for a in table[:,48]]) + "\\\\",
    "\\multicolumn{2}{l}{Number of Obs.} &"          + "&".join([f1(round(float(a))) for a in table[:,49]])+ "\\\\",
    "\\bottomrule",
    "\\end{tabular}",
    ]
    with open(pathOutput + 'param_ests_'+name+'.tex', 'w') as tf:
        for z in Z:
            tf.write(z + "\n")






# PDX_PSP
# LIH_PDX
# BOS_SEA # DUOPOLY
# BOS_PDX # DUOPOLY
# PDX_SMF # DUOPOLY
# OKC: numerical stability


chunks = np.array_split(np.array(mkts), 3)

table1    = np.array([gatherMarketTable(market) for market in chunks[0]])
table2    = np.array([gatherMarketTable(market) for market in chunks[1]])
table3    = np.array([gatherMarketTable(market) for market in chunks[2]])


writeTable(chunks[0],table1, '1')
writeTable(chunks[1],table2, '2')
writeTable(chunks[2],table3, '3')




def createDF(market):
    pathIn                  = pathData + "estimation/" + market + "/"
    pathSE                  = pathData + "estimation/" + market + "/se/"
    df_route                = pd.read_csv(pathIn + market + ".csv", index_col=0)
    df_route_pt             = pd.read_csv(pathIn + market + "_Pt.csv", header = None)
    prices                  = np.array(np.genfromtxt(pathIn + market + "_prices.csv"))
    info                    = np.genfromtxt(pathIn + market + "_info.csv")
    Pt                      = df_route_pt.values
    qBar                    = int(np.max(df_route.seats))+1
    T                       = len(np.unique(df_route.tdate))
    numP                    = len(prices)
    obs                     = len(df_route.tdate)
    betaHat                 = np.genfromtxt(pathIn + "/robust_estim/" + market + "_robust_params.csv")
    flip = False
    if betaHat[7] > betaHat[8]:
        flip = True
    SEs                     = np.genfromtxt(pathSE + market + "_se.csv")
    tStat                   = np.genfromtxt(pathSE + market + "_tstat.csv")
    if flip == True:
        betaHat[7], betaHat[8] = betaHat[8], betaHat[7]
        SEs[7], SEs[8]         = SEs[8], SEs[7]
        tStat[7], tStat[8]     = tStat[8], tStat[7]
    return betaHat



df = [createDF(c) for c in mkts]
df = pd.DataFrame(df)

df["ones"] = 1


np.corrcoef(df[[0,1,2,3,4,5,6]].mean().values, df[["ones",16,17,18,19,20,21]].mean().values)
# array([[ 1.        , -0.84948472],
#        [-0.84948472,  1.        ]])



>>> df[["ones", 16,17,18,19,20,21]].idxmax(1).value_counts()
# >>> df[["ones", 16,17,18,19,20,21]].idxmax(1).value_counts()
# 21      5
# ones    4
# 19      4
# 18      4
# 20      2
# 16      2
# 17      1
# dtype: int64
# >>> 

