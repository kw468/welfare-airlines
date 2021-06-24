"""
    This script creates the summary statistics table in
    "The Welfare Effects of Dynamic Pricing:Evidence from Airline Markets"
    It also calculates various statistics that appear in the main text.
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

import numpy as np
import pandas as pd
import sys
import statsmodels.formula.api as smf

INPUT = "../../data"
OUTPUT = "../../output"

# -------------------------------------------------------------------------------
# IMPORT DATA
# -------------------------------------------------------------------------------

df = pd.read_parquet(f"{INPUT}/efdata_clean.parquet")
df_n = pd.read_parquet(f"{INPUT}/asdata_clean.parquet")

df_n["lf"] = (df_n.capY- df_n.sY) / df_n.capY # adjust to look at coach only

main_columns = ["origin", "dest", "flightNum", "tdate", "ddate", "fare", "seats", "lf"]
df = df[main_columns]
df_n["seats"] = df_n.sY # adjust to look at coach only
df = df.append(df_n[main_columns])


cols = ["origin", "dest", "flightNum", "ddate"]
df["difS"] = df.groupby(cols).seats.shift(-1) - df.seats
df["difP"] = df.groupby(cols).fare.shift(-1) - df.fare


# LB = df.groupby(["origin", "dest"]).difS.quantile(.0025).reset_index(drop=False).rename(columns = {"difS" : "LB"})
# UB = df.groupby(["origin", "dest"]).difS.quantile(.9975).reset_index(drop=False).rename(columns = {"difS" : "UB"})

# df = df.merge(LB, how = "inner", on = ["origin", "dest"])
# df = df.merge(UB, how = "inner", on = ["origin", "dest"])


# df = df.loc[(df.difS > df.LB) | (df.difS.isnull())]
# df = df.loc[(df.difS < df.UB) | (df.difS.isnull())]

#print sell outs
(df.groupby(cols).lf.max() == 1).sum() / (df.groupby(cols).lf.max()).shape[0]
#0.157191187391699

df["difS"] = -df.difS

# df_n[df_n.sY == 0][["origin", "dest", "flightNum", "ddate"]].drop_duplicates().shape
# #(853, 4)
# df_n[df_n.lf == 1][["origin", "dest", "flightNum", "ddate"]].drop_duplicates().shape
# #(230, 4)
# df_n[["origin", "dest", "flightNum", "ddate"]].drop_duplicates().shape
# #(9333, 4)
# 853/9333
# #0.09139612129004607

# replace time until departure variable from -60,0 to 0,60
df["ttdate"] = -df["tdate"] + 60


cols = ["origin", "dest", "ddate", "flightNum", "tdate"]

#df["ddate"] = df["ddate"].astype("category").cat.codes

df = df.sort_values(cols, ascending = False).reset_index(drop = True)

dfR = pd.read_csv(f"{INPUT}/airline_routes.csv", sep = "\t", header = None)
dfR[0] = dfR[0].str.strip()
dfR[1] = dfR[1].str.strip()
dfR[0] = dfR[0].astype("str")
dfR[1] = dfR[1].astype("str")
dfR .rename(
    columns = {0 : "origin", 1 : "dest", 2 : "year", 3 : "comp"},
    inplace = True
)
df = df.merge(dfR, on = ["origin", "dest"], how = "left")

#df = df.loc[df.comp <= 2]


# cols = ["origin", "dest", "flightNum", "ddate"]
# df["difS"] = df.groupby(cols).seats.shift(-1) - df.seats
# df["difP"] = df.groupby(cols).fare.shift(-1) - df.fare

# df.loc[df.lf1.isnull(), "lf1"] = df.lf

print("number of complete flights")
df[["origin", "dest", "flightNum", "ddate"]].drop_duplicates().shape
#(12119, 4)
print("number of obs")
df.shape
#(738625, 13)
print("LF by market")
(df.loc[df.tdate == 0].groupby(["origin", "dest"]).lf.mean()).describe()
# count    56.000000
# mean      0.898386
# std       0.056810
# min       0.700933
# 25%       0.862637
# 50%       0.916964
# 75%       0.942294
# max       0.982134
# Name: lf, dtype: float64

print("initial LF")
df.loc[df.tdate >= 59].lf.describe()


print("seat maps do not change")
(df.loc[df.difS == 0].shape[0]) / df.shape[0]
#0.6101919106447792
print("number of p changes")
(df.loc[df.difP != 0].shape[0]) / df.shape[0]
#0.169884582839736
print("number of p changes by flight")
df["indP"] = df.difP != 0
(df.groupby(["origin", "dest", "flightNum", "ddate"]).indP.sum()).describe()

# count    12119.000000
# mean        10.354072
# std          4.994251
# min          1.000000
# 25%          6.000000
# 50%          9.000000
# 75%         13.000000
# max         32.000000
# Name: indP, dtype: float64

cols = ["origin", "dest", "flightNum", "ddate"]
df2 = df.groupby(cols).fare.nunique().reset_index(drop = False)
df1 = df.loc[df.tdate == 0].reset_index(drop = True)

def meanSummaryStats(df,df1,df2):
    a = [df.fare.mean(), df.fare.std(), df.fare.median(), df.fare.quantile(.05), df.fare.quantile(.95)]
    b = [100 * c for c in [df1.lf.mean(), df1.lf.std(), df1.lf.median(), df1.lf.quantile(.05), df1.lf.quantile(.95)]]
    d = [df.difP.mean(), df.difP.std(), df.difP.median(), df.difP.quantile(.05), df.difP.quantile(.95)]
    e = [df.difS.mean(), df.difS.std(), df.difS.median(), df.difS.quantile(.05), df.difS.quantile(.95)]
    g = [df2.fare.mean(), df2.fare.std(), df2.fare.median(), df2.fare.quantile(.05), df2.fare.quantile(.95)]
    
    p1 = "Oneway Fare (\$)" + " ".join(["&" + str("{0:.2f}".format(f)) for f in a]) +  "\\\\[.5ex]"
    p2 = "Load Factor" + " ".join(["&" + str("{0:.2f}".format(f)) for f in b]) +  "\\\\[.5ex]"
    p3 = "Booking Rate" + " ".join(["&" + str("{0:.2f}".format(f)) for f in e]) +  "\\\\[.5ex]"
    p4 = "Daily Fare Change (\$)" + " ".join(["&" + str("{0:.2f}".format(f)) for f in d]) +  "\\\\[.5ex]"
    p5 = "Unique Fares (per itin.)" + " ".join(["&" + str("{0:.2f}".format(f)) for f in g]) +  "\\\\[.5ex]"
    
    with open(f"{OUTPUT}/summaryStatsTable.txt", "w") as f:
        f.writelines([line + "\n" for line in [p1,p2,p3,p4,p5]])


meanSummaryStats(df, df1, df2)

df["year"] = pd.to_datetime(df.ddate, format = "%Y-%m-%d").dt.year
df.groupby("year")[["fare", "difP", "difS"]].mean()
df1["year"] = pd.to_datetime(df1.ddate, format = "%Y-%m-%d").dt.year
df1.groupby("year")[["lf"]].mean()
df2["year"] = pd.to_datetime(df2.ddate, format = "%Y-%m-%d").dt.year
df2.groupby("year")[["fare"]].mean()

# >>> df["year"] = pd.to_datetime(df.ddate, format = "%Y-%m-%d").dt.year
# >>> df.groupby("year")[["fare", "difP", "difS"]].mean()
#             fare      difP      difS
# year                                
# 2012  364.483868  3.588136  0.851086
# 2019  193.185241  3.376421  0.625303
# >>> df1["year"] = pd.to_datetime(df1.ddate, format = "%Y-%m-%d").dt.year
# >>> df1.groupby("year")[["lf"]].mean()
#             lf
# year          
# 2012  0.945527
# 2019  0.870363
# >>> df2["year"] = pd.to_datetime(df2.ddate, format = "%Y-%m-%d").dt.year
# >>> df2.groupby("year")[["fare"]].mean()
#           fare
# year          
# 2012  7.387294
# 2019  6.847316
# >>> 
# >>> 

df1["f"] = df1[["origin", "dest"]].min(axis=1) # first OD pair
df1["s"] = df1[["origin", "dest"]].max(axis=1) # second OD pair
df1.groupby(["f", "s"]).lf.std() / df1.groupby(["f", "s"]).lf.mean()
# f    s  
# AUS  BOS    0.063373
# BIL  SEA    0.098930
# BOI  PDX    0.192106
# BOS  JAX    0.052566
#      MCI    0.107967
#      PDX    0.061678
#      SAN    0.037041
#      SEA    0.059649
# BZN  PDX    0.159525
# CHS  SEA    0.096122
# CMH  SEA    0.065500
# FAT  PDX    0.110919
# GEG  PDX    0.226779
# GTF  SEA    0.148844
# HLN  SEA    0.050335
# ICT  SEA    0.053466
# LIH  PDX    0.152257
# MSO  PDX    0.060611
# OKC  SEA    0.046572
# OMA  SEA    0.078021
# PDX  PSP    0.267332
#      RNO    0.177032
#      SBA    0.093676
#      SMF    0.099318
#      STS    0.257827
# SBA  SEA    0.099596
# SEA  STS    0.097030
#      SUN    0.068763
# Name: lf, dtype: float64
(df1.groupby(["f", "s"]).lf.std() / df1.groupby(["f", "s"]).lf.mean()).min()
(df1.groupby(["f", "s"]).lf.std() / df1.groupby(["f", "s"]).lf.mean()).max()

df1["r"] = df1.f  + df1.s
df1["ddate"] = df1.ddate.astype("str")
df1["flightNum"] = df1.flightNum.astype("str")

res = smf.ols(formula="lf ~ C(ddate) + C(r) + C(flightNum)", data=df1).fit()
res.summary()
# >>> res.summary()
# <class "statsmodels.iolib.summary.Summary">
# # """
#                             OLS Regression Results                            
# ==============================================================================
# Dep. Variable:                     lf   R-squared:                       0.555
# Model:                            OLS   Adj. R-squared:                  0.538
# Method:                 Least Squares   F-statistic:                     33.33
# Date:                Tue, 18 May 2021   Prob (F-statistic):               0.00
# Time:                        11:20:35   Log-Likelihood:                 11961.
# No. Observations:               12114   AIC:                        -2.305e+04
# Df Residuals:                   11676   BIC:                        -1.980e+04
# Df Model:                         437                                         
# Covariance Type:            nonrobust                                         

df2 = df.loc[df.tdate == 59]
df2["f"] = df2[["origin", "dest"]].min(axis=1) #first OD pair
df2["s"] = df2[["origin", "dest"]].max(axis=1) #second OD pair
df2.groupby(["f", "s"]).lf.std() / df2.groupby(["f", "s"]).lf.mean()

# f    s  
# AUS  BOS    0.175498
# BIL  SEA    0.283387
# BOI  PDX    0.360681
# BOS  JAX    0.245685
#      MCI    0.439568
#      PDX    0.199887
#      SAN    0.149529
#      SEA    0.210151
# BZN  PDX    0.397357
# CHS  SEA    0.220998
# CMH  SEA    0.222944
# FAT  PDX    0.328304
# GEG  PDX    0.302396
# GTF  SEA    0.340003
# HLN  SEA    0.161493
# ICT  SEA    0.235231
# LIH  PDX    0.254010
# MSO  PDX    0.197131
# OKC  SEA    0.288875
# OMA  SEA    0.202060
# PDX  PSP    0.440085
#      RNO    0.430187
#      SBA    0.263164
#      SMF    0.423073
#      STS    0.422166
# SBA  SEA    0.224373
# SEA  STS    0.265835
#      SUN    0.209187
# Name: lf, dtype: float64

df.groupby(["origin", "dest"]).fare.nunique()

# origin  dest
# AUS     BOS      40
# BIL     SEA      33
# BOI     PDX      33
# BOS     AUS      40
#         JAX      39
#         MCI      79
#         PDX      51
#         SAN      34
#         SEA      40
# BZN     PDX      39
# CHS     SEA      36
# CMH     SEA      54
# FAT     PDX      33
# GEG     PDX      29
# GTF     SEA      32
# HLN     SEA      25
# ICT     SEA     104
# JAX     BOS      41
# LIH     PDX      40
# MCI     BOS      84
# MSO     PDX      25
# OKC     SEA      90
# OMA     SEA     106
# PDX     BOI      32
#         BOS      49
#         BZN      37
#         FAT      33
#         GEG      29
#         LIH      26
#         MSO      27
#         PSP      43
#         RNO      42
#         SBA      29
#         SMF      58
#         STS      28
# PSP     PDX      50
# RNO     PDX      43
# SAN     BOS      34
# SBA     PDX      26
#         SEA      28
# SEA     BIL      33
#         BOS      39
#         CHS      34
#         CMH      48
#         GTF      31
#         HLN      29
#         ICT     116
#         OKC      88
#         OMA     105
#         SBA      30
#         STS      25
#         SUN      33
# SMF     PDX      56
# STS     PDX      28
#         SEA      25
# SUN     SEA      34
# Name: fare, dtype: int64
