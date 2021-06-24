# Script: Alternative Stats in paper
# Program details:
# By:               Kevin Williams
# Modified:         12/14/2016


# _________________________________________
# IMPORT REQUIRED PACKAGES
# _________________________________________

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# -------------------------------------------------------------------------------
# DEFINE PATHS
# -------------------------------------------------------------------------------



pathIn                  = "/home/kw468/Projects/airlines_jmp/"
pathOutput              = "/home/kw468/Projects/airlines_jmp/output/"


# -------------------------------------------------------------------------------
# IMPORT DATA
# -------------------------------------------------------------------------------


df                      = pd.read_parquet(pathIn + "efdata_clean.parquet")
df_n                    = pd.read_parquet(pathIn + "asdata_clean.parquet")


df_n["lf"]              = (df_n.capY- df_n.sY)/df_n.capY # adjust to look at coach only


df = df[["origin", "dest", "flightNum" ,"tdate", "ddate", "fare", "seats", "lf", "capacity"]]
df_n["seats"] = df_n.sY # adjust to look at coach only
df_n["capacity"] = df_n.capY # add capacity
df = df.append(df_n[["origin", "dest", "flightNum", "tdate", "ddate", "fare", "seats", "lf", "capacity"]])


cols = ["origin", "dest", "flightNum", "ddate"]
df["difS"] = df.groupby(cols).seats.shift(-1) - df.seats
df["difP"] = df.groupby(cols).fare.shift(-1) - df.fare



dfR                     = pd.read_csv(pathIn + "airline_routes.csv", sep="\t", header=None)
dfR[0]                  = dfR[0].str.strip()
dfR[1]                  = dfR[1].str.strip()
dfR[0]                  = dfR[0].astype("str")
dfR[1]                  = dfR[1].astype("str")
dfR                     .rename(columns = {0 : "origin", 1 : "dest", 2 : "year", 3 : "comp"}, inplace=True )

df 						= df.merge(dfR, on = ["origin", "dest"], how = "left")

df_n 					= df_n.merge(dfR, on = ["origin", "dest"], how="inner")
df_n					= df_n.loc[df_n.comp <= 1]

# % of flights with first class
df_n.loc[df_n.capF.notnull()][["origin", "dest", "flightNum", "ddate"]].drop_duplicates().shape[0]/df_n[["origin", "dest", "flightNum", "ddate"]].drop_duplicates().shape[0]
#0.5836279867138112

# how important is first class
df_n.loc[df_n.capF.notnull()].capF.mean()
#11.99129333735303
df_n["capacity"] = df_n[["capF", "capY"]].sum(axis = 1)
df_n["fracCabin"] = df_n.capF/df_n.capacity
df_n.loc[df_n.capF.notnull()].fracCabin.mean()
#0.14352479603061152

# change in capacities
# df_o                    = pd.read_stata("/mnt/data0/airlines_jmp/data/cleaned/" + "master.dta")
# df_o                    = df_o.loc[df_o.flightnum2.isnull()]
# df_o                    = df_o.reset_index(drop=True)
# df_o["ones"]            = 1
# cols = ["origin", "dest", "ddate", "flightnum"]
# df_o["numObs"]          = df_o.groupby(cols)["ones"].transform("sum")
# df_o                    = df_o.loc[df_o.numObs >= 59]
# df_o["fare"]            = df_o["fare"]*1.12

#df1 = df_n.groupby(["origin", "dest", "flightNum", "ddate"])["capacity"].nunique().reset_index()
#df2 = df_o.groupby(["origin", "dest", "flightnum", "ddate"])["capacity"].nunique().reset_index()

df1 					= df.groupby(["origin", "dest", "flightNum", "ddate"])["capacity"].nunique().reset_index()

(df1.loc[df1.capacity > 1].shape[0])/(df1.shape[0])
#0.030117996534367523

# df1 = df.loc[df.tdate == 0 ].groupby(["origin", "dest", "flightNum", "ddate"])["capacity"].mean().reset_index()
# df2 = df.loc[df.tdate == 59].groupby(["origin", "dest", "flightNum", "ddate"])["capacity"].mean().reset_index()
# df1 = df1.merge(df2, on=["origin", "dest", "flightNum", "ddate"], how="inner")
# (df1.capacity_x != df1.capacity_y).sum()/df1.shape[0]
# #0.020683903252710592
# # (df1.loc[(df1.capacity_x != df1.capacity_y)].capacity_x  - df1.loc[(df1.capacity_x != df1.capacity_y)].capacity_y).mean()
# # #1.7137096774193548

# (df1.loc[(df1.capacity_x > df1.capacity_y)].capacity_x  - df1.loc[(df1.capacity_x > df1.capacity_y)].capacity_y).mean()
# #8.330708661417322
# (df1.loc[(df1.capacity_x < df1.capacity_y)].capacity_x  - df1.loc[(df1.capacity_x < df1.capacity_y)].capacity_y).mean()
# #-5.231404958677686

# changeEquip = df_n.copy()
# modes 		= changeEquip.groupby(["origin", "dest", "flightNum", "ddate"]).capacity.agg(pd.Series.mode).reset_index()
# modes 		= modes.rename(columns = {"capacity" : "modeCap"})
# changeEquip = changeEquip.merge(modes)
# changeEquip["indC"] = changeEquip["modeCap"] != changeEquip["capacity"]

changeEquip1 = df.copy()
modes1 		= changeEquip1.groupby(["origin", "dest", "flightNum", "ddate"]).capacity.agg(pd.Series.mode).reset_index()
modes1 		= modes1.rename(columns = {"capacity" : "modeCap"})
changeEquip1 = changeEquip1.merge(modes1)
changeEquip1["indC"] = changeEquip1["modeCap"] != changeEquip1["capacity"]


#changeSM = changeEquip.loc[changeEquip.indC == 1].groupby(["origin", "dest", "flightNum", "ddate"]).tdate.max().value_counts()
changeSM = changeEquip1.loc[changeEquip1.indC == 1].groupby(["origin", "dest", "flightNum", "ddate"]).tdate.max().value_counts()
# changeSM is overwhelming far from departure.
changeSM
# >>> changeSM
# 2     201
# 1      47
# 0      39
# 60     24
# 5      22
# 10      5
# 13      5
# 9       4
# 15      4
# 3       3
# 6       3
# 11      3
# 12      2
# 42      2
# 8       1
# Name: tdate, dtype: int64
# >>> 

changeSM.reset_index(drop = False).loc[changeSM.index <= 2].sum().tdate / changeSM.sum()
#0.7863013698630137

changes 			= df.copy()
changes 			= changes.sort_values(["origin", "dest", "flightNum", "ddate", "tdate"], ascending = False).reset_index(drop=True)
changes["difC"] 	= changes.groupby(["origin", "dest", "flightNum", "ddate"]).capacity.shift(1)
changes["difC_D"] 	= (changes.difC - changes.capacity > 0) & (changes.difC.notnull())
changes["difC_U"] 	= (changes.difC - changes.capacity < 0) & (changes.difC.notnull())

changes["elf"] 		= changes.groupby(["origin", "dest", "flightNum", "ddate"]).lf.transform("last")

means 				= changes[["origin", "dest", "flightNum", "ddate", "elf"]].drop_duplicates().groupby(["origin", "dest"]).elf.mean().reset_index()
means 				= means.rename(columns = {"elf" : "avg"})
changes 			= changes.merge(means, on = ["origin", "dest"])
changes["delLF"] 	= changes.elf - changes.avg

changes.loc[changes.difC_U == True].delLF.mean()
changes.loc[changes.difC_D == True].delLF.mean()

a = stats.ttest_rel(changes.loc[changes.difC_U == True].elf, changes.loc[changes.difC_U == True].avg)
b = stats.ttest_rel(changes.loc[changes.difC_D == True].elf, changes.loc[changes.difC_D == True].avg)

def t_test(x,y,alternative='both-sided'):
        _, double_p = stats.ttest_ind(x,y,equal_var = False)
        if alternative == 'both-sided':
            pval = double_p
        elif alternative == 'greater':
            if np.mean(x) > np.mean(y):
                pval = double_p/2.
            else:
                pval = 1.0 - double_p/2.
        elif alternative == 'less':
            if np.mean(x) < np.mean(y):
                pval = double_p/2.
            else:
                pval = 1.0 - double_p/2.
        return pval


print(t_test(changes.loc[changes.difC_U == True].elf,changes.loc[changes.difC_U == True].avg,alternative='greater'))
print(t_test(changes.loc[changes.difC_D == True].avg,changes.loc[changes.difC_D == True].elf,alternative='greater'))


# >>> changes.loc[changes.difC_U == True].delLF.mean()
# -0.0321720699993666
# >>> changes.loc[changes.difC_D == True].delLF.mean()
# -0.007989449720960586
# >>> stats.ttest_rel(changes.loc[changes.difC_U == True].elf, changes.loc[changes.difC_U == True].avg)
# Ttest_relResult(statistic=-5.509321255134468, pvalue=9.163242107189687e-08)
# >>> stats.ttest_rel(changes.loc[changes.difC_D == True].elf, changes.loc[changes.difC_D == True].avg)
# Ttest_relResult(statistic=-1.1843204139538213, pvalue=0.23747763601134167)
# >>>