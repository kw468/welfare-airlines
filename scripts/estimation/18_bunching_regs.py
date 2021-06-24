# _________________________________________
# IMPORT REQUIRED PACKAGES
# _________________________________________

import numpy as np
import pandas as pd
import subprocess
import os

# _________________________________________
# DEFINE READ DATA FUNCTIONS
# _________________________________________


pathIn                  = "/home/kw468/Projects/airlines_jmp/"
pathOutput              = "/home/kw468/Projects/airlines_jmp/output/"

# _________________________________________
# DEFINE READ DATA FUNCTIONS
# _________________________________________

## so that AB, BA both become A-B
def determine_OD_Pair(D,O):
    return "_".join(sorted([D,O]))


df                    	= pd.read_parquet(pathIn + "efdata_clean.parquet")
df_n                    = pd.read_parquet(pathIn + "asdata_clean.parquet")

df_n["lf"]              = (df_n.capY- df_n.sY)/df_n.capY # adjust to look at coach only

df 						= df[["origin", "dest", "flightNum" ,"tdate", "ddate", "fare", "seats", "lf"]]
df_n["seats"] 			= df_n.sY # adjust to look at coach only
df 						= df.append(df_n[["origin", "dest", "flightNum", "tdate", "ddate", "fare", "seats", "lf"]])

dfR                     = pd.read_csv(pathIn + "airline_routes.csv", sep="\t", header=None)
dfR[0]                  = dfR[0].str.strip()
dfR[1]                  = dfR[1].str.strip()
dfR[0]                  = dfR[0].astype("str")
dfR[1]                  = dfR[1].astype("str")
dfR                     .rename(columns = {0 : "origin", 1 : "dest", 2 : "year", 3 : "comp"}, inplace=True )

df 						= df.merge(dfR, on = ["origin", "dest"], how = "left")

# replace time until departure variable from -60,0 to 0,60
df['ttdate'] 			= -df['tdate'] + 60

cols 					= ["origin", "dest", "ddate", "flightNum", "tdate"]

df                      = df.sort_values(cols, ascending = False).reset_index(drop=True)

cols 					= ["origin", "dest", "flightNum", "ddate"]
df["difS"] 				= df.groupby(cols).seats.shift(-1) - df.seats
df["difP"] 				= df.groupby(cols).fare.shift(-1) - df.fare


df['route'] 			= np.vectorize(determine_OD_Pair)(df['origin'], df['dest'])

df["DDmonth"] 			= df.ddate.dt.month
df["DOWS"] 				= df.ddate  -  pd.to_timedelta(df['tdate'], unit='d')
df["DOWS"] 				= df.DOWS.dt.dayofweek
df["DOWD"] 				= df.ddate.dt.dayofweek

df["APD3"]  			= 1*(df.tdate == 3)
df["APD7"]  			= 1*(df.tdate == 7)
df["APD14"]  			= 1*(df.tdate == 14)
df["APD21"]  			= 1*(df.tdate == 21)

df.to_csv(pathOutput + "subregs.csv")

dofile 					= pathOutput + "bunching_subRegressions.do"
cmd  					= ["stata", "-b", "do", dofile, "&"]
subprocess.call(cmd)


os.remove(pathOutput + "subregs.csv")

