"""
    This script plots the percentage fare in changes over time for each
    market separately in 
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
import matplotlib.pyplot as plt
import seaborn as sns


# -------------------------------------------------------------------------------
# DEFINE PATHS
# -------------------------------------------------------------------------------


pathIn                  = "/home/kw468/Projects/airlines_jmp/"
pathOutput              = "/home/kw468/Projects/airlines_jmp/output/"

# -------------------------------------------------------------------------------
# OPEN THE DATA / CLEAN OBS
# -------------------------------------------------------------------------------

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
df["ddate"] 			= df["ddate"].astype('category').cat.codes

df                      = df.sort_values(cols, ascending = False).reset_index(drop=True)

cols 					= ["origin", "dest", "flightNum", "ddate"]
df["difS"] 				= df.groupby(cols).seats.shift(-1) - df.seats
df["difP"] 				= df.groupby(cols).fare.shift(-1) - df.fare


df 						= df.loc[df.difS.notnull()]
df["seatC"] 			= 0
df 						.loc[df.difS < 0, "seatC"] = 1

df["route"] 			= df[["origin", "dest"]].min(axis=1) + df[["origin", "dest"]].max(axis=1) 



# -------------------------------------------------------------------------------
# CREATE PLOT GRID AND PLOT THE RESULTS
# -------------------------------------------------------------------------------


ncols = int(4)
nrows = int(np.ceil(df["route"].nunique()/ncols))
csfont 						= {'fontname':"Liberation Serif", 'fontsize':20}
palette 					= ["#FF6700", "#FCB07E", "#6B717E", "#3A6EA5", "#004E98", "#070707"]
sns.set(style="white",color_codes=False)
fig, axs = plt.subplots(nrows, ncols,figsize=(ncols*1.5*6.4, nrows*1.1*4.8))
fig.subplots_adjust(hspace=.4)
counter = 0
markets = sorted(list(df["route"].unique()))
for i in range(nrows - 1):
    for j in range(ncols):
        df1 = df.loc[df.route == markets[counter]]
        axs[i, j].plot(100*df1.groupby("ttdate").fare.mean().pct_change(),color = palette[4])   
        axs[i, j].set_ylim((-10,50))
        axs[i, j].set_title(markets[counter], fontsize=20, fontname="Liberation Serif")
        counter += 1

remain = len(markets) - counter
for j in range(remain):
    df1 = df.loc[df.route == markets[counter]]
    axs[nrows-1, j].plot(100*df1.groupby("ttdate").fare.mean().pct_change(),color = palette[4])  
    axs[nrows-1, j].set_ylim((-10,50))
    axs[nrows-1, j].set_title(markets[counter], fontsize=20, fontname="Liberation Serif")
    counter += 1

for ax in axs.flat:
    ax.set(xlabel="Booking Horizon", ylabel='Fare % Change')
    ax.xaxis.get_label().set_fontsize(20)
    ax.yaxis.get_label().set_fontsize(20)
    ax.xaxis.get_label().set_fontname("Liberation Serif")
    ax.yaxis.get_label().set_fontname("Liberation Serif")


numDelete = ncols - remain
for j in range(numDelete):
    fig.delaxes(axs[nrows-1][ncols-j-1])


plt.yticks(fontname = "Liberation Serif", fontsize = 20) 
plt.xticks(fontname = "Liberation Serif", fontsize = 20) 

name = "avg_fare_pct_change_allRoutes.pdf"
plt.savefig(pathOutput + name,bbox_inches='tight',format= "pdf",dpi=600)
plt.close()

