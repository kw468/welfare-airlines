"""
    This script plots the frequency and magnitude of fare changes in
    "The Welfare Effects of Dynamic Pricing:Evidence from Airline Markets"
    Both monopoly and duopoly markets are considered
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
import pandas as pd
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

df                      = df.sort_values(cols, ascending = False).reset_index(drop=True)

cols 					= ["origin", "dest", "flightNum", "ddate"]
df["difS"] 				= df.groupby(cols).seats.shift(-1) - df.seats
df["difP"] 				= df.groupby(cols).fare.shift(-1) - df.fare


df 						= df.loc[df.difP.notnull()]
df["fareC"] 			= 0
df 						.loc[df.difP > 0, "fareC"] = 1
df 						.loc[df.difP < 0, "fareC"] = 2


# -------------------------------------------------------------------------------
# CREATE THE PLOT
# -------------------------------------------------------------------------------



def plotFareChange(df,comp):
	df1 					= df.loc[df["comp"] == comp]
	df1 					= df1.groupby(["ttdate", "fareC"])["fare"].count().reset_index()
	df1 					= pd.pivot_table(df1, values='fare', index=['ttdate'],
	                    						columns=['fareC'])
	df1["total"] 			= df1[0] + df1[1] + df1[2]
	df1[0] 					= df1[0]/df1["total"]
	df1[1] 					= df1[1]/df1["total"]
	df1[2] 					= df1[2]/df1["total"]
	#
	df2 					= df.loc[df["comp"] == comp].copy().groupby(["ttdate", "fareC"])["difP"].mean().reset_index()
	#
	df2 					= pd.pivot_table(df2, values='difP', index=['ttdate'],
	                    					columns=['fareC'])
	# NOW PLOT THE RESULTS
	if comp == 1:
		name = "lffarechange_comp"
	elif comp == 0:
		name = "lffarechange"
	#
	csfont 						= {'fontname':"Liberation Serif", 'fontsize':20}
	palette 					= ["#FF6700", "#FCB07E", "#6B717E", "#3A6EA5", "#004E98", "#070707"]
	sns.set(style="white",color_codes=False)
	fig 						= plt.figure(figsize=(1.5*6.4, 1.1*4.8))
	plt.plot(range(0,60), 100*df1[2].values, label='Fare Declines',color=palette[0],linewidth = 3, linestyle='--')
	plt.plot(range(0,60), 100*df1[1], label='Fare Increases',color=palette[4],linewidth = 3,linestyle='-')
	L 							= plt.legend()
	plt.setp(L.texts, family='Liberation Serif', fontsize = 18)
	plt.xlabel('Booking Horizon',**csfont)
	plt.ylabel('Frequency (%)',**csfont)
	plt.axvline(x=54,color = palette[2],linewidth = 2,linestyle=':')
	plt.axvline(x=47,color = palette[2],linewidth = 2,linestyle=':')
	plt.axvline(x=40,color = palette[2],linewidth = 2,linestyle=':')
	plt.yticks(fontname = "Liberation Serif", fontsize = 20) 
	plt.xticks(fontname = "Liberation Serif", fontsize = 20) 
	plt.savefig(pathOutput + name + "0.pdf",bbox_inches='tight',format= "pdf",dpi=600)
	plt.close()
	#
	csfont 						= {'fontname':"Liberation Serif", 'fontsize':20}
	palette 					= ["#FF6700", "#FCB07E", "#6B717E", "#3A6EA5", "#004E98", "#070707"]
	sns.set(style="white",color_codes=False)
	fig 						= plt.figure(figsize=(1.5*6.4, 1.1*4.8))
	plt.plot(range(0,60), df2[2], label='Fare Declines',color=palette[0],linewidth = 3, linestyle='--')
	plt.plot(range(0,60), df2[1], label='Fare Increases',color=palette[4],linewidth = 3,linestyle='-')
	L 							= plt.legend()
	plt.setp(L.texts, family='Liberation Serif', fontsize = 18)
	plt.xlabel('Booking Horizon',**csfont)
	plt.ylabel('Magnitude ($)',**csfont)
	plt.axvline(x=54,color = palette[2],linewidth = 2,linestyle=':')
	plt.axvline(x=47,color = palette[2],linewidth = 2,linestyle=':')
	plt.axvline(x=40,color = palette[2],linewidth = 2,linestyle=':')
	plt.yticks(fontname = "Liberation Serif", fontsize = 20) 
	plt.xticks(fontname = "Liberation Serif", fontsize = 20) 
	plt.savefig(pathOutput + name + "1.pdf",bbox_inches='tight',format= "pdf",dpi=600)
	plt.close()



#run the program
plotFareChange(df,1)
plotFareChange(df,0)





