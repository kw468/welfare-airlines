"""
    This code processes the Expert Flyer and Yapta data in Williams (2021)
    Inputs:
        * Bucket availability and seat maps are pulled from the expertflyer.com API
        * Prices come from queries to the Yapta API.
    Operations:
        * The code constructs enplanement totals from xml files
        * Prices are gathered from queries on yapta.
    Output:
        * Nonstop data
        * Onestop data
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
df["ddate"] 			= df["ddate"].astype('category').cat.codes

df                      = df.sort_values(cols, ascending = False).reset_index(drop=True)

cols 					= ["origin", "dest", "flightNum", "ddate"]
df["difS"] 				= df.groupby(cols).seats.shift(-1) - df.seats
df["difP"] 				= df.groupby(cols).fare.shift(-1) - df.fare


df 						= df.loc[df.difS.notnull()]
df["seatC"] 			= 0
df 						.loc[df.difS < 0, "seatC"] = 1

# -------------------------------------------------------------------------------
# CREATE THE PLOTS
# -------------------------------------------------------------------------------


# this function plots the fare response of nonstop bookings on nonstop fares
# both monopoly and duopoly markets are considered.
def plotFareResponse(df,comp):
	if comp == 1:
		name = "fareresponse_comp.pdf"
	elif comp == 0:
		name = "fareresponse.pdf"
		#
	df1 						= df.loc[df["comp"] == comp]
	df1 						= df1.groupby(["ttdate", "seatC"])["difP"].mean().reset_index()
	df1 						= pd.pivot_table(df1, values='difP', index=['ttdate'],
	                    						columns=['seatC'])
	#
	#
	csfont 						= {'fontname':"Liberation Serif", 'fontsize':20}
	palette 					= ["#FF6700", "#FCB07E", "#6B717E", "#3A6EA5", "#004E98", "#070707"]
	sns.set(style="white",color_codes=False)
	fig 						= plt.figure(figsize=(1.5*6.4, 1.1*4.8))
	plt.plot(range(0,60), df1[0], label='No Sales',color=palette[0],linewidth = 3, linestyle='--')
	plt.plot(range(0,60), df1[1], label='Positive Sales',color=palette[4],linewidth = 3,linestyle='-')
	L 							= plt.legend()
	plt.setp(L.texts, family='Liberation Serif', fontsize = 18)
	plt.xlabel('Booking Horizon',**csfont)
	plt.ylabel('Fare Response ($)',**csfont)
	# adjust vline for dif() in data construction
	plt.axvline(x=54,color = palette[2],linewidth = 2,linestyle=':')
	plt.axvline(x=47,color = palette[2],linewidth = 2,linestyle=':')
	plt.axvline(x=40,color = palette[2],linewidth = 2,linestyle=':')
	plt.yticks(fontname = "Liberation Serif", fontsize = 20) 
	plt.xticks(fontname = "Liberation Serif", fontsize = 20) 
	plt.axhline(y=0, color=palette[-1], linewidth=3)
	plt.savefig(pathOutput + name,bbox_inches='tight',format= "pdf",dpi=600)
	plt.clf()


# this function plots the fare response of first class bookings on first class fares.
def plotFareResponseFirst(df_n):
	# # replace time until departure variable from -60,0 to 0,60
	df_n['ttdate'] 			= -df_n['tdate'] + 60
	#
	df_n["seats"] 			=  df_n.sF
	#
	cols 					= ["origin", "dest", "ddate", "flightNum", "tdate"]
	df["ddate"] 			= df_n["ddate"].astype('category').cat.codes
	df_n                    = df_n.sort_values(cols, ascending = False).reset_index(drop=True)
	cols 					= ["origin", "dest", "flightNum", "ddate"]
	df_n["difS"] 			= df_n.groupby(cols).seats.shift(-1) - df_n.seats
	df_n["difP"] 			= df_n.groupby(cols).firstFare.shift(-1) - df_n.firstFare
	#
	df_n 					= df_n.loc[df_n.difS.notnull()]
	df_n["seatC"] 			= 0
	df_n 					.loc[df_n.difS < 0, "seatC"] = 1
	#
	df2 					= df_n.copy()
	df2 					= df2.groupby(["ttdate", "seatC"])["difP"].mean().reset_index()
	df2 					= pd.pivot_table(df2, values=['difP'], index=['ttdate'],
	                    						columns=['seatC'])
	#
	csfont 						= {'fontname':"Liberation Serif", 'fontsize':20}
	palette 					= ["#FF6700", "#FCB07E", "#6B717E", "#3A6EA5", "#004E98", "#070707"]
	sns.set(style="white",color_codes=False)
	fig 						= plt.figure(figsize=(1.5*6.4, 1.1*4.8))
	plt.plot(range(0,60), df2["difP"][0], label='No Sales',color=palette[0],linewidth = 3, linestyle='--')
	plt.plot(range(0,60), df2["difP"][1], label='Positive Sales',color=palette[4],linewidth = 3,linestyle='-')
	L 							= plt.legend()
	plt.setp(L.texts, family='Liberation Serif', fontsize = 18)
	plt.xlabel('Booking Horizon',**csfont)
	plt.ylabel('Fare Response ($)',**csfont)
	# adjust vline for dif() in data construction
	plt.axvline(x=54,color = palette[2],linewidth = 2,linestyle=':')
	plt.axvline(x=47,color = palette[2],linewidth = 2,linestyle=':')
	plt.axvline(x=40,color = palette[2],linewidth = 2,linestyle=':')
	plt.yticks(fontname = "Liberation Serif", fontsize = 20) 
	plt.xticks(fontname = "Liberation Serif", fontsize = 20) 
	plt.axhline(y=0, color=palette[-1], linewidth=3)
	plt.savefig(pathOutput + "fareresponse_firstC.pdf",bbox_inches='tight',format= "pdf",dpi=600)
	plt.close()




plotFareResponse(df,0)
plotFareResponse(df,1)
plotFareResponseFirst(df_n)


