"""
    This script investigates the price response of nonstop bookings
    on connecting fares in
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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------------------------------
# DEFINE PATHS
# -------------------------------------------------------------------------------


pathIn                  = "/home/kw468/Projects/airlines_jmp/"
pathOutput              = "/home/kw468/Projects/airlines_jmp/output/"

# -------------------------------------------------------------------------------
# CREATE THE PLOT
# -------------------------------------------------------------------------------

df1                    	= pd.read_parquet(pathIn + "asdata_cleanCon.parquet")
df1 					= df1.groupby(["origin", "dest", "sdate", "ddate"]).fare.mean()
df1 					= df1.reset_index(drop=False)

df 						= pd.read_parquet(pathIn + "asdata_clean.parquet")
df 						= df.drop(columns = "fare")


df 						= df.merge(df1, on = ["origin", "dest", "sdate", "ddate"], how = "inner")

# replace time until departure variable from -60,0 to 0,60
df['ttdate'] 			= -df['tdate'] + 60

cols 					= ["origin", "dest", "ddate", "flightNum", "tdate"]
df                      = df.sort_values(cols, ascending = False).reset_index(drop=True)

df["seats"] 			= df.sY + df.sF
df.loc[df.capY == 76, "seats"] = df.sY

cols 					= ["origin", "dest", "flightNum", "ddate"]
df["difS"] 				= df.groupby(cols).seats.shift(-1) - df.seats
df["difP"] 				= df.groupby(cols).fare.shift(-1) - df.fare


df 						= df.loc[df.difS.notnull()]
df["seatC"] 			= 0
df 						.loc[df.difS < 0, "seatC"] = 1

df2 					= df.copy()
df2 					= df2.groupby(["ttdate", "seatC"])["difP"].mean().reset_index()
df2 					= pd.pivot_table(df2, values=['difP'], index=['ttdate'],
                    						columns=['seatC'])


def plotFareResponse(df2):
	#
	csfont 						= {'fontname':"Liberation Serif", 'fontsize':20}
	palette 					= ["#FF6700", "#FCB07E", "#6B717E", "#3A6EA5", "#004E98", "#070707"]
	sns.set(style="white",color_codes=False)
	fig 						= plt.figure(figsize=(1.5*6.4, 1.1*4.8))
	plt.plot(range(0,60), df2["difP"][0].values, label='No Sales',color=palette[0],linewidth = 3, linestyle='--')
	plt.plot(range(0,60), df2["difP"][1].values, label='Positive Sales',color=palette[4],linewidth = 3,linestyle='-')
	L 							= plt.legend()
	plt.setp(L.texts, family='Liberation Serif', fontsize = 18)
	plt.xlabel('Booking Horizon',**csfont)
	plt.ylabel('Fare Response ($)',**csfont)
	# adjust vlines by 1 because of dif() in data creation
	plt.axvline(x=54,color = palette[2],linewidth = 2,linestyle=':')
	plt.axvline(x=47,color = palette[2],linewidth = 2,linestyle=':')
	plt.axvline(x=40,color = palette[2],linewidth = 2,linestyle=':')
	plt.yticks(fontname = "Liberation Serif", fontsize = 20) 
	plt.xticks(fontname = "Liberation Serif", fontsize = 20) 
	plt.axhline(y=0, color=palette[-1], linewidth=3)
	plt.savefig(pathOutput + "fareresponse_1stop.pdf",bbox_inches='tight',format= "pdf",dpi=600)
	plt.clf()



plotFareResponse(df2)