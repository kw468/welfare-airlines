"""
    This script plots fare availability over different fare classes
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
# OPEN THE DATA / CLEAN OBS
# -------------------------------------------------------------------------------


# only AS data has first class observations, so we only use those data
df                    = pd.read_parquet(pathIn + "asdata_clean.parquet")
df 					  = df.loc[df.nonstop == 1] # no connecting obs

# mark availability of each fare type
df.loc[df.mainFare.notnull(), "mainFare"] 		= 1
df.loc[df.saverFare.notnull(), "saverFare"] 	= 1
df.loc[df.refundYFare.notnull(), "refundYFare"] = 1

df.loc[df.mainFare.isnull(), "mainFare"] 		= 0
df.loc[df.saverFare.isnull(), "saverFare"] 		= 0
df.loc[df.refundYFare.isnull(), "refundYFare"] 	= 0

df.loc[(df.firstFare.notnull()) & (df.capF.notnull()), "firstFare"] 	= 1
df.loc[(df.refundFFare.notnull()) & (df.capF.notnull()), "refundFFare"] = 1

df.loc[(df.firstFare.isnull()) & (df.capF.notnull()), "firstFare"] 		= 0
df.loc[(df.refundFFare.isnull()) & (df.capF.notnull()), "refundFFare"] 	= 0

# replace time until departure variable from -60,0 to 0,60
df['ttdate'] = -df['tdate'] + 60


# this code obtains the fraction of flights that have first class
df1 	= df.groupby(["flightNum", "ddate", "origin", "dest"]).sF.max().reset_index() 
frac 	= df1.sF.notnull().sum() / df[["flightNum", "ddate", "origin", "dest"]].drop_duplicates().shape[0]

# -------------------------------------------------------------------------------
# CREATE THE PLOT
# -------------------------------------------------------------------------------



csfont                      = {'fontname':"Liberation Serif", 'fontsize':20}
palette                     = ["#FF6700", "#FCB07E", "#6B717E", "#3A6EA5", "#004E98", "#070707"]
sns.set(style="white",color_codes=False)
fig                         = plt.figure(figsize=(1.5*6.4, 1.1*4.8))
plt.plot(range(0,61), 100*df.groupby("ttdate")["saverFare"].mean(), label='Saver Economy',color=palette[0],linewidth = 3,linestyle=':')
plt.plot(range(0,61), 100*df.groupby("ttdate")["mainFare"].mean(), label='Economy Class',color=palette[1],linewidth = 3,linestyle='-.')
plt.plot(range(0,61), frac*100*df.groupby("ttdate")["firstFare"].mean(), label='First Class',color=palette[3],linewidth = 3, linestyle='-')
L                           = plt.legend()
plt.setp(L.texts, family='Liberation Serif', fontsize = 18)
plt.xlabel('Booking Horizon',**csfont)
plt.ylabel('Percent of Flights with Available Fares',**csfont)
plt.axvline(x=53,color = palette[2],linewidth = 2,linestyle=':')
plt.axvline(x=46,color = palette[2],linewidth = 2,linestyle=':')
plt.axvline(x=39,color = palette[2],linewidth = 2,linestyle=':')
plt.yticks(fontname = "Liberation Serif", fontsize = 20) 
plt.xticks(fontname = "Liberation Serif", fontsize = 20) 



#
plt.savefig(pathOutput + "fareavail.pdf",bbox_inches='tight',format= "pdf",dpi=600)
plt.close()






