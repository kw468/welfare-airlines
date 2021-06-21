"""
    This script plots average fares across fare classes in 
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

# _________________________________________
# DEFINE READ DATA FUNCTIONS
# _________________________________________

# first class is only observed in the AS data, so load asdata.
df_n                    = pd.read_parquet(pathIn + "asdata_clean.parquet")


# replace time until departure variable from -60,0 to 0,60
df_n['ttdate'] 		= -df_n['tdate'] + 60
df_n 				= df_n.loc[df_n.nonstop == 1]

# compute mean by day before departure
mainFare 			= df_n.groupby(['ttdate'])['mainFare'].mean()
saverFare 			= df_n.groupby(['ttdate'])['saverFare'].mean()
refundYFare 		= df_n.groupby(['ttdate'])['refundYFare'].mean()
firstFare 			= df_n.groupby(['ttdate'])['firstFare'].mean()
refundFFare 		= df_n.groupby(['ttdate'])['refundFFare'].mean()


df_n["FF"]  		= df_n[["firstFare", "refundFFare"]].min(axis=1)
FF 					= df_n.groupby(['ttdate'])['FF'].mean()

# -------------------------------------------------------------------------------
# CREATE THE PLOT
# -------------------------------------------------------------------------------



csfont 						= {'fontname':"Liberation Serif", 'fontsize':20}
palette 					= ["#FF6700", "#FCB07E", "#6B717E", "#3A6EA5", "#004E98", "#070707"]
sns.set(style="white",color_codes=False)
fig 						= plt.figure(figsize=(1.5*6.4, 1.1*4.8))
plt.plot(saverFare,   label='Saver Economy',color=palette[0],linewidth = 3, linestyle=':')
plt.plot(mainFare,    label='Economy',color=palette[1],linewidth = 3, linestyle='-')
plt.plot(refundYFare, label='Unrestricted Economy',color=palette[2],linewidth = 3, linestyle='-.')
plt.plot(firstFare,   label='First Class',color=palette[3],linewidth = 3, linestyle='-')
plt.plot(refundFFare, label='Unrestricted First Class',color=palette[4],linewidth = 3, linestyle='-.')
L 							= plt.legend()
plt.setp(L.texts, family='Liberation Serif', fontsize = 18)
plt.xlabel('Booking Horizon',**csfont)
plt.ylabel('Average Fare',**csfont)
plt.axvline(x=53,color = palette[2],linewidth = 2,linestyle=':')
plt.axvline(x=46,color = palette[2],linewidth = 2,linestyle=':')
plt.axvline(x=39,color = palette[2],linewidth = 2,linestyle=':')
plt.yticks(fontname = "Liberation Serif", fontsize = 20) 
plt.xticks(fontname = "Liberation Serif", fontsize = 20) 

#
plt.savefig(pathOutput + "fyfares.pdf",bbox_inches='tight',format= "pdf",dpi=600)
plt.close()