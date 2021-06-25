"""
    This script plots the estimated arrival processes in 
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
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob
import re
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import os

pathData                = "/home/kw468/Projects/airlines_jmp/"
pathIn                  = "/home/kw468/Projects/airlines_jmp/estimation/"
pathOutput              = "/home/kw468/Projects/airlines_jmp/output/"

os.chdir(pathData)
from estim_markets import *


# -------------------------------------------------------------------------------
# DEFINE FUNCTIONS FOR DEMAND
# -------------------------------------------------------------------------------

routeDirs 	= glob.glob(pathIn + "*_*")
routes 		= [re.split("/",f)[-1] for f in routeDirs]
routes 		= sorted(routes)


routes 		= [r for r in routes if r in mkts]

paramFiles 	= [f + "_robust_params.csv" for f in routes]
dataFiles 	= [f + ".csv" for f in routes]
priceFiles 	= [f + "_prices.csv" for f in routes]
T = 60


# gather the parameters for a given market
def processRoute(num):
	VAR = np.genfromtxt(pathIn + "/" + routes[num] + "/robust_estim/" + paramFiles[num])
	beta                =       np.array(VAR[0:7])
	bL                  =       np.minimum(VAR[7], VAR[8])
	bB                  =       np.maximum(VAR[7], VAR[8])
	gamma               =       1/(np.exp(-VAR[9] - np.arange(0,60)*VAR[10] - (np.arange(0,60)**2)*VAR[11]) + 1)
	# equivalent to jnp.array([1/(1 + jnp.exp(-g[0] + -t*g[1] - t**2*g[2])) for t in range(0,60)])#range(int(min(Tdata)),int(max(Tdata)+1))])
	muT                 =       np.array([VAR[12]]*(T-20)+[VAR[13]]*7 + [VAR[14]]*7 + [VAR[15]]*6 )
	muD                 =       np.append(np.array([1]),np.array(VAR[16:22]))
	mu                  =       muT[:,None]*muD[None,:]
	sig                 =       VAR[-1]
	return gamma, mu, routes[num]


# stack the parameters into df
df = pd.DataFrame()
for it in range(len(routes)):
    gamma,mu,mkt        = processRoute(it)
    tempStore           = pd.DataFrame()
    tempStore["gamma"]  = gamma
    #tempStore["mu"]     = mu
    tempStore["market"] = mkt
    tempStore["tdate"]  = np.arange(len(tempStore["market"]))   
    df = df.append(tempStore)
    #

df = df.reset_index(drop=True)


# -------------------------------------------------------------------------------
# PLOTTING FUNCTIONS
# -------------------------------------------------------------------------------

def gammaPlot(df):
	csfont 						= {'fontname':"Liberation Serif", 'fontsize':20}
	palette 					= ["#FF6700", "#FCB07E", "#6B717E", "#3A6EA5", "#004E98", "#070707"]
	sns.set(style="white",color_codes=False)
	fig 						= plt.figure(figsize=(1.5*6.4, 1.1*4.8))
	plt.plot(df.groupby("tdate").gamma.mean(),color=palette[0],linewidth = 3, linestyle='-', label="Mean Pr(Business) over Markets")
	plt.plot(df.groupby("tdate").gamma.quantile(.25),color=palette[1],linewidth = 2, linestyle='-.', label="25th-75th Percentiles")
	plt.plot(df.groupby("tdate").gamma.quantile(.75),color=palette[1],linewidth = 2, linestyle='-.', label="_nolabel_")
	plt.plot(df.groupby("tdate").gamma.quantile(.05),color=palette[4],linewidth = 2, linestyle='--', label="5th-95th Percentiles")
	plt.plot(df.groupby("tdate").gamma.quantile(.95),color=palette[4],linewidth = 2, linestyle='--', label="_nolabel_")
	L 							= plt.legend()
	plt.setp(L.texts, family='Liberation Serif', fontsize = 18)
	plt.xlabel('Booking Horizon',**csfont)
	plt.ylabel('Pr(Business)',**csfont)
	plt.yticks(fontname = "Liberation Serif", fontsize = 20) 
	plt.xticks(fontname = "Liberation Serif", fontsize = 20) 
	name = "arrivalprocess_gamma.pdf"
	plt.savefig(pathOutput + name,bbox_inches='tight',format= "pdf",dpi=600)
	plt.close()



def processDOW(num):
	VAR 				= 		np.genfromtxt(pathIn + "/" + routes[num] + "/robust_estim/" + paramFiles[num])
	prices 				= 		np.genfromtxt(pathIn + "/" + routes[num] + "/" +  priceFiles[num])
	data 				= 		pd.read_csv(  pathIn + "/" + routes[num] + "/" +  dataFiles[num])
	beta                =       np.array(VAR[0:7])
	bL                  =       np.minimum(VAR[7], VAR[8])
	bB                  =       np.maximum(VAR[7], VAR[8])
	gamma               =       1/(np.exp(-VAR[9] - np.arange(0,60)*VAR[10] - (np.arange(0,60)**2)*VAR[11]) + 1)
	# equivalent to jnp.array([1/(1 + jnp.exp(-g[0] + -t*g[1] - t**2*g[2])) for t in range(0,60)])#range(int(min(Tdata)),int(max(Tdata)+1))])
	muT                 =       np.array([VAR[12]]*(T-20)+[VAR[13]]*7 + [VAR[14]]*7 + [VAR[15]]*6 )
	muD                 =       np.append(np.array([1]),np.array(VAR[16:22]))
	mu                  =       muT[:,None]*muD[None,:]
	sig                 =       VAR[-1]
	wMean = 0
	for b in range(7):
		wMean				+= 		np.average(bL*(1-gamma) + bB*gamma, weights=mu[:,b])
	return -beta/(wMean/7)


def DOWplot():
	X = []
	for it in range(len(routes)):
	    vec        = processDOW(it)
	    X.append(vec)
	#
	X 			= 100*np.array(X)
	DOW_prefs 	= (X - X.min(1)[:,None]).mean(0)
	#
	Y 			= np.zeros(X.shape)
	Y[X - X.min(1)[:,None] == 0] = 1
	Y.sum(0)
	#
	Z 						= X - X.min(1)[:,None]
	#
	csfont 						= {'fontname':"Liberation Serif", 'fontsize':20}
	palette 					= ["#FF6700", "#FCB07E", "#6B717E", "#3A6EA5", "#004E98", "#070707"]
	sns.set(style="white",color_codes=False)
	fig 						= plt.figure(figsize=(1.5*6.4, 1.1*4.8))
	boxprops 				= dict(linestyle='-', linewidth=1)#, color='b')
	flierprops 				= dict(marker='o', markerfacecolor=palette[4], markersize=6,
	                 		  		linestyle='none')
	medianprops 			= dict(linestyle='-', linewidth=2, color=palette[1])
	meanpointprops 			= dict(marker='D', markeredgecolor=palette[2],
	                      			markerfacecolor=palette[0],markersize=10)
	#
	plt.boxplot(Z, meanprops=meanpointprops, meanline=False,showmeans=False,medianprops=medianprops, boxprops=boxprops, showfliers=False) #flierprops=flierprops)
	plt.xticks([1, 2, 3, 4, 5, 6, 7], ['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun'],fontname = "Liberation Serif", fontsize = 14) 
	plt.yticks(fontname = "Liberation Serif", fontsize = 20) 
	plt.xlabel('Departure Day-of-Week',**csfont)
	plt.ylabel('WTP Relative to Least Pref. DOW',**csfont)
	legend_elements = [Patch(facecolor=palette[4], edgecolor=palette[4], alpha=0.4,
	                         label='Mean'),
						Line2D([0], [0], color=palette[1], lw=2, label='Median')]
	L 							= plt.legend(handles=legend_elements)
	plt.setp(L.texts, family='Liberation Serif', fontsize = 18)
	plt.bar(range(1, 8), height=DOW_prefs,align='center', alpha=0.4, color=palette[4])
	name = "dow_prefs.pdf"
	plt.savefig(pathOutput + name,bbox_inches='tight',format= "pdf",dpi=600)
	plt.close()


def plotAllGamma(df):
	ncols = int(4)
	nrows = int(np.ceil(len(routes)/ncols))
	csfont 						= {'fontname':"Liberation Serif", 'fontsize':20}
	palette 					= ["#FF6700", "#FCB07E", "#6B717E", "#3A6EA5", "#004E98", "#070707"]
	sns.set(style="white",color_codes=False)
	#fig                     = plt.figure(figsize=(2*1.5*6.4, 4*1.1*4.8))
	fig, axs = plt.subplots(nrows, ncols,figsize=(3*1.5*6.4, 5*1.1*4.8))
	fig.subplots_adjust(hspace=.4)
	counter = 0
	for i in range(nrows - 1):
	    for j in range(ncols):
	        mkt                 = df.loc[df.market == routes[counter]]
	        axs[i, j].plot(mkt.tdate, mkt.gamma)   
	        axs[i, j].set_ylim((-.1,1.1))
	        axs[i, j].set_title(routes[counter], fontsize=20, fontname="Liberation Serif")
	        counter += 1
	#
	remain = len(routes) - counter
	for j in range(remain):
	    mkt                 = df.loc[df.market == routes[counter]]
	    axs[nrows-1, j].plot(mkt.tdate, mkt.gamma)  
	    axs[nrows-1, j].set_ylim((-.1,1.1))
	    axs[nrows-1, j].set_title(routes[counter], fontsize=20, fontname="Liberation Serif")
	    counter += 1
	#
	for ax in axs.flat:
	    ax.set(xlabel="Booking Horizon", ylabel=r'$\gamma_t$')
	    ax.xaxis.get_label().set_fontsize(20)
	    ax.yaxis.get_label().set_fontsize(20)
	    ax.xaxis.get_label().set_fontname("Liberation Serif")
	    ax.yaxis.get_label().set_fontname("Liberation Serif")
	#
	numDelete = ncols - remain
	for j in range(numDelete):
	    fig.delaxes(axs[nrows-1][ncols-j-1])
	#
	plt.yticks(fontname = "Liberation Serif", fontsize = 20) 
	plt.xticks(fontname = "Liberation Serif", fontsize = 20) 
	name = "gamma_alls.pdf"
	plt.savefig(pathOutput + name,bbox_inches='tight',format= "pdf",dpi=600)
	plt.clf()


# -------------------------------------------------------------------------------
# RUN THE PROGRAM
# -------------------------------------------------------------------------------

gammaPlot(df)
DOWplot()
plotAllGamma(df)