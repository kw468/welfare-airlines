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
from estim_markets import *
from plot_setup import *

INPUT = "../../data/estimation/"
OUTPUT = "../../output/"

# -------------------------------------------------------------------------------
# DEFINE FUNCTIONS FOR DEMAND
# -------------------------------------------------------------------------------

routeDirs = glob.glob(INPUT + "*_*")
routes = [re.split("/",f)[-1] for f in routeDirs]
routes = sorted(routes)


routes = [r for r in routes if r in mkts]

paramFiles = [f + "_robust_params.csv" for f in routes]
dataFiles = [f + ".csv" for f in routes]
priceFiles = [f + "_prices.csv" for f in routes]
T = 60


# gather the parameters for a given market
def processRoute(num):
    VAR = np.genfromtxt(
        INPUT + "/" + routes[num] + "/robust_estim/" + paramFiles[num]
    )
    beta = np.array(VAR[0:7])
    bL = np.minimum(VAR[7], VAR[8])
    bB = np.maximum(VAR[7], VAR[8])
    gamma = 1 / (np.exp(
        -VAR[9] - np.arange(0, 60) * VAR[10] - (np.arange(0, 60) ** 2) * VAR[11]
    ) + 1)
    # equivalent to jnp.array([1/(1 + jnp.exp(-g[0] + -t*g[1] - t**2*g[2])) for t in range(0,60)])
    # range(int(min(Tdata)),int(max(Tdata)+1))])
    muT = np.array(
        [VAR[12]] * (T - 20) + [VAR[13]] * 7 + [VAR[14]] * 7 + [VAR[15]] * 6
    )
    muD = np.append(np.array([1]), np.array(VAR[16:22]))
    mu = muT[:, None] * muD[None, :]
    sig = VAR[-1]
    return gamma, mu, routes[num]

# stack the parameters into df
df = pd.DataFrame()
for it in range(len(routes)):
    gamma, mu, mkt = processRoute(it)
    tempStore = pd.DataFrame()
    tempStore["gamma"] = gamma
    #tempStore["mu"] = mu
    tempStore["market"] = mkt
    tempStore["tdate"] = np.arange(len(tempStore["market"]))
    df = df.append(tempStore)
    #

df = df.reset_index(drop = True)

# -------------------------------------------------------------------------------
# PLOTTING FUNCTIONS
# -------------------------------------------------------------------------------

def gammaPlot(df):
    fig = plt.figure(figsize = FIG_SIZE)
    plt.plot(
        df.groupby("tdate").gamma.mean(),
        color=PALETTE[0],
        linewidth = LINE_WIDTH,
        linestyle = "-",
        label = "Mean Pr(Business) over Markets"
    )
    plt.plot(
        df.groupby("tdate").gamma.quantile(.25),
        color = PALETTE[1],
        linewidth = LINE_WIDTH - 1,
        linestyle = "-.",
        label = "25th-75th Percentiles"
    )
    plt.plot(
        df.groupby("tdate").gamma.quantile(.75),
        color = PALETTE[1],
        linewidth = LINE_WIDTH - 1,
        linestyle = "-.",
        label = "_nolabel_"
    )
    plt.plot(
        df.groupby("tdate").gamma.quantile(.05),
        color = PALETTE[4],
        linewidth = LINE_WIDTH - 1,
        linestyle = "--",
        label = "5th-95th Percentiles"
    )
    plt.plot(
        df.groupby("tdate").gamma.quantile(.95),
        color = PALETTE[4],
        linewidth = LINE_WIDTH - 1,
        linestyle = "--",
        label = "_nolabel_"
    )
    L 							= plt.legend()
    plt.setp(L.texts, family = FONT, fontsize = FONT_SIZE - 2)
    plt.xlabel("Booking Horizon", **CSFONT)
    plt.ylabel("Pr(Business)",**CSFONT)
    plt.yticks(fontname = FONT, fontsize = FONT_SIZE)
    plt.xticks(fontname = FONT, fontsize = FONT_SIZE)
    plt.savefig(
        OUTPUT + "arrivalprocess_gamma.pdf",
        bbox_inches = "tight",
        format = FIG_FORMAT,
        dpi = DPI
    )
    plt.close()

def processDOW(num):
    VAR = np.genfromtxt(
        INPUT + "/" + routes[num] + "/robust_estim/" + paramFiles[num]
    )
    prices = np.genfromtxt(
        INPUT + "/" + routes[num] + "/" +  priceFiles[num]
    )
    data = pd.read_csv(
        INPUT + "/" + routes[num] + "/" +  dataFiles[num]
    )
    beta = np.array(VAR[0:7])
    bL = np.minimum(VAR[7], VAR[8])
    bB = np.maximum(VAR[7], VAR[8])
    gamma = 1 / (np.exp(
        -VAR[9] - np.arange(0, 60) * VAR[10] - (np.arange(0, 60) ** 2) * VAR[11]
    ) + 1)
    # equivalent to jnp.array([1/(1 + jnp.exp(-g[0] + -t*g[1] - t**2*g[2])) for t in range(0,60)])
    # range(int(min(Tdata)),int(max(Tdata)+1))])
    muT = np.array(
        [VAR[12]] * (T - 20) + [VAR[13]] * 7 + [VAR[14]] * 7 + [VAR[15]] * 6
    )
    muD = np.append(np.array([1]),np.array(VAR[16:22]))
    mu = muT[:, None] * muD[None, :]
    sig = VAR[-1]
    wMean = 0
    for b in range(7):
        wMean += np.average(bL * (1 - gamma) + bB * gamma, weights = mu[:, b])
    return -beta / (wMean / 7)


def DOWplot():
    X = []
    for it in range(len(routes)):
        vec = processDOW(it)
        X.append(vec)
    #
    X = 100 * np.array(X)
    DOW_prefs = (X - X.min(1)[:, None]).mean(0)
    #
    Y = np.zeros(X.shape)
    Y[X - X.min(1)[:, None] == 0] = 1
    Y.sum(0)
    #
    Z = X - X.min(1)[:, None]
    #
    fig = plt.figure(figsize = FIG_SIZE)
    boxprops = dict(linestyle = "-", linewidth = LINE_WIDTH - 2)#, color="b")
    flierprops = dict(
        marker = "o",
        arkerfacecolor = PALETTE[4],
        markersize = 6,
        linestyle = "none"
    )
    medianprops = dict(
        linestyle = "-",
        linewidth = LINE_WIDTH - 1,
        color = PALETTE[1]
    )
    meanpointprops = dict(
        marker = "D",
        markeredgecolor = PALETTE[2],
        markerfacecolor = PALETTE[0],
        markersize = 10
    )
    #
    plt.boxplot(
        Z,
        meanprops = meanpointprops,
        meanline = False,
        showmeans = False,
        medianprops = medianprops,
        boxprops = boxprops,
        showfliers = False
    ) #flierprops=flierprops)
    plt.xticks(
        [1, 2, 3, 4, 5, 6, 7],
        ["Mon", "Tues", "Wed", "Thurs", "Fri", "Sat", "Sun"],
        fontname = FONT,
        fontsize = FONT_SIZE - 6
    )
    plt.yticks(fontname = FONT, fontsize = FONT_SIZE)
    plt.xlabel("Departure Day-of-Week", **CSFONT)
    plt.ylabel("WTP Relative to Least Pref. DOW", **CSFONT)
    legend_elements = [
        Patch(
            facecolor = PALETTE[4],
            edgecolor = PALETTE[4],
            alpha = 0.4,
            label = "Mean"
        ),
        Line2D(
            [0],
            [0],
            color = PALETTE[1],
            lw = LINE_WIDTH - 1,
            label = "Median"
        )
    ]
    plt.setp(
        plt.legend(handles = legend_elements).texts,
        family = FONT,
        fontsize = FONT_SIZE - 2
    )
    plt.bar(
        range(1, 8),
        height = DOW_prefs,
        align = "center",
        alpha = 0.4,
        color = PALETTE[4]
    )
    plt.savefig(
        OUTPUT + "dow_prefs.pdf",
        bbox_inches = "tight",
        format = FIG_FORMAT,
        dpi = DPI
    )
    plt.close()

def plotAllGamma(df):
    ncols = int(4)
    nrows = int(np.ceil(len(routes) / ncols))
    fig, axs = plt.subplots(
        nrows, ncols, figsize =(3 * FIG_SIZE[0], 5 * FIG_SIZE[1])
    )
    fig.subplots_adjust(hspace = .4)
    counter = 0
    for i in range(nrows - 1):
        for j in range(ncols):
            mkt = df.loc[df.market == routes[counter]]
            axs[i, j].plot(mkt.tdate, mkt.gamma)
            axs[i, j].set_ylim((-.1, 1.1))
            axs[i, j].set_title(
                routes[counter], fontsize = FIG_SIZE, fontname = FONT
            )
            counter += 1
    remain = len(routes) - counter
    for j in range(remain):
        mkt = df.loc[df.market == routes[counter]]
        axs[nrows - 1, j].plot(mkt.tdate, mkt.gamma)
        axs[nrows - 1, j].set_ylim((-.1,1.1))
        axs[nrows - 1, j].set_title(
            routes[counter], fontsize = FIG_SIZE, fontname = FONT
        )
        counter += 1
    #
    for ax in axs.flat:
        ax.set(xlabel = "Booking Horizon", ylabel = r"$\gamma_t$")
        ax.xaxis.get_label().set_fontsize(FIG_SIZE)
        ax.yaxis.get_label().set_fontsize(FIG_SIZE)
        ax.xaxis.get_label().set_fontname(FONT)
        ax.yaxis.get_label().set_fontname(FONT)
    #
    numDelete = ncols - remain
    for j in range(numDelete):
        fig.delaxes(axs[nrows - 1][ncols - j - 1])
    #
    plt.yticks(fontname = FONT, fontsize = FONT_SIZE)
    plt.xticks(fontname = FONT, fontsize = FONT_SIZE)
    plt.savefig(
        OUTPUT + "gamma_alls.pdf",
        bbox_inches = "tight",
        format = FIG_FORMAT,
        dpi = DPI
    )
    plt.clf()

# -------------------------------------------------------------------------------
# RUN THE PROGRAM
# -------------------------------------------------------------------------------

gammaPlot(df)
DOWplot()
plotAllGamma(df)
