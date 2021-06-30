"""
    This script estimates the stochastic limit couterfactual in
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

from multiprocessing import Pool
from functools import partial
import numpy as np
import math
from scipy.stats import poisson, bernoulli
from scipy.special import binom
from scipy import stats
from scipy.optimize import minimize
import scipy
import sys
import csv
import time
import argparse
import pandas as pd
import itertools
import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap, devices, jacfwd, partial, device_put # Set jax to GPU for computing the gradient using AD
import math
from scipy import stats
from scipy.optimize import minimize
import sys
import pandas as pd
from jax.scipy.special import gammaln, logsumexp
from multiprocessing import cpu_count
from sklearn.cluster import KMeans
import os
import subprocess as sp
import time
import random
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
from plot_setup  import *

start = time.time()

# --------------------------------------------------------------------------------
# Set program parameters
# --------------------------------------------------------------------------------
numThreads = 21
numSim     = 5000
# adjust Jax to 64 bit floats
from jax.config import config
config.update("jax_enable_x64", True)

np.random.seed(31187)

market = sys.argv[1] #"SEA_SUN"

# --------------------------------------------------------------------------------
# Set path for data and logs
# --------------------------------------------------------------------------------
INPUT = "../../data/"
OUTPUT = "../../output/"


def get_gpu_memory():
    _output_to_list = lambda x: x.decode("ascii").split("\n")[:-1]
    ACCEPTABLE_AVAILABLE_MEMORY = 1024
    COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values

gpuUsage = get_gpu_memory()

gpu = 0
if gpuUsage[0] < 30000:
    gpu = 1
    if gpuUsage[1] < 30000:
        gpu = 2


print("using gpu number: " + str(gpu))
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)


# -------------------------------------------------------------------------------
# DEFINE FUNCTIONS FOR DEMAND
# -------------------------------------------------------------------------------

# define the probability that a low type wants to buy
# rewrite exp(a)/(1 + exp(a)) as 1/(1/exp(a) + 1) = 1/(1 + exp(-a))
@jit
def purchInL(beta, bL, gamma):   # p is dim(P), gam is dim(T), , beta is dim FE
    return (1 - gamma)[:, None, None] * 1 / \
        (1 + jnp.exp(-beta[None, None, :] - bL *p rices[None, :, None]))
    # returned object is T,P dimension

# define the probability that a high type wants to buy
@jit
def purchInB(beta, bB, gamma):   # p is dim(P), gam is dim(T), beta is dim FE
    return (gamma)[:, None, None] * 1 / \
        (1 + jnp.exp(-beta[None, None, :] - bB * prices[None, :, None]))
    # returned object is T,P dimension

# define, this returns a T X P matrix of probability of purchase across both consumer types
@jit
def purchIn(beta, bB, bL, gamma):
    return purchInL(beta, bL, gamma) + purchInB(beta, bB, gamma)

# define the log probability of that demand is equal to q given the parameters of the model
@jit
def log_demandQ(beta, bL, bB, gamma, mu, q):
    return q * jnp.log(mu[:, None, :]) + q * \
        jnp.log(purchIn(beta, bB, bL, gamma)) - \
        (mu[:, None, :] * purchIn(beta, bB, bL, gamma)) - gammaln(q + 1)

# define the probability of that demand is equal to q given the parameters of the model, this is just exp(log(p)),
# which is done for numerical stablility
@jit
def demandQ(beta, bL, bB, gamma, mu, q):
    #return (mu[:,None]*purchIn(aB,bB,aL,bL,prices,gamma))** \
    # q*jnp.exp(-mu[:,None]*(purchIn(aB,bB,aL,bL,prices,gamma)))/factQ[q]
    return jnp.exp(log_demandQ(beta, bL, bB, gamma, mu, q))



def allDemand(beta, bL, bB, gamma, mu):
    vlookup = vmap((
        lambda x: demandQ(beta, bL, bB, gamma, mu, x)
    ))(jnp.arange(0, qBar))
    f = jnp.zeros((qBar, qBar, len(gamma), len(prices), len(beta)))
    for q in range(1, qBar):         # seats remaining
        f = f.at[q, 0:q, :, :, :].set(vlookup[0:q, :, :, :])
        f = f.at[q, q, :, :, :] \
            .set(jnp.maximum(1 - jnp.sum(f[q, 0:(q), :, :, :], axis = 0), 1e-100))
    f = jnp.where((f > 0) & (f < 1e-100), 1e-100, f)
    return f
    # returning dimension is seats remaining, seats sold, time, and prices


allDemand_jit = jit(allDemand)




# define the CCP and EV for the dynamic firm problem
def optDynNoError(f, ER, gamma, beta):
    np.seterr(all ="raise")
    # create storage for EV, V, CCP
    EV = jnp.zeros((T, qBar, numP, len(beta)))
    V = jnp.zeros((qBar, T, len(beta)))
    CCP = jnp.zeros((T, qBar, numP, len(beta)))
    for t in range(1, T + 1):
        # work backwards in time. In the last period, we just get last period revenues
        if t == 1:
            # the softmax functin can be rewritten,
            # so let"s use logsum(exp) = x* + log sum (exp (x-x*))
            grp = ER[:, -t, :, :] * Pt[-t, 1:][None, :, None]
            V = V.at[:, -t, :].set(jnp.max(grp , axis = 1))
            # tmp = jnp.argmax(grp , axis = 1)
            # FINISH THIS
            #for q in range(qBar):
            #    for b in range(len(beta)):
            #        CCP = CCP.at[-t, q, tmp[q, b], b].set(1)
            tmp = jnp.zeros_like(grp)
            tmp = tmp.at[
                jnp.where(grp == jnp.max(grp, axis = 1)[:, None, :])
            ].set(1)
            CCP = CCP.at[-t, :, :, :].set(tmp)
        else:
            grp = (ER[:, -t, :, :] + EV[-t + 1, :, :, :]) * \
                Pt[-t, 1:][None, :, None]
            V = V.at[:, -t, :].set(jnp.max(grp, axis = 1))
        # this allows us to calc that the Pr(Q = 1) aligns with Pr(c" = c - 1)
        r,c = jnp.tril_indices_from(f[:, :, 0, 0, 0])
        for b in range(len(beta)):
            if t != T: #update expected value function, this is for not the last period
                g = jnp.array(f[:, :, -t - 1, :, b])
                g = g.at[r, c, :].set(g[r, r - c, :])
                EV = EV.at[-t, :, :, b]\
                    .set(jnp.sum(g * V[:, -t, b][None, :, None], axis = 1) * \
                    Pt[-t, 1:])
        if t != 1:
            XX = (ER[:, -t, :, :] + EV[-t + 1, :, :, :]) * \
                Pt[-t, 1:][None, :, None]
            # tmp             = jnp.argmax(XX , axis = 1)
            # # FINISH THIS
            # for q in range(qBar):
            #     for b in range(len(beta)):
            #         CCP = CCP.at[-t,q,tmp[q,b],b].set(1)
            tmp = jnp.zeros_like(XX)
            tmp = tmp.at[jnp.where(XX == jnp.max(XX, axis = 1)[:, None,:])].set(1)
            CCP = CCP.at[-t, :, :, :].set(tmp)
    return CCP

def optJump(f, ER, gamma, beta, jump):
    np.seterr(all = "raise")
    EV = jnp.zeros((T, qBar, numP, len(beta)))
    V = jnp.zeros((qBar, T, len(beta)))
    CCP = np.zeros((T,qBar,numP,len(beta)))
    l = np.arange(1, 61)
    chunks = [list(l[i: i+jump]) for i in np.arange(0, len(l), jump)]
    for ch in chunks:
        for t in ch[:-1]:
            # work backwards in time. In the last period, we just get last period revenues
            if t == 1:
                grp = ER[:, -t, :, :] * Pt[-t, 1:][None, :, None]
                V = V.at[:, -t, :].set(jnp.max(grp , axis = 1))
            else:
                grp = (ER[:, -t, :, :] + EV[-t + 1, :, :, :]) * \
                    Pt[-t, 1:][None, :, None]
                V = V.at[:, -t, :].set(jnp.max(grp , axis = 1))
            r, c = jnp.tril_indices_from(f[:, :, 0, 0, 0])
            for b in range(len(beta)):
                if t != T: #update expected value function, this is for not the last period
                    g = jnp.array(f[:, :, -t - 1, :, b])
                    g = g.at[r, c, :].set(g[r, r - c, :])
                    EV = EV.at[-t, :, :, b].set(jnp.sum(
                        g * V[:, -t, b][None, :, None],
                        axis = 1
                    ) * Pt[-t, 1:])
        t = ch[-1]
        grp = (ER[:, -t, :, :] + EV[-t + 1, :, :, :]) * Pt[-t, 1:][None, :, None]
        interV = jnp.max(grp, axis = 1)
        pstar = jnp.zeros_like(grp)
        pstar = pstar.at[
            jnp.where(grp == jnp.max(grp, axis = 1)[:, None, :])
        ].set(1)
        if t != T:
            r, c = jnp.tril_indices_from(f[:, :, 0, 0, 0])
            for b in range(len(beta)):
                if t != T: #update expected value function, this is for not the last period
                    g = jnp.array(f[:, :, -t - 1, :, b])
                    g = g.at[r, c, :].set(g[r, r - c, :])
                    EV = EV.at[-t, :, :, b].set(jnp.sum(
                        g * interV[:, b][None, :, None],
                        axis = 1
                    ) * Pt[-t, 1:])
        for q in range(qBar):
            for b in range(len(beta)):
                if (-ch[-1] + jump) == 0:
                    CCP[-ch[-1]:, q, int(pstar[q, :, b].argmax()), b] = 1
                else:
                    CCP[
                        -ch[-1]:(-ch[-1] + jump),
                        q,
                        int(pstar[q, :, b].argmax()), b
                    ] = 1
    return CCP

# Calculate the opt uniform price
def optUniform(f, ER, gamma, beta):
    EV = jnp.zeros((qBar, len(gamma), len(prices), len(beta)))
    V = jnp.zeros((qBar, len(gamma), len(prices), len(beta)))
    #jnp.zeros((qBar,len(gamma), len(prices), len(beta)))
    for t in range(1, T + 1):
        # work backwards in time. In the last period, we just get last period revenues
        if t == 1:
            V = V.at[:, -t, :, :].set( ER[:, -t, :, :])
        else:
            V = V.at[:, -t, :, :].set( ER[:, -t, :, :] + EV[:,-t + 1, :, :])
        r, c = jnp.tril_indices_from(f[:, :, 0, 0, 0])
        for b in range(len(beta)):
            if t != T: #update expected value function, this is for not the last period
                g = jnp.array(f[:, :, -t - 1, :, b])
                g = g.at[r, c, :].set(g[r, r - c, :])
                EV = EV.at[:, -t, :, b].set(jnp.sum(
                    g * V[:, -t, :, b][None, :, :],
                    axis = 1
                ))
    pstar = jnp.argmax(V[:, 0, :, :], axis = 1)
    return pstar

def allPT(Pt):
    Pt[:, 1:] = 1
    return Pt

def poissonProcess(R, aB, bL, bB, prices, p, remainCap):
    low = np.sum(R == 0) # number low types
    high = np.sum(R == 1)
    if remainCap == 0:
        sales = 0
        sales_B = 0
        sales_L = 0
        CS_L = 0 # np.nansum(T1EVout[R==0]/low)
        CS_B = 0 # np.nansum(T1EVout[R==1]/high)
        CS_ALL = 0 # T1EVout.sum()
    elif remainCap > 0:
        sj = np.exp(aB + ((1 - R) * bL + R * bB) * prices[p]) / \
            (1 + np.exp(aB + ((1 - R) * bL + R * bB)  prices[p]))
        QD = (np.random.uniform(size = len(R)) < sj) * 1
        #QD =   aB + ((1-R)*bL + R*bB)*prices[p] + T1EVin >= T1EVout
        cs = (-1 / ((1 - R) * bL + R * bB)) * \
            np.log(1 + np.exp(aB + ((1 - R) * bL + R * bB) * prices[p]))
        if QD.sum() <= remainCap:
            CS_L = np.nansum(cs[R == 0])
            CS_B = np.nansum(cs[R == 1])
            CS_ALL = cs.sum()
            sales = QD.sum()
            sales_B = QD[R == 1].sum()
            sales_L = QD[R == 0].sum()
        elif QD.sum() > remainCap:
            dif = int(QD.sum() - remainCap)
            csHitCap = cs.copy()
            csHitCap[(QD == True).nonzero()[0][:dif]] = 0
            QDHitCap = QD.copy()
            QDHitCap[(QD == True).nonzero()[0][:dif]] = 0
            CS_L = np.nansum(csHitCap[R == 0])
            CS_B = np.nansum(csHitCap[R == 1])
            CS_ALL = csHitCap.sum()
            sales = remainCap
            sales_B = QDHitCap[R == 1].sum()
            sales_L = QDHitCap[R == 0].sum()
    return sales, CS_L, CS_B, CS_ALL, sales_B, sales_L

def storeSim(Store_fl, CCP, gamma, mu, prices,VAR):
    b = int(Store_fl[0, -1])
    beta = np.array(VAR[0:7])
    bL = np.minimum(VAR[7], VAR[8])
    bB = np.maximum(VAR[7], VAR[8])
    aL = beta[b]
    for t in range(60):
        Store_flt0D = Store_fl[t, 0] # dynamic pricing entry (0)
        pstarD = np.argmax(CCP[t, Store_fl[t, 0].astype("int"), :, b])
        pois = np.random.poisson(mu[t,b]) # Arrival draw
        R = bernoulli.rvs(gamma[t], size = pois) # draw types
        if pois > 0:
            Store_fl[t, 2], Store_fl[t, 3], Store_fl[t, 4], Store_fl[t, 5], Store_fl[t, 6], Store_fl[t, 7] = poissonProcess(R, aL, bL, bB, prices, pstarD, Store_flt0D)
        else:
            Store_fl[t, 2:] = 0
        # now estimate consumer surplus
        Store_fl[t, 1] = prices[pstarD]
        Store_fl[t + 1, 0] = Store_flt0D-Store_fl[t, 2]
        Store_fl[t, 8] = np.sum(R == 0)
        Store_fl[t, 9] = np.sum(R == 1)
    Store_fl[:, 10] = b
    return Store_fl

def transformResults(S,market):
    numSim = len(S)
    sx2 = S.reshape((-1, 11))
    df = pd.DataFrame(
        {
            "capD": sx2[:, 0],
            "priceD": sx2[:, 1] * 100,
            "salesD": sx2[:, 2],
            "CS_D_L": sx2[:, 3],
            "CS_D_B": sx2[:, 4],
            "CS_D_ALL": sx2[:, 5],
            "salesD_B": sx2[:, 6],
            "salesD_L": sx2[:, 7],
            "arrivals_L": sx2[:, 8],
            "arrivals_B": sx2[:, 9],
            "dow": sx2[:, 10]
        }
    )
    df["t"] = np.tile(range(61), numSim)
    df["fl"] = np.repeat(range(numSim), 61)
    df.loc[df.capD == 0, "priceD"] = np.nan
    df.loc[df.capD == 0, "salesD"] = np.nan
    df["revD"] = df.salesD * df.priceD
    df["market"] = market
    df.loc[df.capD == 0, "salesD_B"] = np.nan
    df.loc[df.capD == 0, "salesD_L"] = np.nan
    # mark type CS = np.nan if no one arrives
    df.loc[df.arrivals_L == 0, "CS_D_L"] = np.nan
    df.loc[df.arrivals_B == 0, "CS_D_B"] = np.nan
    # now create load factors
    df["LF_D"] = np.nan
    df.loc[df.t == 0, "LF_D"] = df.capD
    df.LF_D.fillna(method="ffill", inplace = True)
    df["LF_D"] = 100 * (1 - df.capD / df.LF_D)
    return df

def create_stochLimit_plot(df, a):
    df1 = df.loc[df.initC != 0]
    df1 = df1.groupby(["initC", "fl"]).salesD.sum().reset_index()
    df2 = df1.groupby("initC").salesD.mean()

    df11 = df.loc[df.initC != 0]
    df11 = df11.groupby(["initC", "fl"]).revD.sum().reset_index()
    df22 = df11.groupby("initC").revD.mean()

    pctchange = df2.pct_change()
    # polynomial_features= PolynomialFeatures(degree=2)
    # xp = polynomial_features.fit_transform(np.arange(len(pctchange) - 1).reshape(-1, 1))
    # y = pctchange.values[1:]
    # y[y < 0] = 0
    # model = sm.OLS(y, xp).fit()
    # ypred = model.predict(xp)
    dif01 = np.argmax((np.abs(pctchange) < .001) & (pctchange > 0))

    df3 = df.loc[df.initC != 0].loc[df.initC == int(a) - 6] \
        .groupby("t").priceD.mean()
    df4 = df.loc[df.initC != 0].loc[df.initC == int(a)] \
        .groupby("t").priceD.mean()
    df5 = df.loc[df.initC != 0].loc[df.initC == dif01] \
        .groupby("t").priceD.mean()
    #

    fig = plt.figure(figsize = (2* FIG_SIZE[0], FIG_SIZE[1]))
    ax1 = fig.add_subplot(121)
    ax1.plot(
        pctchange,
        color = PALETTE[0],
        linewidth = LINE_WIDTH,
        linestyle = ":",
        label = "Percent Change in Sales with +1 Capacity"
    )
    ax1.axhline(
        0,
        linewidth = LINE_WIDTH - 1,
        color = PALETTE[-1],
        linestyle = "-"
    )
    ax1.axvline(
        a,
        linewidth = LINE_WIDTH,
        color = PALETTE[2],
        linestyle = "--",
        label="Avg. Observed Capacity"
    )
    # ax1.plot(int(a)-10,pctchange[int(a)-10],"s", ms=8,color = sns.xkcd_PALETTE(colors)[1])
    # ax1.plot(int(a),pctchange[int(a)],"o", ms=8,color = sns.xkcd_PALETTE(colors)[2])
    # ax1.plot(int(dif01),pctchange[int(dif01)],"^", ms=8,color = sns.xkcd_PALETTE(colors)[0])
    ax1.set_ylabel("Percent Change in Sales", color = PALETTE[0], **CSFONT)
    ax1.set_xlabel("Initial Capacity", **CSFONT)

    ax3 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax3.set_ylabel("Avg. Revenue", color=PALETTE[4], **CSFONT)  # we already handled the x-label with ax1
    #ax3.text(1,1,"blah")
    ax3.plot(
        df22,
        color = PALETTE[4],
        linewidth = LINE_WIDTH,
        linestyle = "-.",
        label = "Avg. Flight Rev. Given Capacity"
    )
    ax3.plot(int(a) - 6, df22[int(a) - 6], "s", ms = 10, color = PALETTE[1])
    ax3.plot(int(a), df22[int(a)], "o", ms = 10, color = PALETTE[2])
    ax3.plot(int(dif01), df22[int(dif01)], "^", ms = 10, color = PALETTE[0])
    ax3.tick_params(axis = "y", labelcolor = PALETTE[4])
    laba = Line2D(
        [0],
        [0],
        color = PALETTE[0],
        linewidth = LINE_WIDTH,
        linestyle = ":"
    )
    labb = Line2D(
        [0],
        [0],
        color = PALETTE[2],
        linewidth = LINE_WIDTH,
        linestyle = "-"
    )
    labc = Line2D(
        [0],
        [0],
        color = PALETTE[4],
        linewidth = LINE_WIDTH,
        linestyle = "-."
    )
    labels = [
        "Percent Change in Sales with +1 Capacity",
        "Avg. Observed Capacity",
        "Avg. Flight Revenue Given Capacity"
    ]
    lines = [laba, labb, labc]
    L = ax1.legend(
        lines,
        labels,
        loc = "center right",
        frameon = True,
        framealpha = 0.8
    )
    plt.setp(L.texts, family = FONT, fontsize = FONT_SIZE - 2)
    plt.yticks(fontname = FONT, fontsize = FONT_SIZE)
    plt.xticks(fontname = FONT, fontsize = FONT_SIZE)
    #
    ax2 = fig.add_subplot(122)
    ax2.plot(
        df3[:60],
        "-s",
        color = PALETTE[1],
        ms = 10,
        markevery = (len(df3) + 1),
        linewidth = LINE_WIDTH,
        label = "Capacity=" + str(int(a) - 6)
    )
    ax2.plot(
        df4[:60],
        "--o",
        color = PALETTE[2],
        ms = 10,
        markevery = (len(df3) + 1),
        linewidth = LINE_WIDTH,
        label = "Capacity=" + str(int(a)))
    ax2.plot(
        df5[:60],
        "-.^",
        color = PALETTE[0],
        ms = 10,
        markevery = (len(df3) + 1),
        linewidth = LINE_WIDTH,
        label = "Capacity=" + str(int(dif01)))
    L = ax2.legend(loc = 2,frameon = True)
    plt.setp(L.texts, family = FONT, fontsize = FONT_SIZE - 2)
    plt.yticks(fontname = FONT, fontsize = FONT_SIZE)
    plt.xticks(fontname = FONT, fontsize = FONT_SIZE)
    ax2.set_xlabel("Booking Horizon", **CSFONT)
    ax2.set_ylabel("Prices", **CSFONT)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    #
    plt.savefig(
        OUTPUT + "stochastic_limit_" + market + ".pdf",
        bbox_inches = "tight",
        format = FIG_FORMAT,
        dpi = DPI
    )
    plt.close()
    return int(dif01)



if __name__ == "__main__":
    truncate = False
    pathIn = INPUT + "estimation/" + market + "/"
    pathOut = INPUT + "estimation/" + market + "/"
    df_route = pd.read_csv(pathIn + market + ".csv", index_col = 0)
    df_route_pt = pd.read_csv(pathIn + market + "_Pt.csv", header = None)
    prices = jnp.array(np.genfromtxt(pathIn + market + "_prices.csv"))
    Pt = df_route_pt.values
    data = np.array(df_route.values)
    qBar = int(np.max(df_route.seats)) + 20
    T = len(np.unique(df_route.tdate))
    numP = len(prices)
    obs = len(df_route.tdate)
    if truncate == False:
        Pt[:, 1:] = 1
    Pt = jnp.array(Pt)
    xInit = np.genfromtxt(pathIn + "/robust_estim/" + market + "_robust_params.csv")
    VAR = jnp.array(xInit)
    beta = jnp.array(VAR[0:7])
    bL = jnp.minimum(VAR[7], VAR[8])
    bB = jnp.maximum(VAR[7], VAR[8])
    gamma = 1 / (jnp.exp(
        -VAR[9] - jnp.arange(0, 60) * VAR[10] - \
            (jnp.arange(0, 60) ** 2) * VAR[11]
    ) + 1)
    # equivalent to jnp.array([1/(1 + jnp.exp(-g[0] + -t*g[1] - t**2*g[2])) for t in range(0,60)])
    # range(int(min(Tdata)),int(max(Tdata)+1))])
    muT = jnp.array(
        [VAR[12]] * (T - 20)+[VAR[13]] * 7 + [VAR[14]] * 7 + [VAR[15]] * 6
    )
    muD = jnp.append(jnp.array([1]), jnp.array(VAR[16:22]))
    mu = muT[:, None] * muD[None, :]
    sig = VAR[-1]
    ######################
    # define all sales possibilities
    f = allDemand_jit(beta, bL, bB, gamma, mu)
    # define expected revenues
    ER = jnp.sum(
        f * jnp.array(range(qBar))[None, :, None, None, None] * \
            prices[None, None, None, :, None],
        axis = 1
    )
    CCP = np.array(optDynNoError(f, ER, gamma, beta))
    rate = mu[:, None, :] * purchIn(beta, bB, bL, gamma)
    VAR = np.array(VAR)
    gamma = np.array(gamma)
    mu = np.array(mu)
    prices = np.array(prices)
    rate = np.array(rate)
    # ADJUST NUMBER OF SIMULATIONS
    # f, t, (SeatsRemain, P, Sales, Larrive, Harrive, Lbuy, Hbuy)
    StoreX = np.zeros((len(np.arange(1, qBar)) * numSim, 61, 11))
    Q0 = np.repeat(np.arange(1, qBar), numSim)
    StoreX [:, 0, 0] = Q0.astype("int")
    StoreX [:, 0, -1] = df_route.loc[df_route.tdate == 0] \
        .sample(len(np.arange(1, qBar)) * numSim, replace = True)[["dd_dow"]]\
        .values.reshape(-1).astype("int")
    # pool2 = Pool(processes=numThreads)
    with Pool(processes = numThreads) as pool2:
        results = np.array(pool2.map(partial(
            storeSim,
            CCP = CCP,
            gamma = gamma,
            mu = mu,
            prices = prices,
            VAR = VAR
        ), StoreX))
        pool2.close()
        pool2.join()
    df = transformResults(results, market)
    df["initC"] = df.groupby("fl").capD.transform("max")
    a = df_route.loc[df_route.tdate == 0].seats.mean()
    df.to_parquet(pathOut + market + "stochLim_counterfactuals.parquet")
    end = time.time()
    print("finished counterfactual in " + str(end - start))
    if market == "PDX_SBA":
        newLim = create_stochLimit_plot(df, a)
