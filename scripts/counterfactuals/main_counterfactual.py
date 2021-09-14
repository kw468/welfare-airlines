"""
    This script estimates the couterfactual in
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

start = time.time()

# --------------------------------------------------------------------------------
# Set program parameters
# --------------------------------------------------------------------------------
numThreads = 21
numSim     = 100000
# adjust Jax to 64 bit floats
from jax.config import config
config.update("jax_enable_x64", True)

np.random.seed(31187)

market = sys.argv[1] #"SEA_SUN"

# --------------------------------------------------------------------------------
# Set path for data and logs
# --------------------------------------------------------------------------------
INPUT = "../../estimation/"

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

def optStatic(f,ER,gamma,beta):
    EV = jnp.zeros((T, qBar, numP, len(beta)))
    V = jnp.zeros((qBar, T, len(beta)))
    CCP = jnp.zeros((T, qBar, numP, len(beta)))
    for t in range(1,T + 1):
        grp = ER[:, -t, :, :] * Pt[-t, 1:][None, :, None]
        V = V.at[:, -t, :].set(jnp.max(grp, axis = 1))
        tmp = jnp.zeros_like(grp)
        tmp = tmp.at[jnp.where(grp == jnp.max(grp , axis = 1)[:, None, :])].set(1)
        CCP = CCP.at[-t, :, :, :].set(tmp)
    return CCP

# THIS HAS NOT BEEN UPDATED
def optStaticInf(rate,prices,Pt):
    V = rate * prices[None, :, None] * Pt[:, 1:][:, :, None]
    return np.argmax(V, axis = 1)

# Calculate the opt uniform price
def optUniform(f,ER,gamma,beta):
    EV = jnp.zeros((qBar, len(gamma), len(prices), len(beta)))
    V = jnp.zeros((qBar, len(gamma), len(prices), len(beta)))
    #jnp.zeros((qBar,len(gamma), len(prices), len(beta)))
    for t in range(1, T + 1):
        # work backwards in time. In the last period, we just get last period revenues
        if t == 1:
            V = V.at[:, -t, :, :].set(ER[:,-t, :, :])
        else:
            V = V.at[:, -t, :, :].set(ER[:,-t, :, :] + EV[:,-t+1, :, :])
        r,c = jnp.tril_indices_from(f[:, :, 0, 0, 0])
        for b in range(len(beta)):
            if t != T: #update expected value function, this is for not the last period
                g = jnp.array(f[:, :,-t - 1, :, b])
                g = g.at[r, c, :].set(g[r, r - c, :])
                EV = EV.at[:, -t, :, b] \
                    .set(jnp.sum(g * V[:, -t, :, b][None, :, :], axis = 1))
    pstar = jnp.argmax(V[:, 0, :, :], axis = 1)
    return pstar



def calcAPDvalue(f,ER,gamma,qBar,b,pvec):
    pstar = jnp.array(
        [pvec[0]] * 39 + [pvec[1]] * 7 + [pvec[2]] * 7 + \
        [pvec[3]] * 4 + [pvec[4]] * 3
    )
    EV = jnp.zeros((qBar, len(gamma)))
    V = jnp.zeros((qBar, len(gamma)))
    for t in range(1, T + 1):
        # work backwards in time. In the last period, we just get last period revenues
        if t == 1:
            #V[:,-t]     = ER[:,-t,pstar[-t]]
            V = V.at[:, -t].set(ER[:, -t, pstar[-t], b])
        else:
            #V[:,-t]     = ER[:,-t,pstar[-t]] + EV[:,-t+1]
            V = V.at[:, -t].set(ER[:, -t, pstar[-t], b] + EV[:, -t + 1])
        r, c = jnp.tril_indices_from(f[:,:,0,0,0])
        if t != T: #update expected value function, this is for not the last period
            g = jnp.array(f[:, :, -t - 1, :, b])
            g = g.at[r, c, :].set(g[r ,r - c, :])
            EV = EV.at[:, -t] \
                .set(jnp.sum(g[:, :, pstar[-t]] * V[:, -t][None, :], axis = 1))
    return [pstar, V[:, 0]]



executeAPDjit = jit(lambda x, y: calcAPDvalue(f, ER, gamma, qBar, y, x))


# def executeAPDpool(args):
#     f,ER,gamma,qBar,arrivalType,pvec = args
#     return calcAPDvalue(f,ER,gamma,qBar,arrivalType,pvec)

def checker(Pt, pset):
    X = []
    true1 = Pt[0:39][:, 1:].max(axis = 0)
    true2 = Pt[0 + 39:39 + 7][:, 1:].max(axis = 0)
    true3 = Pt[0 + 39 + 7:39 + 7 + 7][:, 1:].max(axis = 0)
    true4 = Pt[0 + 39 + 7 + 7:39 + 7 + 7 + 4][:, 1:].max(axis = 0)
    true5 = Pt[0 + 39 + 7 + 7 + 4:39 + 7 + 7 + 4 + 3][:, 1:].max(axis = 0)
    for p in pset:
        check1 = true1[p[0]] == 1
        check2 = true2[p[1]] == 1
        check3 = true3[p[2]] == 1
        check4 = true4[p[3]] == 1
        check5 = true5[p[4]] == 1
        if np.sum([check1, check2, check3, check4, check5]) == 5:
            X.append(p)
    return X

def optAPD(f,ER,gamma,qBar,Pt):
    pset = list(itertools.product(range(Pt.shape[1] - 1), repeat = 5))
    pset = checker(Pt, pset)
    Pstar = np.zeros((qBar,len(gamma), len(beta)))
    for b in range(len(beta)):
        results = [executeAPDjit(p,b) for p in pset]
        V = np.array([results[i][1] for i in range(len(results))])
        P = np.array([results[i][0] for i in range(len(results))])
        Pstar[:, :, b] = P[np.argmax(V, axis = 0), :]
        del(results, V, P)
    return Pstar

def allPT(Pt):
    Pt[:, 1:] = 1
    return Pt

def poissonProcess(R, aB, bL, bB, prices, p, remainCap):
    low = np.sum(R==0) # number low types
    high = np.sum(R==1) 
    if remainCap == 0:
        sales = 0
        sales_B = 0
        sales_L = 0
        CS_L = 0 #np.nansum(T1EVout[R==0]/low)
        CS_B = 0 #np.nansum(T1EVout[R==1]/high)
        CS_ALL = 0 #T1EVout.sum() 
    elif remainCap > 0:
        sj = np.exp(aB + ((1 - R) * bL + R * bB) * prices[p]) / \
            (1 + np.exp(aB + ((1 - R) * bL + R * bB) * prices[p]))
        QD = (np.random.uniform(size = len(R)) < sj) * 1
        #QD = aB + ((1-R)*bL + R*bB)*prices[p] + T1EVin >= T1EVout
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

def storeSim(Store_fl, CCP, pu, pa, ps, p0, gamma, mu, prices,VAR):
    b = int(Store_fl[0, -1])
    beta = np.array(VAR[0:7])
    bL = np.minimum(VAR[7], VAR[8])
    bB = np.maximum(VAR[7], VAR[8])
    aL = beta[b]
    for t in range(60):
        Store_flt0D = Store_fl[t, 0] # dynamic pricing entry (0)
        Store_flt0U = Store_fl[t, 1] # APD pricing entry (1)
        Store_flt0A = Store_fl[t, 2] # uniform pricing entry (2)
        Store_flt0S = Store_fl[t, 3] # static pricing entry (2)
        Store_flt00 = Store_fl[t, 4] # static pricing entry (2)
        pstarU = pu[Store_fl[0, 0].astype("int"), b]
        pstarD = np.argmax(CCP[t, Store_fl[t, 0].astype("int"), :, b])
        pstarA = pa[Store_fl[0,2].astype("int"),t,b]
        pstarS = np.argmax(ps[t, Store_fl[t, 3].astype("int"), :, b])
        pstar0 = p0[t, b]
        pois = np.random.poisson(mu[t, b]) # Arrival draw
        R = bernoulli.rvs(gamma[t], size = pois) # draw types 
        # low = np.sum(R==0) # number low types
        # high = np.sum(R==1) # number high types
        # T1EVin = np.random.gumbel(loc=0.0, scale=1.0, size=pois)
        # T1EVout = np.random.gumbel(loc=0.0, scale=1.0, size=pois)
        # pathOut is sales, CS_L, CS_B, CS_ALL
        if pois > 0:        
            Store_fl[t, 10], Store_fl[t, 15], Store_fl[t, 20], Store_fl[t, 25], Store_fl[t, 30], Store_fl[t, 35] = poissonProcess(R, aL, bL, bB, prices, pstarD, Store_flt0D)
            Store_fl[t, 11], Store_fl[t, 16], Store_fl[t, 21], Store_fl[t, 26], Store_fl[t, 31], Store_fl[t, 36] = poissonProcess(R, aL, bL, bB, prices, pstarU, Store_flt0U)
            Store_fl[t, 12], Store_fl[t, 17], Store_fl[t, 22], Store_fl[t, 27], Store_fl[t, 32], Store_fl[t, 37] = poissonProcess(R, aL, bL, bB, prices, pstarA, Store_flt0A)
            Store_fl[t, 13], Store_fl[t, 18], Store_fl[t, 23], Store_fl[t, 28], Store_fl[t, 33], Store_fl[t, 38] = poissonProcess(R, aL, bL, bB, prices, pstarS, Store_flt0S)
            Store_fl[t, 14], Store_fl[t, 19], Store_fl[t, 24], Store_fl[t, 29], Store_fl[t, 34], Store_fl[t, 39] = poissonProcess(R, aL, bL, bB, prices, pstar0, Store_flt00)
        else:
            Store_fl[t, 10:] = 0
        # now estimate consumer surplus
        Store_fl[t, 5] = prices[pstarD]
        Store_fl[t, 6] = prices[pstarU]
        Store_fl[t, 7] = prices[pstarA]
        Store_fl[t, 8] = prices[pstarS]
        Store_fl[t, 9] = prices[pstar0]
        Store_fl[t + 1, 0] = Store_flt0D-Store_fl[t, 10]
        Store_fl[t + 1, 1] = Store_flt0U-Store_fl[t, 11]
        Store_fl[t + 1, 2] = Store_flt0A-Store_fl[t, 12]
        Store_fl[t + 1, 3] = Store_flt0S-Store_fl[t, 13]
        Store_fl[t + 1, 4] = Store_flt00-Store_fl[t, 14]
        Store_fl[t, 40] = np.sum(R == 0) 
        Store_fl[t, 41] = np.sum(R == 1) 
    Store_fl[:, 42] = b
    return Store_fl



def transformResults(S,market):
    numSim = len(S)
    sx2 = S.reshape((-1, 43))
    df = pd.DataFrame(
        {
            "capD": sx2[:, 0],
            "capU": sx2[:, 1],
            "capA": sx2[:, 2], 
            "capS": sx2[:, 3], 
            "cap0": sx2[:, 4], 
            "priceD": sx2[:, 5] * 100,
            "priceU": sx2[:, 6] * 100,
            "priceA": sx2[:, 7] * 100,
            "priceS": sx2[:, 8] * 100,
            "price0": sx2[:, 9] * 100,
            "salesD": sx2[:, 10],
            "salesU": sx2[:, 11],
            "salesA": sx2[:, 12],
            "salesS": sx2[:, 13],
            "sales0": sx2[:, 14],
            "CS_D_L": sx2[:, 15],
            "CS_U_L": sx2[:, 16],
            "CS_A_L": sx2[:, 17],
            "CS_S_L": sx2[:, 18],
            "CS_0_L": sx2[:, 19],
            "CS_D_B": sx2[:, 20],
            "CS_U_B": sx2[:, 21],
            "CS_A_B": sx2[:, 22],
            "CS_S_B": sx2[:, 23],
            "CS_0_B": sx2[:, 24],
            "CS_D_ALL": sx2[:, 25],
            "CS_U_ALL": sx2[:, 26],
            "CS_A_ALL": sx2[:, 27],
            "CS_S_ALL": sx2[:, 28],
            "CS_0_ALL": sx2[:, 29],
            "salesD_B": sx2[:, 30],
            "salesU_B": sx2[:, 31],
            "salesA_B": sx2[:, 32],
            "salesS_B": sx2[:, 33],
            "sales0_B": sx2[:, 34],
            "salesD_L": sx2[:, 35],
            "salesU_L": sx2[:, 36],
            "salesA_L": sx2[:, 37],
            "salesS_L": sx2[:, 38],
            "sales0_L": sx2[:, 39],
            "arrivals_L": sx2[:, 40],
            "arrivals_B": sx2[:, 41],
            "dow": sx2[:, 42]
        }
    )
    df["t"] = np.tile(range(61), numSim)
    df["fl"] = np.repeat(range(numSim), 61)
    df.loc[df.capU == 0, "priceU"] = np.nan
    df.loc[df.capU == 0, "salesU"] = np.nan
    df.loc[df.capD == 0, "priceD"] = np.nan
    df.loc[df.capD == 0, "salesD"] = np.nan
    df.loc[df.capA == 0, "priceA"] = np.nan
    df.loc[df.capA == 0, "salesA"] = np.nan
    df.loc[df.capS == 0, "priceS"] = np.nan
    df.loc[df.capS == 0, "salesS"] = np.nan
    df.loc[df.cap0 == 0, "price0"] = np.nan
    df.loc[df.cap0 == 0, "sales0"] = np.nan
    df["revD"] = df.salesD * df.priceD
    df["revU"] = df.salesU * df.priceU
    df["revA"] = df.salesA * df.priceA
    df["revS"] = df.salesS * df.priceS
    df["rev0"] = df.sales0 * df.price0
    df["market"] = market
    df.loc[df.capU == 0, "salesU_B"] = np.nan
    df.loc[df.capU == 0, "salesU_L"] = np.nan
    df.loc[df.capD == 0, "salesD_B"] = np.nan
    df.loc[df.capD == 0, "salesD_L"] = np.nan
    df.loc[df.capA == 0, "salesA_B"] = np.nan
    df.loc[df.capA == 0, "salesA_L"] = np.nan
    df.loc[df.capS == 0, "salesS_B"] = np.nan
    df.loc[df.capS == 0, "salesS_L"] = np.nan
    df.loc[df.cap0 == 0, "sales0_B"] = np.nan
    df.loc[df.cap0 == 0, "sales0_L"] = np.nan
    # mark type CS = np.nan if no one arrives
    df.loc[df.arrivals_L == 0, "CS_D_L"] = np.nan
    df.loc[df.arrivals_L == 0, "CS_U_L"] = np.nan
    df.loc[df.arrivals_L == 0, "CS_A_L"] = np.nan
    df.loc[df.arrivals_L == 0, "CS_S_L"] = np.nan
    df.loc[df.arrivals_L == 0, "CS_0_L"] = np.nan
    df.loc[df.arrivals_B == 0, "CS_D_B"] = np.nan
    df.loc[df.arrivals_B == 0, "CS_U_B"] = np.nan
    df.loc[df.arrivals_B == 0, "CS_A_B"] = np.nan
    df.loc[df.arrivals_B == 0, "CS_S_B"] = np.nan
    df.loc[df.arrivals_B == 0, "CS_0_B"] = np.nan
    # now create load factors
    df["LF_D"] = np.nan
    df.loc[df.t == 0, "LF_D"] = df.capD
    df["LF_U"] = np.nan
    df.loc[df.t == 0, "LF_U"] = df.capU
    df["LF_A"] = np.nan
    df.loc[df.t == 0, "LF_A"] = df.capA
    df["LF_S"] = np.nan
    df.loc[df.t == 0, "LF_S"] = df.capS
    df["LF_0"] = np.nan
    df.loc[df.t == 0, "LF_0"] = df.cap0
    df.LF_D.fillna(method = "ffill", inplace = True)
    df.LF_U.fillna(method = "ffill", inplace = True)
    df.LF_A.fillna(method = "ffill", inplace = True)
    df.LF_S.fillna(method = "ffill", inplace = True)
    df.LF_0.fillna(method = "ffill", inplace = True)
    df["LF_D"] = 100 * (1 - df.capD / df.LF_D)
    df["LF_U"] = 100 * (1 - df.capU / df.LF_U)
    df["LF_A"] = 100 * (1 - df.capA / df.LF_A)
    df["LF_S"] = 100 * (1 - df.capS / df.LF_S)
    df["LF_0"] = 100 * (1 - df.cap0 / df.LF_0)
    return df


if __name__ == "__main__":
    truncate = False
    pathIn = INPUT + market + "/"
    pathOut = INPUT + market + "/"
    df_route = pd.read_csv(pathIn + market + ".csv", index_col = 0)
    df_route_pt = pd.read_csv(pathIn + market + "_Pt.csv", header = None)
    prices = jnp.array(np.genfromtxt(pathIn + market + "_prices.csv"))
    Pt = df_route_pt.values
    data = np.array(df_route.values)
    qBar = int(np.max(df_route.seats)) + 1
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
        -VAR[9] - jnp.arange(0, 60) * VAR[10] - (jnp.arange(0, 60) ** 2) * VAR[11]
    ) + 1)
    # equivalent to jnp.array([1/(1 + jnp.exp(-g[0] + -t*g[1] - t**2*g[2])) for t in range(0,60)])
    # range(int(min(Tdata)),int(max(Tdata)+1))])
    muT = jnp.array(
        [VAR[12]] * (T - 20)+[VAR[13]] * 7 + [VAR[14]] * 7 + [VAR[15]] * 6
    )
    muD = jnp.append(jnp.array([1]), jnp.array(VAR[16:22]))
    mu = muT[:, None] * muD[None,:]
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
    # uniform Pricing
    print("now working on uniform pricing")
    pu = np.array(optUniform(f, ER, gamma, beta))
    # APD alone
    print("now working on APD alone")
    pa = np.array(optAPD(f, ER, gamma, qBar, Pt)).astype("int")
    #pa = np.ones((qBar,len(gamma), len(beta))).astype("int")
    # static firm
    print("now working on static pricing")
    ps = np.array(optStatic(f, ER, gamma, beta))
    rate = mu[:, None, :] * purchIn(beta, bB, bL, gamma)
    p0 = np.array(optStaticInf(rate, prices, Pt))
    VAR = np.array(VAR)
    gamma = np.array(gamma)
    mu = np.array(mu)
    prices = np.array(prices)
    rate = np.array(rate)
    # ADJUST NUMBER OF SIMULATIONS
    StoreX = np.zeros((numSim, 61, 43)) # f, t, (SeatsRemain, P, Sales, Larrive, Harrive, Lbuy, Hbuy)
    Q0 = df_route.loc[df_route.tdate == 0] \
        .sample(numSim, replace = True)[["seats", "dd_dow"]].values
    StoreX [:, 0, 0] = Q0[:,0].astype("int")
    StoreX [:, 0, -1] = Q0[:,1].astype("int")
    StoreX [:, 0, 0] = Q0[:,0].astype("int")
    StoreX [:, 0, 1] = Q0[:,0].astype("int")
    StoreX [:, 0, 2] = Q0[:,0].astype("int")
    StoreX [:, 0, 3] = Q0[:,0].astype("int")
    StoreX [:, 0, 4] = Q0[:,0].astype("int") 
    #pool2 = Pool(processes=numThreads)
    with Pool(processes=numThreads) as pool2:
        results = np.array(
            pool2.map(
                partial(
                    storeSim,
                    CCP = CCP,
                    pu = pu,
                    pa = pa,
                    ps = ps,
                    p0 = p0,
                    gamma = gamma,
                    mu = mu,
                    prices = prices,
                    VAR = VAR
                ),
                StoreX
            )
        )
        pool2.close()
        pool2.join()
    df = transformResults(results, market)
    df.to_parquet(pathOut  + market + "_counterfactuals.parquet")
    end = time.time()
    print("finished counterfactual in " + str(end - start))
