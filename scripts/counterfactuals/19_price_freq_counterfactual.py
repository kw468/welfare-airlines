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
import matplotlib.pyplot as plt
import seaborn as sns





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
pathData = "/gpfs/home/kw468/airlines_jmp/"
#pathData = "/home/kw468/Projects/airlines_jmp/"

def get_gpu_memory():
    _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
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
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)


# -------------------------------------------------------------------------------
# DEFINE FUNCTIONS FOR DEMAND
# -------------------------------------------------------------------------------

# define the probability that a low type wants to buy
# rewrite exp(a)/(1 + exp(a)) as 1/(1/exp(a) + 1) = 1/(1 + exp(-a))
@jit
def purchInL(beta,bL,gamma):   # p is dim(P), gam is dim(T), , beta is dim FE
    return (1 - gamma)[:,None,None]*1/(1 + jnp.exp(-beta[None,None,:] - bL*prices[None,:,None]))
    # returned object is T,P dimension



# define the probability that a high type wants to buy
@jit
def purchInB(beta,bB,gamma):   # p is dim(P), gam is dim(T), beta is dim FE
    return (gamma)[:,None,None]*1/(1 + jnp.exp(-beta[None,None,:] - bB*prices[None,:,None]))
    # returned object is T,P dimension



# define, this returns a T X P matrix of probability of purchase across both consumer types
@jit
def purchIn(beta,bB,bL,gamma):
    return purchInL(beta,bL,gamma) + purchInB(beta,bB,gamma)



# define the log probability of that demand is equal to q given the parameters of the model
@jit
def log_demandQ(beta, bL, bB, gamma, mu, q):
    return q*jnp.log(mu[:,None,:]) + q*jnp.log(purchIn(beta,bB,bL,gamma))  -  (mu[:,None,:]*purchIn(beta,bB,bL,gamma)) - gammaln(q + 1)



# define the probability of that demand is equal to q given the parameters of the model, this is just exp(log(p)),
# which is done for numerical stablility
@jit
def demandQ(beta, bL, bB, gamma, mu, q):
    #return (mu[:,None]*purchIn(aB,bB,aL,bL,prices,gamma))**q*jnp.exp(-mu[:,None]*(purchIn(aB,bB,aL,bL,prices,gamma)))/factQ[q]
    return jnp.exp(log_demandQ(beta, bL, bB, gamma, mu, q))



def allDemand(beta, bL, bB, gamma, mu):
    vlookup = vmap((lambda x: demandQ(beta, bL, bB, gamma, mu, x)))(jnp.arange(0,qBar))
    f                               = jnp.zeros((qBar,qBar,len(gamma), len(prices), len(beta)))
    for q in range(1,qBar):         # seats remaining
        f                   = f.at[q,0:q,:,:,:].set(vlookup[0:q,:,:,:])
        f                   = f.at[q,q,:,:,:].set(jnp.maximum(1 - jnp.sum(f[q,0:(q),:,:,:],axis=0),1e-100))
    f                               = jnp.where((f > 0) & (f < 1e-100), 1e-100, f)
    return f
    # returning dimension is seats remaining, seats sold, time, and prices


allDemand_jit           = jit(allDemand)




# define the CCP and EV for the dynamic firm problem
def optDynNoError(f,ER,gamma,beta):
    np.seterr(all='raise')
        # create storage for EV, V, CCP
    EV  = jnp.zeros((T,qBar,numP,len(beta)))
    V   = jnp.zeros((qBar,T,len(beta)))
    CCP = jnp.zeros((T,qBar,numP,len(beta)))
    for t in range(1,T+1):
        # work backwards in time. In the last period, we just get last period revenues
        if t == 1:
            # the softmax functin can be rewritten, so let's use logsum(exp) = x* + log sum (exp (x-x*))
            grp         = ER[:,-t,:,:]*Pt[-t,1:][None,:,None]
            V           = V.at[:,-t,:].set( jnp.max(grp , axis = 1) )
            #tmp         = jnp.argmax(grp , axis = 1) 
            # FINISH THIS
            #for q in range(qBar):
            #    for b in range(len(beta)):
            #        CCP = CCP.at[-t,q,tmp[q,b],b].set(1)
            tmp = jnp.zeros_like(grp)
            tmp = tmp.at[jnp.where(grp   == jnp.max(grp , axis = 1)[:,None,:] )].set(1)
            CCP = CCP.at[-t,:,:,:].set(tmp)
        else:
            grp         = (ER[:,-t,:,:] + EV[-t+1,:,:,:])*Pt[-t,1:][None,:,None]
            V           = V.at[:,-t,:].set( jnp.max(grp , axis = 1) )
        # this allows us to calc that the Pr(Q = 1) aligns with Pr(c' = c - 1)   
        r,c = jnp.tril_indices_from(f[:,:,0,0,0])
        for b in range(len(beta)):
            if t != T: #update expected value function, this is for not the last period
                g               = jnp.array(f[:,:,-t-1,:,b])
                g               = g.at[r,c,:].set(g[r,r-c,:])
                EV              = EV.at[-t,:,:,b].set(jnp.sum(g*V[:,-t,b][None,:,None],axis=1)*Pt[-t,1:])
        if t != 1:
            XX                  = (ER[:,-t,:,:] + EV[-t+1,:,:,:])*Pt[-t,1:][None,:,None]
            # tmp             = jnp.argmax(XX , axis = 1) 
            # # FINISH THIS
            # for q in range(qBar):
            #     for b in range(len(beta)):
            #         CCP = CCP.at[-t,q,tmp[q,b],b].set(1)
            tmp = jnp.zeros_like(XX)
            tmp = tmp.at[jnp.where(XX   == jnp.max(XX , axis = 1)[:,None,:] )].set(1)
            CCP = CCP.at[-t,:,:,:].set(tmp)
    return CCP




def optJump(f,ER,gamma,beta,jump):
    np.seterr(all='raise')
    EV  = jnp.zeros((T,qBar,numP,len(beta)))
    V   = jnp.zeros((qBar,T,len(beta)))
    CCP = np.zeros((T,qBar,numP,len(beta)))
    l                               = np.arange(1,61)
    chunks                          = [list(l[i:i+jump]) for i in np.arange(0, len(l), jump)]
    for ch in chunks:
        for t in ch[:-1]:
            # work backwards in time. In the last period, we just get last period revenues
            if t == 1:
                grp         = ER[:,-t,:,:]*Pt[-t,1:][None,:,None]
                V           = V.at[:,-t,:].set( jnp.max(grp , axis = 1) )
            else:
                grp         = (ER[:,-t,:,:] + EV[-t+1,:,:,:])*Pt[-t,1:][None,:,None]
                V           = V.at[:,-t,:].set( jnp.max(grp , axis = 1) )
            r,c = jnp.tril_indices_from(f[:,:,0,0,0])
            for b in range(len(beta)):
                if t != T: #update expected value function, this is for not the last period
                    g               = jnp.array(f[:,:,-t-1,:,b])
                    g               = g.at[r,c,:].set(g[r,r-c,:])
                    EV              = EV.at[-t,:,:,b].set(jnp.sum(g*V[:,-t,b][None,:,None],axis=1)*Pt[-t,1:])
        t                           = ch[-1]
        grp                         = (ER[:,-t,:,:] + EV[-t+1,:,:,:])*Pt[-t,1:][None,:,None]
        interV                      = jnp.max(grp , axis = 1)
        pstar = jnp.zeros_like(grp)
        pstar = pstar.at[jnp.where(grp   == jnp.max(grp , axis = 1)[:,None,:] )].set(1)
        if t != T:
            r,c = jnp.tril_indices_from(f[:,:,0,0,0])
            for b in range(len(beta)):
                if t != T: #update expected value function, this is for not the last period
                    g               = jnp.array(f[:,:,-t-1,:,b])
                    g               = g.at[r,c,:].set(g[r,r-c,:])
                    EV              = EV.at[-t,:,:,b].set(jnp.sum(g*interV[:,b][None,:,None],axis=1)*Pt[-t,1:])
        for q in range(qBar):
            for b in range(len(beta)):
                if (-ch[-1]+jump) == 0:
                    CCP[-ch[-1]:,q,int(pstar[q,:,b].argmax()),b]                = 1
                else:
                    CCP[-ch[-1]:(-ch[-1]+jump),q,int(pstar[q,:,b].argmax()),b]  = 1
    return CCP


# Calculate the opt uniform price
def optUniform(f,ER,gamma,beta):
    EV  = jnp.zeros((qBar,len(gamma), len(prices), len(beta)))
    V   = jnp.zeros((qBar,len(gamma), len(prices), len(beta)))
    #jnp.zeros((qBar,len(gamma), len(prices), len(beta)))
    for t in range(1,T+1):
        # work backwards in time. In the last period, we just get last period revenues
        if t == 1:
            V                       = V.at[:,-t,:,:].set( ER[:,-t, :, :])
        else:
            V                       = V.at[:,-t,:,:].set( ER[:,-t, :, :] + EV[:,-t+1, :, :] )
        r,c = jnp.tril_indices_from(f[:,:,0,0,0])
        for b in range(len(beta)):
            if t != T: #update expected value function, this is for not the last period
                g               = jnp.array(f[:,:,-t-1,:,b])
                g               = g.at[r,c,:].set(g[r,r-c,:])
                EV              = EV.at[:,-t,:,b].set(jnp.sum(g*V[:,-t,:,b][None,:,:],axis=1))
    pstar                       = jnp.argmax(V[:,0,:,:],axis=1)
    return pstar


def allPT(Pt):
    Pt[:,1:]    = 1
    return Pt




def poissonProcess(R,aB,bL,bB,prices,p,remainCap):
    low             =   np.sum(R==0)                                        # number low types
    high            =   np.sum(R==1) 
    if remainCap == 0:
        sales       = 0
    elif remainCap > 0:
        sj              =   np.exp(aB + ((1-R)*bL + R*bB)*prices[p]) / (1 + np.exp(aB + ((1-R)*bL + R*bB)*prices[p]))
        QD              =   (np.random.uniform(size=len(R)) < sj) * 1
        if QD.sum() <= remainCap:
            sales       = QD.sum()
        elif QD.sum() > remainCap:        
            sales       = remainCap
    return sales



def storeSim(Store_fl, CCP, CCP2, CCP3, CCP6, CCP10, CCP20, CCP30, pu, gamma, mu, prices,VAR):
    b                   =       int(Store_fl[0, -1])
    beta                =       np.array(VAR[0:7])
    bL                  =       np.minimum(VAR[7], VAR[8])
    bB                  =       np.maximum(VAR[7], VAR[8])
    aL                  =       beta[b]
    for t in range(60):
        Store_flt0D     =   Store_fl[t, 0]                                      # dynamic pricing entry (0)
        Store_flt02     =   Store_fl[t, 1]                                      # 2 day adjust
        Store_flt03     =   Store_fl[t, 2]                                      # 3 day adjust
        Store_flt06     =   Store_fl[t, 3]                                      # 6 day adjust
        Store_flt010    =   Store_fl[t, 4]                                      # 10 day adjust
        Store_flt020    =   Store_fl[t, 5]                                      # 20 day adjust
        Store_flt030    =   Store_fl[t, 6]                                      # 30 day adjust
        Store_flt0U     =   Store_fl[t, 7] 
        pstarU          =   pu[Store_fl[0,0].astype("int"),b]
        pstarD          =   np.argmax(CCP[t,   Store_fl[t, 0].astype("int"), :,b])
        pstarD2         =   np.argmax(CCP2[t,  Store_fl[t, 1].astype("int"), :,b])
        pstarD3         =   np.argmax(CCP3[t,  Store_fl[t, 2].astype("int"), :,b])
        pstarD6         =   np.argmax(CCP6[t,  Store_fl[t, 3].astype("int"), :,b])
        pstarD10        =   np.argmax(CCP10[t, Store_fl[t, 4].astype("int"), :,b])
        pstarD20        =   np.argmax(CCP20[t, Store_fl[t, 5].astype("int"), :,b])
        pstarD30        =   np.argmax(CCP30[t, Store_fl[t, 6].astype("int"), :,b])
        pois            =   np.random.poisson(mu[t,b])                              # Arrival draw
        R               =   bernoulli.rvs(gamma[t], size=pois)                      # draw types 
        # output is sales
        if pois > 0:        
            Store_fl[t, 16] = poissonProcess(R,aL,bL,bB,prices,pstarD,Store_flt0D)
            Store_fl[t, 17] = poissonProcess(R,aL,bL,bB,prices,pstarD2,Store_flt02)
            Store_fl[t, 18] = poissonProcess(R,aL,bL,bB,prices,pstarD3,Store_flt03)
            Store_fl[t, 19] = poissonProcess(R,aL,bL,bB,prices,pstarD6,Store_flt06)
            Store_fl[t, 20] = poissonProcess(R,aL,bL,bB,prices,pstarD10,Store_flt010)
            Store_fl[t, 21] = poissonProcess(R,aL,bL,bB,prices,pstarD20,Store_flt020)
            Store_fl[t, 22] = poissonProcess(R,aL,bL,bB,prices,pstarD30,Store_flt030)
            Store_fl[t, 23] = poissonProcess(R,aL,bL,bB,prices,pstarU,Store_flt0U)
        else:
            Store_fl[t, 15:] = 0
        # now estimate consumer surplus
        Store_fl[t, 8]       =   prices[pstarD]
        Store_fl[t, 9]       =   prices[pstarD2]
        Store_fl[t, 10]       =   prices[pstarD3]
        Store_fl[t, 11]      =   prices[pstarD6]
        Store_fl[t, 12]      =   prices[pstarD10]
        Store_fl[t, 13]      =   prices[pstarD20]
        Store_fl[t, 14]      =   prices[pstarD30]
        Store_fl[t, 15]      =   prices[pstarU]
        Store_fl[t+1, 0]     =   Store_flt0D-Store_fl[t, 16]
        Store_fl[t+1, 1]     =   Store_flt02-Store_fl[t, 17]
        Store_fl[t+1, 2]     =   Store_flt03-Store_fl[t, 18]
        Store_fl[t+1, 3]     =   Store_flt06-Store_fl[t, 19]
        Store_fl[t+1, 4]     =   Store_flt010-Store_fl[t, 20]
        Store_fl[t+1, 5]     =   Store_flt020-Store_fl[t, 21]
        Store_fl[t+1, 6]     =   Store_flt030-Store_fl[t, 22]
        Store_fl[t+1, 7]     =   Store_flt0U-Store_fl[t, 23]
    return Store_fl



def transformResults(S,market):
    numSim                      = len(S)
    sx2                         = S.reshape((-1,24))
    df = pd.DataFrame({'capD':sx2[:,0],
        'cap2':sx2[:,1],
        'cap3':sx2[:,2], 
        'cap6':sx2[:,3], 
        'cap10':sx2[:,4], 
        'cap20':sx2[:,5], 
        'cap30':sx2[:,6], 
        'capU':sx2[:,7], 
        'priceD':sx2[:,8]*100,
        'price2':sx2[:,9]*100,
        'price3':sx2[:,10]*100,
        'price6':sx2[:,11]*100,
        'price10':sx2[:,12]*100,
        'price20':sx2[:,13]*100,
        'price30':sx2[:,14]*100,
        'priceU':sx2[:,15]*100,
        'salesD':sx2[:,16],
        'sales2':sx2[:,17],
        'sales3':sx2[:,18],
        'sales6':sx2[:,19],
        'sales10':sx2[:,20],
        'sales20':sx2[:,21],
        'sales30':sx2[:,22],
        'salesU':sx2[:,23]
         })
    df['t']                       = np.tile(range(61), numSim)
    df['fl']                      = np.repeat(range(numSim), 61)
    df.loc[df.capU ==0, 'priceU'] = np.nan
    df.loc[df.capU ==0, 'salesU'] = np.nan
    df.loc[df.capD ==0, 'priceD'] = np.nan
    df.loc[df.capD ==0, 'salesD'] = np.nan
    df.loc[df.cap2 ==0, 'price2'] = np.nan
    df.loc[df.cap2 ==0, 'sales2'] = np.nan
    df.loc[df.cap3 ==0, 'price3'] = np.nan
    df.loc[df.cap3 ==0, 'sales3'] = np.nan
    df.loc[df.cap6 ==0, 'price6'] = np.nan
    df.loc[df.cap6 ==0, 'sales6'] = np.nan
    df.loc[df.cap10 ==0, 'price10'] = np.nan
    df.loc[df.cap10 ==0, 'sales10'] = np.nan
    df.loc[df.cap20 ==0, 'price20'] = np.nan
    df.loc[df.cap20 ==0, 'sales20'] = np.nan
    df.loc[df.cap30 ==0, 'price30'] = np.nan
    df.loc[df.cap30 ==0, 'sales30'] = np.nan
    df["revD"]                    = df.salesD*df.priceD
    df["revU"]                    = df.salesU*df.priceU
    df["rev2"]                    = df.sales2*df.price2
    df["rev3"]                    = df.sales3*df.price3
    df["rev6"]                    = df.sales6*df.price6
    df["rev10"]                    = df.sales10*df.price10
    df["rev20"]                    = df.sales20*df.price20
    df["rev30"]                    = df.sales30*df.price30
    df["market"]                  = market
    return df


if __name__ == "__main__":
    truncate = False
    pathIn                  =       pathData + "estimation/" + market + "/"
    pathOut                 =       pathData + "estimation/" + market + "/"
    df_route                =       pd.read_csv(pathIn + market + ".csv", index_col=0)
    df_route_pt             =       pd.read_csv(pathIn + market + "_Pt.csv", header = None)
    prices                  =       jnp.array(np.genfromtxt(pathIn + market + "_prices.csv"))
    Pt                      =       df_route_pt.values
    data                    =       np.array(df_route.values)
    qBar                    =       int(np.max(df_route.seats))+1
    T                       =       len(np.unique(df_route.tdate))
    numP                    =       len(prices)
    obs                     =       len(df_route.tdate)
    if truncate == False:
        Pt[:,1:]            =       1
    Pt                      =       jnp.array(Pt)
    xInit                   =       np.genfromtxt(pathIn + "/robust_estim/" + market + "_robust_params.csv")
    VAR                     =       jnp.array(xInit)
    beta                    =       jnp.array(VAR[0:7])
    bL                      =       jnp.minimum(VAR[7], VAR[8])
    bB                      =       jnp.maximum(VAR[7], VAR[8])
    gamma                   =       1/(jnp.exp(-VAR[9] - jnp.arange(0,60)*VAR[10] - (jnp.arange(0,60)**2)*VAR[11]) + 1)
    # equivalent to jnp.array([1/(1 + jnp.exp(-g[0] + -t*g[1] - t**2*g[2])) for t in range(0,60)])#range(int(min(Tdata)),int(max(Tdata)+1))])
    muT                     =       jnp.array([VAR[12]]*(T-20)+[VAR[13]]*7 + [VAR[14]]*7 + [VAR[15]]*6 )
    muD                     =       jnp.append(jnp.array([1]),jnp.array(VAR[16:22]))
    mu                      =       muT[:,None]*muD[None,:]
    sig                     =       VAR[-1]
    rate                    =       mu[:,None,:]*purchIn(beta,bB,bL,gamma)
    ######################
    # define all sales possibilities
    f                       =       allDemand_jit(beta, bL, bB, gamma, mu)
        # define expected revenues
    ER                      =       jnp.sum(f*jnp.array(range(qBar))[None,:,None,None,None]*prices[None,None,None,:,None],axis=1)
    CCP                     =       np.array(optDynNoError(f,ER,gamma,beta))
    # uniform Pricing
    print("now working on uniform pricing")
    pu                      =       np.array(optUniform(f,ER,gamma,beta))
    # APD alone
    print("now working on APD alone")
    CCP2        = np.array(optJump(f,ER,gamma,beta, 2))
    CCP3        = np.array(optJump(f,ER,gamma,beta, 3))
    CCP6        = np.array(optJump(f,ER,gamma,beta, 6))
    CCP10       = np.array(optJump(f,ER,gamma,beta, 10))
    CCP20       = np.array(optJump(f,ER,gamma,beta, 20))
    CCP30       = np.array(optJump(f,ER,gamma,beta, 30))
    # convert to np
    VAR                     =       np.array(VAR)
    gamma                   =       np.array(gamma)
    mu                      =       np.array(mu)
    prices                  =       np.array(prices)
    rate                    =       np.array(rate)
    # ADJUST NUMBER OF SIMULATIONS
    StoreX                  =       np.zeros((numSim, 61, 24)) # f, t, (SeatsRemain, P, Sales, Larrive, Harrive, Lbuy, Hbuy)
    Q0                      =       df_route.loc[df_route.tdate == 0].sample(numSim, replace = True)[["seats", "dd_dow"]].values
    StoreX [:, 0, 0]        =       Q0[:,0].astype("int")
    StoreX [:, 0, 1]        =       Q0[:,0].astype("int")
    StoreX [:, 0, 2]        =       Q0[:,0].astype("int")
    StoreX [:, 0, 3]        =       Q0[:,0].astype("int")
    StoreX [:, 0, 4]        =       Q0[:,0].astype("int")
    StoreX [:, 0, 5]        =       Q0[:,0].astype("int") 
    StoreX [:, 0, 6]        =       Q0[:,0].astype("int") 
    StoreX [:, 0, 7]        =       Q0[:,0].astype("int") 
    StoreX [:, 0, -1]       =       Q0[:,1].astype("int")
    #pool2               = Pool(processes=numThreads)
    with Pool(processes=numThreads) as pool2:
        results             = np.array(pool2.map(partial(storeSim, CCP=CCP,CCP2=CCP2,CCP3=CCP3,CCP6=CCP6,CCP10=CCP10,CCP20=CCP20,CCP30=CCP30, pu=pu, gamma=gamma, mu=mu, prices=prices,VAR=VAR), StoreX))
        pool2.close()
        pool2.join()
    df                      = transformResults(results,market)
    df.to_parquet(pathOut  + market + "_priceFreq_counterfactuals.parquet")
    end                     = time.time()
    print("finished counterfactual in " + str(end - start))




