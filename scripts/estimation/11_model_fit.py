# this must be run in py_tf_plot

from multiprocessing import Pool
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
import os

pathData                = "/home/kw468/Projects/airlines_jmp/"
pathOutput              = "/home/kw468/Projects/airlines_jmp/output/"

os.chdir(pathData)
from estim_markets import *


EC = 0.5772156649
T  = 60
# --------------------------------------------------------------------------------
# Set program parameters
# --------------------------------------------------------------------------------
numThreads = 31
# adjust Jax to 64 bit floats
from jax.config import config
config.update("jax_enable_x64", True)


# def get_gpu_memory():
#     _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
#     ACCEPTABLE_AVAILABLE_MEMORY = 1024
#     COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
#     memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
#     memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
#     return memory_free_values

# gpuUsage = get_gpu_memory()

# gpu = 0
# if gpuUsage[0] < 10000:
#     gpu = 1
#     if gpuUsage[1] < 10000:
#         gpu = 2


# print("using gpu number: " + str(gpu))
# os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)


# -------------------------------------------------------------------------------
# DEFINE FUNCTIONS FOR DEMAND
# -------------------------------------------------------------------------------


# define the probability that a low type wants to buy
# rewrite exp(a)/(1 + exp(a)) as 1/(1/exp(a) + 1) = 1/(1 + exp(-a))
def purchInL(beta,bL,gamma,prices):   # p is dim(P), gam is dim(T), , beta is dim FE
    return (1 - gamma)[:,None,None]*1/(1 + jnp.exp(-beta[None,None,:] - bL*prices[None,:,None]))
    # returned object is T,P dimension



# define the probability that a high type wants to buy
def purchInB(beta,bB,gamma,prices):   # p is dim(P), gam is dim(T), beta is dim FE
    return (gamma)[:,None,None]*1/(1 + jnp.exp(-beta[None,None,:] - bB*prices[None,:,None]))
    # returned object is T,P dimension



# define, this returns a T X P matrix of probability of purchase across both consumer types
def purchIn(beta,bB,bL,gamma,prices):
    return purchInL(beta,bL,gamma,prices) + purchInB(beta,bB,gamma,prices)



# define the log probability of that demand is equal to q given the parameters of the modelt
def log_demandQ(beta, bL, bB, gamma, mu, prices, q):
    return q*jnp.log(mu[:,None,:]) + q*jnp.log(purchIn(beta,bB,bL,gamma,prices))  -  (mu[:,None,:]*purchIn(beta,bB,bL,gamma,prices)) - gammaln(q + 1)




# define the probability of that demand is equal to q given the parameters of the model, this is just exp(log(p)),
# which is done for numerical stablility
def demandQ(beta, bL, bB, gamma, mu, prices, q):
    #return (mu[:,None]*purchIn(aB,bB,aL,bL,prices,gamma))**q*jnp.exp(-mu[:,None]*(purchIn(aB,bB,aL,bL,prices,gamma)))/factQ[q]
    return jnp.exp(log_demandQ(beta, bL, bB, gamma, mu, prices, q))



def allDemand(beta, bL, bB, gamma, mu,qBar,prices):
    vlookup = vmap((lambda x: demandQ(beta, bL, bB, gamma, mu, prices, x)))(jnp.arange(0,qBar))
    f                               = jnp.zeros((qBar,qBar,len(gamma), len(prices), len(beta)))
    for q in range(1,qBar):         # seats remaining
        f                   = f.at[q,0:q,:,:,:].set(vlookup[0:q,:,:,:])
        f                   = f.at[q,q,:,:,:].set(jnp.maximum(1 - jnp.sum(f[q,0:(q),:,:,:],axis=0),1e-100))
    f                               = jnp.where((f > 0) & (f < 1e-100), 1e-100, f)
    return f
    # returning dimension is seats remaining, seats sold, time, and prices


# define the CCP and EV for the dynamic firm problem
def dynEst(f,ER,gamma,sig,beta,Pt,qBar,prices):
    np.seterr(all='raise')
        # create storage for EV, V, CCP
    EV  = jnp.zeros((T,qBar,len(prices),len(beta)))
    V   = jnp.zeros((qBar,T,len(beta)))
    CCP = jnp.zeros((T,qBar,len(prices),len(beta)))
    for t in range(1,T+1):
        # work backwards in time. In the last period, we just get last period revenues
        if t == 1:
            # the softmax functin can be rewritten, so let's use logsum(exp) = x* + log sum (exp (x-x*))
            grp         = ER[:,-t,:,:]/(sig)*Pt[-t,1:][None,:,None]
            grp         = jnp.where(grp == 0, -jnp.inf, grp) 
            V           = V.at[:,-t,:].set(sig*logsumexp(grp , axis = 1) + EC*sig )
            V           = V.at[0,-t,:].set(0)
            CCP             = CCP.at[-t,:,:,:].set(grp - logsumexp(grp , axis = 1)[:,None,:]) #CCP[-t,:,:] = Pt[-t,1:][None,:]*np.exp(ER[:,-t,:]/sig)/np.sum(np.exp(ER[:,-t,:]*Pt[-t,1:][None,:]/sig),axis=1)[:,None]
        else:
            grp         = (ER[:,-t,:,:]/sig + EV[-t+1,:,:,:]/sig)*Pt[-t,1:][None,:,None]
            grp         = jnp.where(grp == 0, -jnp.inf, grp) 
            V           = V.at[:,-t,:].set(sig*logsumexp(grp , axis = 1) + EC*sig )
            V           = V.at[0,-t,:].set(0)
        # now we need to define EV, which is int_c' V f(c'), so we'll use tril to reset array(x) as reversearray(x)
        # this allows us to calc that the Pr(Q = 1) aligns with Pr(c' = c - 1)   
        r,c = jnp.tril_indices_from(f[:,:,0,0,0])
        for b in range(len(beta)):
            if t != T: #update expected value function, this is for not the last period
                g               = jnp.array(f[:,:,-t-1,:,b])
                g               = g.at[r,c,:].set(g[r,r-c,:])
                EV              = EV.at[-t,:,:,b].set(jnp.sum(g*V[:,-t,b][None,:,None],axis=1)*Pt[-t,1:])
        if t != 1:
            XX              = (ER[:,-t,:,:] + EV[-t+1,:,:,:])/sig*Pt[-t,1:][None,:,None]
            XX              = jnp.where(XX == 0, -jnp.inf, XX) 
            CCP             = CCP.at[-t,:,:,:].set(XX - logsumexp(XX , axis = 1)[:,None,:])
    return CCP, EV



def process(market):
    np.seterr(over='ignore')
    np.seterr(under='ignore')
    pathIn                  = pathData + "estimation/" + market + "/"
    df_route                = pd.read_csv(pathIn + market + ".csv", index_col=0)
    df_route_pt             = pd.read_csv(pathIn + market + "_Pt.csv", header = None)
    prices                  = jnp.array(np.genfromtxt(pathIn + market + "_prices.csv"))
    xInit = np.genfromtxt(pathIn + "/robust_estim/" + market + "_robust_params.csv")
    Pt                      = df_route_pt.values
    data                    = np.array(df_route.values)
    qBar                    = int(np.max(df_route.seats))+1
    obs                     = len(df_route.tdate)
    Pt                      = jnp.array(Pt)
    EC                      = 0.5772156649
    VAR                     = jnp.array(xInit)
    # The first parameters are consumer preferences:
    beta                    =       jnp.array(VAR[0:7])
    bL                      =       jnp.minimum(VAR[7], VAR[8])
    bB                      =       jnp.maximum(VAR[7], VAR[8])
    gamma                   =       1/(jnp.exp(-VAR[9] - jnp.arange(0,60)*VAR[10] - (jnp.arange(0,60)**2)*VAR[11]) + 1)
    # equivalent to jnp.array([1/(1 + jnp.exp(-g[0] + -t*g[1] - t**2*g[2])) for t in range(0,60)])#range(int(min(Tdata)),int(max(Tdata)+1))])
    muT                     =       jnp.array([VAR[12]]*(T-20)+[VAR[13]]*7 + [VAR[14]]*7 + [VAR[15]]*6 )
    muD                     =       jnp.append(jnp.array([1]),jnp.array(VAR[16:22]))
    mu                      =       muT[:,None]*muD[None,:]
    sig                     =       VAR[-1]
    ######################
    # define all sales possibilities
    f0                      =       allDemand(beta, bL, bB, gamma, mu, qBar, prices)
    # define expected revenues
    ER0                     =       jnp.sum(f0*jnp.array(range(qBar))[None,:,None,None,None]*prices[None,None,None,:,None],axis=1)
    # solve the DP, calculate the CCP and EV
    CCP, EV                 =       dynEst(f0,ER0,gamma,sig,beta,Pt,qBar,prices)
    Pt                      =       np.array(Pt)
    Pt[1:,:]                =       1
    Pt                      =       jnp.array(Pt)
    CCP1, EV1               =       dynEst(f0,ER0,gamma,sig,beta,Pt,qBar,prices)
    prices                  =       np.array(prices)
    CCP                     =       np.array(CCP)
    CCP[CCP < -300]         =       -300 #prevents underflow
    CCP1                    =       np.array(CCP1)
    CCP1[CCP1 < -300]       =       -300
    df                      = pd.DataFrame()
    df['prices_sim_Pt']     = (100*np.exp(CCP[df_route.tdate.values,df_route.seats.values,:,df_route.dd_dow.values])*prices[None,:]).sum(axis=1)
    df['prices_sim_P1']     = (100*np.exp(CCP1[df_route.tdate.values,df_route.seats.values,:,df_route.dd_dow.values])*prices[None,:]).sum(axis=1)
    df['prices_actual']     = 100*prices[df_route.fareI.values]
    df["tdate"]             = df_route.tdate.values
    return df






df      = pd.DataFrame()
for m in mkts:
    print(m)
    df = df.append(process(m))


csfont                      = {'fontname':"Liberation Serif", 'fontsize':18}
palette                     = ["#FF6700", "#FCB07E", "#6B717E", "#3A6EA5", "#004E98", "#070707"]
sns.set(style="white",color_codes=False)
fig                         = plt.figure(figsize=(1.5*6.4, 1.1*4.8))
plt.plot(df.groupby("tdate").prices_actual.mean(), label='Mean: Data',color=palette[2],linewidth = 3, linestyle='-')
plt.plot(df.groupby("tdate").prices_sim_Pt.mean(), label='Mean: Model',color=palette[0],linewidth = 3, linestyle='--')
plt.plot(df.groupby("tdate").prices_sim_P1.mean(), label='Mean: Model, no fare restrictions',color=palette[3],linewidth = 3, linestyle=':')
plt.plot(df.groupby("tdate").prices_actual.quantile(.95), label='5th,95th Percentiles: Data',color=palette[2],linewidth = 1, linestyle='-')
plt.plot(df.groupby("tdate").prices_sim_P1.quantile(.95), label='5th,95th Percentiles: Model',color=palette[0],linewidth = 1, linestyle='--')
plt.plot(df.groupby("tdate").prices_actual.quantile(.05), label='_nolegend_',color=palette[2],linewidth = 1, linestyle='-')
plt.plot(df.groupby("tdate").prices_sim_P1.quantile(.05), label='_nolegend_',color=palette[0],linewidth = 1, linestyle='--')
L                           = plt.legend()
plt.setp(L.texts, family='Liberation Serif', fontsize = 18)
plt.xlabel('Booking Horizon',**csfont)
plt.ylabel('Magnitude ($)',**csfont)
plt.axvline(x=54,color = palette[2],linewidth = 2,linestyle=':')
plt.axvline(x=47,color = palette[2],linewidth = 2,linestyle=':')
plt.axvline(x=40,color = palette[2],linewidth = 2,linestyle=':')
plt.yticks(fontname = "Liberation Serif", fontsize = 18) 
plt.xticks(fontname = "Liberation Serif", fontsize = 18) 
name = "model_fit.pdf"
plt.savefig(pathOutput + name,bbox_inches='tight',format= "pdf",dpi=600)
plt.close()


# if __name__ == '__main__':
#     df = pd.DataFrame()
#     for it in list(params[0].unique()):
#         df = df.append(processAllPaths(it))
#     #    
#     df = df.reset_index(drop=True)
#     #
#     #
#     fig                     = plt.figure(figsize=(1.5*6.4, 1.1*4.8))
#     # NOW PLOT THE RESULTS
#     colors                  = ["windows blue", "greyish", "black"]
#     sns.set(style="white",palette=sns.xkcd_palette(colors),color_codes=False)
#     plt.plot(df.groupby("tdate").prices_actual.mean(), label='Mean: Data',color=sns.xkcd_palette(colors)[2],linewidth = 3, linestyle='-')
#     plt.plot(df.groupby("tdate").prices_sim_Pt.mean(), label='Mean: Model',color=sns.xkcd_palette(colors)[1],linewidth = 3, linestyle='--')
#     plt.plot(df.groupby("tdate").prices_sim_P1.mean(), label='Mean: Model, no fare restrictions',color=sns.xkcd_palette(colors)[0],linewidth = 3, linestyle=':')
#     plt.plot(df.groupby("tdate").prices_actual.quantile(.95), label='5th,95th Percentiles: Data',color=sns.xkcd_palette(colors)[2],linewidth = 1, linestyle='-')
#     plt.plot(df.groupby("tdate").prices_sim_P1.quantile(.95), label='5th,95th Percentiles: Model',color=sns.xkcd_palette(colors)[1],linewidth = 1, linestyle='--')
#     plt.plot(df.groupby("tdate").prices_actual.quantile(.05), label='_nolegend_',color=sns.xkcd_palette(colors)[2],linewidth = 1, linestyle='-')
#     plt.plot(df.groupby("tdate").prices_sim_P1.quantile(.05), label='_nolegend_',color=sns.xkcd_palette(colors)[1],linewidth = 1, linestyle='--')
#     #
#     plt.xlabel("Booking Horizon")
#     plt.ylabel("Fare Response")
#     plt.axvline(x=54,color = sns.xkcd_palette(colors)[1],linewidth = 2,linestyle=':')
#     plt.axvline(x=47,color = sns.xkcd_palette(colors)[1],linewidth = 2,linestyle=':')
#     plt.axvline(x=40,color = sns.xkcd_palette(colors)[1],linewidth = 2,linestyle=':')
#     plt.legend(loc=2,frameon=False)
#     #
#     #
#     name = "model_fit.pdf"
#     plt.savefig(pathOutput + name,bbox_inches='tight',format= "pdf",dpi=600)
#     plt.close()


# >>> (df.prices_actual - df.prices_sim_Pt).describe()
# count    573353.000000
# mean         -6.351556
# std          58.222029
# min        -503.780712
# 25%         -37.670529
# 50%          -8.412408
# 75%          15.814168
# max         605.145977
# dtype: float64

>>> (df.prices_actual - df.prices_sim_Pt).describe()
count    566982.000000
mean         -7.500791
std          58.225724
min        -679.124394
25%         -37.249328
50%          -9.651583
75%          12.639062
max         709.878169
dtype: float64
>>> 





df["dif"] = df.prices_actual - df.prices_sim_Pt
df.groupby("tdate").dif.mean()
tdate
0    -11.359852
1    -11.391304
2    -11.476550
3    -10.920658
4    -10.019190
5     -9.954452
6     -9.545033
7     -8.686065
8     -8.454609
9     -8.360721
10    -8.211906
11    -7.752776
12    -7.223271
13    -7.408378
14    -6.755814
15    -6.534964
16    -6.769771
17    -6.492318
18    -6.043718
19    -5.950478
20    -6.250955
21    -6.131704
22    -5.576035
23    -5.991496
24    -5.928584
25    -6.018717
26    -6.092949
27    -6.023106
28    -6.270080
29    -6.233817
30    -6.330683
31    -9.747614
32   -10.087328
33   -10.493982
34   -11.537463
35   -11.817602
36   -12.804961
37   -13.222790
38   -14.216841
39   -15.652330
40   -22.458900
41    -1.024763
42    -4.044898
43    -8.966239
44   -11.382152
45   -28.561025
46   -30.528393
47   -27.423048
48     0.564519
49     0.061509
50    -0.685787
51    -2.472344
52   -26.182051
53   -24.569172
54   -15.287936
55    17.415952
56    18.468638
57    22.012519
58    20.673944
59    29.003753
Name: dif, dtype: float64
