"""
    This code solves estimates the model parameters in Williams (2021)
    Demand:
        * Discrete type random coefficients model
        * Consumers choose to buy or no
        * Consumers arrive according to a Poisson distribution
        * Hence, demand is Poisson.
        * Demand may be censored; random rationing is assumed
    Firm:
        * Firm knows the demand process and solves a dynamic logit model
        * The states are seats remaining and time left to sell
    LLN:
        * The log-likelihood is constructed based on demand transitions
          and the conditional choice probabilities
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

from knitro import *                     # This program requires Knitro from Artleys
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
from estimation_setup import * # import jax params and common functions

# --------------------------------------------------------------------------------
# Set program parameters
# --------------------------------------------------------------------------------

market = sys.argv[1] #"SEA_SUN"
speed = sys.argv[2] #"fast" / "exact"

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

# --------------------------------------------------------------------------------
# Set path for data and logs
# --------------------------------------------------------------------------------
INPUT = "../../data/estimation/"
OUTPUT = INPUT + market + "/robust_estim/"
LOG_PATH = "logs/" + market + "/robust_estim/"
for path in [LOG_PATH, OUTPUT]
    if not os.path.exists(path):
        os.makedirs(path)

# change path so log to the right place
os.chdir(LOG_PATH)

# define the probability that a low type wants to buy
# rewrite exp(a)/(1 + exp(a)) as 1/(1/exp(a) + 1) = 1/(1 + exp(-a))
@jit
def purchInL(beta, bL, gamma): # p is dim(P), gam is dim(T), beta is dim FE
    return (1 - gamma)[:, None, None] * 1 / \
        (1 + jnp.exp(-beta[None, None, :] - bL * prices[None, :, None]))
    # returned object is T, P dimension

# define the probability that a high type wants to buy
@jit
def purchInB(beta, bB, gamma): # p is dim(P), gam is dim(T), beta is dim FE
    return (gamma)[:, None, None] * 1 / \
        (1 + jnp.exp(-beta[None, None, :] - bB * prices[None, :, None]))
    # returned object is T,P dimension

# define, this returns a T X P matrix of probability of purchase across both consumer types
@jit
def purchIn(beta, bB, bL, gamma):
    return purchInL(beta, bL, gamma) + purchInB(beta, bB, gamma)

# define the log probability of that demand is equal to q given
# the parameters of the model
@jit
def log_demandQ(beta, bL, bB, gamma, mu, q):
    return q * jnp.log(mu[:, None, :]) + q * \
        jnp.log(purchIn(beta, bB, bL, gamma)) - \
        (mu[:, None, :] * purchIn(beta, bB, bL, gamma)) - gammaln(q + 1)

# define the probability of that demand is equal
# to q given the parameters of the model, this is just exp(log(p)),
# which is done for numerical stablility
@jit
def demandQ(beta, bL, bB, gamma, mu, q):
    # return (mu[:, None] * purchIn(aB, bB, aL, bL, prices, gamma)) ** q * \
    #     jnp.exp(-mu[:, None] * (purchIn(aB, bB, aL, bL, prices, gamma))) \
    #     / factQ[q]
    return jnp.exp(log_demandQ(beta, bL, bB, gamma, mu, q))

def allDemand(beta, bL, bB, gamma, mu):
    vlookup = vmap((
        lambda x: demandQ(beta, bL, bB, gamma, mu, x)
    ))(jnp.arange(0, qBar))
    f = jnp.zeros((qBar, qBar, len(gamma), len(prices), len(beta)))
    for q in range(1, qBar): # seats remaining
        f = f.at[q, 0:q, :, :, :].set(vlookup[0:q, :, :, :])
        f = f.at[q, q, :, :, :] \
            .set(jnp.maximum(1 - jnp.sum(f[q, 0:(q), :, :, :], axis = 0), 1e-100))
    f = jnp.where((f > 0) & (f < 1e-100), 1e-100, f)
    return f
    # returning dimension is seats remaining, seats sold, time, and prices

# define the CCP and EV for the dynamic firm problem
def dynEst(f, ER, gamma, sig, beta):
    np.seterr(all = "raise")
        # create storage for EV, V, CCP
    EV = jnp.zeros((T, qBar, numP, len(beta)))
    V = jnp.zeros((qBar, T, len(beta)))
    CCP = jnp.zeros((T, qBar, numP, len(beta)))
    for t in range(1, T+1):
        # work backwards in time. In the last period, we just get last period revenues
        if t == 1:
            # the softmax function can be rewritten,
            # so let"s use logsum(exp) = x* + log sum (exp (x-x*))
            grp = ER[:, -t, :, :] / (sig) * Pt[-t, 1:][None, :, None]
            grp = jnp.where(grp == 0, -jnp.inf, grp)
            V = V.at[:, -t, :].set(sig * logsumexp(grp , axis = 1) + EC * sig)
            V = V.at[0, -t, :].set(0)
            # CCP[-t,:,:] = Pt[-t,1:][None,:] * np.exp(ER[:,-t,:] / sig) / \
            #     np.sum(np.exp(ER[:, -t, :] * Pt[-t, 1:][None, :] / sig), axis = 1)[:, None]
            CCP = CCP.at[-t, :, :, :].set(grp - logsumexp(grp, axis = 1)[:, None, :])
        else:
            grp = (ER[:, -t, :, :] / sig + EV[-t+1, :, :, :] / sig) * \
                Pt[-t, 1:][None, :, None]
            grp = jnp.where(grp == 0, -jnp.inf, grp)
            V = V.at[:, -t, :].set(sig * logsumexp(grp, axis = 1) + EC * sig)
            V = V.at[0, -t, :].set(0)
        # now we need to define EV, which is int_c" V f(c"),
        # so we"ll use tril to reset array(x) as reversearray(x)
        # this allows us to calc that the Pr(Q = 1) aligns with Pr(c" = c - 1)
        r, c = jnp.tril_indices_from(f[:, :, 0, 0, 0])
        for b in range(len(beta)):
            # update expected value function, this is for not the last period
            if t != T:
                g = jnp.array(f[:, :, -t - 1, :, b])
                g = g.at[r, c, :].set(g[r, r - c, :])
                EV = EV.at[-t, :, :, b].set(
                    jnp.sum(g * V[:, -t, b][None, :, None], axis = 1) * Pt[-t, 1:]
                )
        if t != 1:
            XX = (ER[:, -t, :, :] + EV[-t + 1, :, :, :]) / \
                sig * Pt[-t, 1:][None, :, None]
            XX = jnp.where(XX == 0, -jnp.inf, XX)
            CCP = CCP.at[-t, :, :, :].set(XX - logsumexp(XX, axis = 1)[:, None, :])
    return CCP

# calculate gradient for scaling factor using central differences
# this avoids a numerical issue with calculating the gradient entry via AD
def gradientSig(VAR, data):
    # The first parameters are consumer preferences:
    beta = jnp.array(VAR[0:7])
    bL = jnp.minimum(VAR[7], VAR[8])
    bB = jnp.maximum(VAR[7], VAR[8])
    gamma = 1 / (
        jnp.exp(
            -VAR[9] - jnp.arange(0, 60) * VAR[10] - \
                (jnp.arange(0, 60) ** 2) * VAR[11]
        ) + 1
    )
    # equivalent to jnp.array([1/(1 + jnp.exp(-g[0] + -t*g[1] - t**2*g[2])) for t in range(0,60)])
    # range(int(min(Tdata)), int(max(Tdata)+1))])
    muT = jnp.array(
        [VAR[12]] * (T - 20) + [VAR[13]] * 7 + \
        [VAR[14]] * 7 + [VAR[15]] * 6
    )
    muD = jnp.append(jnp.array([1]), jnp.array(VAR[16:22]))
    mu = muT[:, None] * muD[None, :]
    sig = VAR[-1]
    # first FE
    f0 = allDemand_jit(beta, bL, bB, gamma, mu)
    ER0 = jnp.sum(
        f0 * jnp.array(range(qBar))[None, :, None, None, None] * \
            prices[None, None, None, :, None],
        axis = 1
    )
    CCP0 = dynEst_jit(f0, ER0, gamma, sig + 1e-4, beta)
    loss0 = jnp.sum(CCP0[data[:, 2], data[:, 0], data[:, 3], data[:, 4]])
    CCP1 = dynEst_jit(f0, ER0, gamma, sig - 1e-4, beta)
    loss1 = jnp.sum(CCP1[data[:, 2], data[:, 0], data[:, 3], data[:, 4]])
    return (loss0 - loss1) / (2 * 1e-4)

# calculate the log-like for the data set
def logLike(VAR, data):
    # The first parameters are consumer preferences:
    beta = jnp.array(VAR[0:7])
    bL = jnp.minimum(VAR[7], VAR[8])
    bB = jnp.maximum(VAR[7], VAR[8])
    gamma = 1 / (jnp.exp(
            -VAR[9] - jnp.arange(0, 60) * VAR[10] - \
                (jnp.arange(0, 60) ** 2) * VAR[11]
        ) + 1
    )
    # equivalent to jnp.array([1/(1 + jnp.exp(-g[0] + -t*g[1] - t**2*g[2])) for t in range(0,60)])
    # range(int(min(Tdata)),int(max(Tdata)+1))])
    muT = jnp.array(
        [VAR[12]] * (T - 20) + [VAR[13]] * 7 + [VAR[14]] * 7 + [VAR[15]] * 6
    )
    muD = jnp.append(jnp.array([1]), jnp.array(VAR[16:22]))
    mu = muT[:, None] * muD[None, :]
    sig = VAR[-1]
    ######################
    # define all sales possibilities
    f0 = allDemand_jit(beta, bL, bB, gamma, mu)
    # define expected revenues
    ER0 = jnp.sum(
        f0 * jnp.array(range(qBar))[None, :, None, None, None] * \
            prices[None, None, None, :, None],
        axis = 1
    )
    # define loss associated with transition probabilities
    loss0 = jnp.sum(
        jnp.log(f0[data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4]])
    )
    # solve the DP, calculate the CCP and EV
    CCP0 = dynEst_jit(f0, ER0, gamma, sig, beta)
    # define loss for CCP
    loss1 = jnp.sum(CCP0[data[:, 2], data[:, 0], data[:, 3], data[:, 4]])
    ## SECOND FE
    return loss0 + loss1

# setup the problem in knitro
def estimKnitro(VAR, data, speed):
    n = len(VAR)
    bndsLo = np.array(VAR) - .5 * np.abs(np.array(VAR))
    bndsUp = np.array(VAR) + .5 * np.abs(np.array(VAR))
    kc = KN_new()
    KN_add_vars(kc, n)
    KN_set_var_lobnds(kc, xLoBnds = bndsLo)
    KN_set_var_upbnds(kc, xUpBnds = bndsUp)
    KN_set_var_types(kc, xTypes = [KN_VARTYPE_CONTINUOUS] * n)
    KN_set_obj_goal(kc, KN_OBJGOAL_MAXIMIZE)
    # KN_ALG_ACT_SQP KN_ALG_BAR_DIRECT
    KN_set_int_param(kc, "algorithm", KN_ALG_ACT_SQP)
    KN_set_int_param(kc, "blasoption", KN_BLASOPTION_INTEL)
    KN_set_int_param(kc, "outlev", KN_OUTLEV_ITER_X)
    KN_set_int_param(kc, "hessopt", KN_HESSOPT_BFGS)
    # KN_GRADOPT_EXACT,KN_GRADOPT_FORWARD
    KN_set_int_param(kc, "gradopt", KN_GRADOPT_EXACT)
    KN_set_double_param(kc, KN_PARAM_FEASTOL, 1.0E-8)
    KN_set_double_param(kc, KN_PARAM_OPTTOL, 1.0E-8)
    KN_set_int_param(kc, "par_numthreads", NUM_THREADS)
    KN_set_int_param(kc, "blasoption", 1)
    KN_set_int_param(kc, "par_blasnumthreads", NUM_THREADS)
    KN_set_int_param(kc, "bar_maxcrossit", 3)
    KN_set_var_primal_init_values(kc, xInitVals = VAR)
    KN_set_int_param(kc, KN_PARAM_MULTISTART, KN_MULTISTART_YES)
    KN_set_int_param(kc, KN_PARAM_MSMAXSOLVES, 42)
    KN_set_int_param(kc, "ms_outsub", KN_MS_OUTSUB_YES)
    KN_set_int_param(kc, "outmode", KN_OUTMODE_BOTH )
    # add objective function
    def callbackEvalF(kc, cb, evalRequest, evalResult, userParams):
        if evalRequest.type != KN_RC_EVALFC:
            print("*** callbackEvalG incorrectly called with eval type %d" % evalRequest.type)
            return -1
        x = evalRequest.x
        evalResult.obj = logLike_jit(x, data)
        return 0
    def callbackEvalG(kc, cb, evalRequest, evalResult, userParams):
        if evalRequest.type != KN_RC_EVALGA:
            print("*** callbackEvalG incorrectly called with eval type %d" % evalRequest.type)
            return -1
        x = evalRequest.x
        # Evaluate gradient of nonlinear objective
        Delta = np.array(gradF(x))
        if speed == "fast":
            Delta[-1] = gradSig(x)
        for i in range(len(Delta)):
            evalResult.objGrad[i] = float(Delta[i])
        return 0
    cb = KN_add_eval_callback(kc, evalObj = True, funcCallback = callbackEvalF)
    KN_set_cb_grad(kc, cb, objGradIndexVars = KN_DENSE, gradCallback = callbackEvalG)
    # solve
    KN_solve(kc)
    nSTatus, objSol, x, lambda_ = KN_get_solution(kc)
    KN_free(kc)
    return x



df_route = pd.read_csv(INPUT + market + ".csv", index_col = 0)

df_route_pt = pd.read_csv(INPUT + market + "_Pt.csv", header = None)

prices = jnp.array(np.genfromtxt(INPUT + market + "_prices.csv"))

Pt = df_route_pt.values

data = np.array(df_route.values)

qBar = int(np.max(df_route.seats)) + 1
T = len(np.unique(df_route.tdate))
numP = len(prices)
obs = len(df_route.tdate)
Pt = jnp.array(Pt)
EC = 0.5772156649

xInit = np.genfromtxt(INPUT + market + "_params.csv")

VAR = jnp.array(xInit)
allDemand_jit = jit(allDemand)
dynEst_jit = jit(dynEst)
logLike_jit = jit(logLike)


if speed !="fast":
    gradF = jit(jacfwd(lambda x: logLike_jit(x, data)))
elif speed == "fast":
    gradSig = jit((lambda x: gradientSig(x, data)))
    gradF = jit(grad(lambda x: logLike_jit(x, data)))

solution0 = estimKnitro(VAR,data,speed)

# back to scripts/estimation so save correctly
os.chdir("../../../")

pd.DataFrame(solution0).to_csv(
    OUTPUT + market + "_robust_params.csv",
    header = None,
    index = None
)
LLN = float(logLike_jit(solution0,data))
tbl = [LLN]
pd.DataFrame(np.array(tbl)).to_csv(
    OUTPUT + market + "_robust_info.csv",
    header = None,
    index = None
)

