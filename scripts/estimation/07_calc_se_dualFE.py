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


# --------------------------------------------------------------------------------
# Import Required Packages
# --------------------------------------------------------------------------------
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
numBS = 500
market = sys.argv[1] # "SEA_SUN"

def get_gpu_memory():
    _output_to_list = lambda x: x.decode("ascii").split("\n")[:-1]
    ACCEPTABLE_AVAILABLE_MEMORY = 1024
    COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values

gpuUsage = get_gpu_memory()

gpu = 0
if gpuUsage[0] < 10000:
    gpu = 1
    if gpuUsage[1] < 10000:
        gpu = 2

print("using gpu number: " + str(gpu))
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

# --------------------------------------------------------------------------------
# Set path for data and logs
# --------------------------------------------------------------------------------
INPUT = "../../data/estimation/"
OUTPUT = INPUT + market + "/se/"
if not os.path.exists(OUTPUT):
    os.makedirs(OUTPUT)

# setup the problem in knitro
def estimKnitro(VAR,data):
    n = len(VAR)
    bndsLo = np.array(VAR) - 2 * np.abs(np.array(VAR))
    bndsUp = np.array(VAR) + 2 * np.abs(np.array(VAR))
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
    KN_set_int_param(kc, "gradopt", KN_GRADOPT_FORWARD)
    KN_set_double_param(kc, KN_PARAM_FEASTOL, 1.0E-8)
    KN_set_double_param(kc, KN_PARAM_OPTTOL, 1.0E-8)
    KN_set_int_param(kc, "par_numthreads", NUM_THREADS)
    KN_set_int_param(kc, "blasoption", 1)
    KN_set_int_param(kc, "par_concurrent_evals", 1)
    KN_set_int_param(kc, "par_blasnumthreads", NUM_THREADS)
    KN_set_int_param(kc, "bar_maxcrossit", 3)
    KN_set_var_primal_init_values(kc, xInitVals = VAR)
    # KN_set_int_param(kc, KN_PARAM_MULTISTART, KN_MULTISTART_YES)
    # KN_set_int_param(kc, KN_PARAM_MSMAXSOLVES, 100)
    # KN_set_int_param(kc, "ms_outsub", KN_MS_OUTSUB_YES)
    KN_set_int_param(kc, "outmode", KN_OUTMODE_BOTH)
    # add objective function
    def callbackEvalF(kc, cb, evalRequest, evalResult, userParams):
        if evalRequest.type != KN_RC_EVALFC:
            print("*** callbackEvalG incorrectly called with eval type %d" % evalRequest.type)
            return -1
        x = evalRequest.x
        evalResult.obj = logLike_jit(x, data)
        return 0
    cb = KN_add_eval_callback(kc, evalObj = True, funcCallback = callbackEvalF)
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

xInit = np.genfromtxt(INPUT + "/robust_estim/" + market + "_robust_params.csv")

VAR = jnp.array(xInit)
allDemand_jit = jit(allDemand)
dynEst_jit = jit(dynEst)
logLike_jit = jit(logLike)

X = []
for it in range(numBS):
    i = np.random.choice(data.shape[0], data.shape[0])
    bs = data[i]
    solution0 = estimKnitro(VAR,bs)
    os.rename("knitro.log", "knitro_" + str(it) + ".log")
    if not np.array_equal(solution0, xInit):
        X.append(solution0)

X = np.array(X)
df = pd.DataFrame(X)
df.to_csv(
    OUTPUT + market + "_bs_samples.csv",
    header = None,
    index = None
)

SE = np.sqrt(1 / len(X) * ((X - X.mean(0)) ** 2).sum(0))
pd.DataFrame(SE).to_csv(OUTPUT + market + "_se.csv", header = None, index = None)

tstat = xInit/SE
pd.DataFrame(tstat).to_csv(OUTPUT + market + "_tstat.csv", header = None, index = None)

#srun -w c1 --pty /bin/bash
