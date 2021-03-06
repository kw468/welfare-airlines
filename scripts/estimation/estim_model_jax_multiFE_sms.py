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
    Thurs 21 Jun 2021
-------------------------------------------------------------------------------
notes:  This program uses two specialized softwares; knitro and gpu-enabled jax.
        A licenses and the callable python api of knitro can be obtained Artleys.
        The programs expects the correct cuda drivers installed as functions are
        compiled and ran on a local nividia GPUs using the jax callable api.

--------------------------------------------------------------------------------
contributors:
    Kevin:
        name:       Kevin Williams
        email:      kevin.williams@yale.edu
--------------------------------------------------------------------------------
Copyright 2021 Yale University
"""

from knitro import * # This program requires Knitro from Artleys
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

market = sys.argv[1] # "SEA_SUN"
speed = sys.argv[2]  # "fast" / "exact"

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
INPUT = "../../data"
OUTPUT  = INPUT + "estimation/" + market + "/"
LOG_PATH = "logs/" + market + "/"
for path in [LOG_PATH, OUTPUT]
    if not os.path.exists(path):
        os.makedirs(path)

# change path so log to the right place
os.chdir(LOG_PATH)

# setup the problem in knitro
def estimKnitro(VAR, data, speed):
    n = len(VAR)
    bndsLo = np.array([
        -10, -10, -10, -10, -10, -10, -10, -10, -10,
        -250, -10, -.06, .1, .1, .1, .1,
        .01, .01, .01, .01, .01, .01, .02
    ])
    bndsUp = np.array([
        15, 15, 15, 15, 15, 15, 15, 0, 0, 40, 10, .15,
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 2
    ])
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
    KN_set_int_param(kc, KN_PARAM_MSMAXSOLVES, 200)
    KN_set_int_param(kc, "ms_outsub", KN_MS_OUTSUB_YES)
    KN_set_int_param(kc, "outmode", KN_OUTMODE_BOTH)
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

## so that AB, BA both become A-B
def determine_OD_Pair(D, O):
    return "_".join(sorted([D,O]))

# load the AS data and create ODs
df = pd.read_parquet(f"{INPUT}/asdata_clean.parquet")
df["route"] = np.vectorize(determine_OD_Pair)(df["origin"], df["dest"])


cols = ["origin", "dest", "ddate", "flightNum"]
df["ones"] = 1
df["numObs"] = df.groupby(cols)["ones"].transform("sum")

df["dd_dow"] = pd.to_datetime(df.ddate).dt.dayofweek
df = df.loc[df.numObs >= 59]


# sort data to the deadline, prep for dif in seat maps
df["ttdate" ] = -df["tdate"] + 60
cols = ["origin", "dest", "ddate", "flightNum", "tdate"]
df["ddate"] = df["ddate"].astype("category").cat.codes
df = df.sort_values(cols, ascending = False).reset_index(drop = True)


df["lf"] = (df.capY + df.capF - df.sY - df.sF) / (df.capY + df.capF)
df.loc[df.capY == 76, "lf"] = (df.capY- df.sY)/df.capY
df["occupied"] = df.capY + df.capF - df.sY - df.sF
df.loc[df.capY == 76, "occupied"] = (df.capY - df.sY)
df["seats"] = df.sY + df.sF
df.loc[df.capY == 76, "seats"] = df.sY

# create dif in seat maps and dif in fares
cols = ["origin", "dest", "flightNum", "ddate"]
df["difS"] = df.groupby(cols).seats.shift(-1) - df.seats
df["difP"] = df.groupby(cols).fare.shift(-1) - df.fare



df_route = df.loc[df.route == market].reset_index(drop=True)

df_route = df_route.loc[df_route.difS.notnull()]


# this code is no longer run. When activated, it treats last-minute sales as early bookings
# the idea is that last minute seat map changes might be restrictive bookings where consumers
# are assigned seats at check in.
#flag                    = False
# if flag == True:
#     bs1    = df_route.loc[df_route.tdate == 1][["flightNum", "ddate", "difS"]]
#     bs2    = df_route.loc[df_route.tdate != 1]
#     bs1    . rename(columns = {"difS" : "adjust"}, inplace = True)
#     bs2    = bs2.merge(bs1, on = ["flightNum", "ddate"], how = "left")
#     bs2["seats"] = np.maximum(bs2.seats + bs2.adjust,0)
#     bs2    = bs2.loc[bs2.difP.notnull()]
#     bs2    = bs2.loc[bs2.difS.notnull()]
#     bs2    = bs2.loc[bs2.seats > 0]
#     bs2    = bs2.loc[bs2.seats > -bs2.difS]
#     df_route = bs2.copy()


df_route.loc[df_route.difS > 0 ,"difS"] = 0 # set any positive increases in RC to 0
df_route["difS"] = df_route.difS.abs() # make sales positive instead of negative

df_route = df_route.loc[df_route.seats > 0]  # exclude sold out obs
df_route["tdate"] = df_route["tdate"].max() - df_route.tdate

# Next, winsorize the data to remove entries in which a large number of seats disappear
# This could happen when:
#   * seat maps get smaller
#   * seat map errors
#   * measurement error in processing data
#   * Delta market has more errors which influences log-like, constrain data more.
if market == ("CHS_SEA") or market == ("OKC_SEA") or market == ("OMA_SEA")  :
    df_route = df_route.loc[df_route.difS < df_route.difS.quantile(.985)]
else:
    df_route = df_route.loc[df_route.difS < df_route.difS.quantile(.995)]

numFlights = df_route[["flightNum", "ddate"]].drop_duplicates().shape[0]
numDDates = df_route[["ddate"]].drop_duplicates().shape[0]
numObs = df_route.shape[0]
df_route = df_route[["fare", "tdate", "seats", "difS", "dd_dow"]]


# Next, create the fare menus by clustering the fares and then bfill() observing a fare
# This preserves the use of AP restrictions with clustered fares.
# K-means threshold set at 99%.
success = False
it = 2
while success == False:
    k = it
    kmeans = KMeans(n_clusters = k, random_state = 0) \
        .fit(df_route.fare.values.reshape(-1, 1))
    idx = np.argsort(kmeans.cluster_centers_.sum(axis = 1))
    lut = np.zeros_like(idx)
    lut[idx] = np.arange(k)
    df_route["fareI"] = lut[kmeans.labels_]
    df_route["fareC"] = np.sort(kmeans.cluster_centers_[:, 0])[df_route["fareI"]]
    cc = (np.corrcoef(df_route.fare, df_route.fareC)[0, 1]) ** 2
    print(cc)
    it += 1
    if cc >= .99:
        success = True
        it -= 1


prices = jnp.sort(df_route.fareC.unique())
prices = prices / 100
Pt = np.zeros((df_route.tdate.nunique(), df_route.fareC.nunique() + 1))
Pt[:, 0] = range(df_route.tdate.nunique())
for t in df_route.tdate.unique():
    pp = list(df_route.loc[df_route.tdate == t].fareI.unique())
    for p in pp:
        Pt[t, p + 1] = 1

# This block of code creates the core estim data as well as key data summaries that enter LLN
df_route = df_route[["seats", "difS", "tdate", "fareI", "dd_dow"]].astype("int")
# Q,S,T,P
data = np.array(df_route.values)
qBar = int(np.max(df_route.seats)) + 1
T = len(np.unique(df_route.tdate))
numP = len(prices)
obs = len(df_route.tdate)

# This block executes the bfill() for fares
first = (Pt != 0).argmax(axis = 0)
last = np.flip(T - (np.flip(Pt) != 0).argmax(axis = 0))
for p in range(numP):
    f = first[p + 1]
    l = last[p + 1]
    Pt[f:l, p + 1] = 1

# define Euler"s constant and an initial starting value for estim
EC = 0.5772156649
xInit = [
    2.49999999, 2.49999999, 2.49999999, 2.49999999, 2.49999999, \
    2.49999999, 2.49999999, -1.05185291, -0.72189149, -13.39650409, \
    0.27373386, 0.0, 1.91183252, 2.46138227, 1.82139054, 2.35728083, \
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.22463165
]

# send data to the GPU
Pt = jnp.array(Pt)
VAR = jnp.array(xInit)
allDemand_jit = jit(allDemand)
dynEst_jit = jit(dynEst)
logLike_jit = jit(logLike)

if speed !="fast":
    gradF = jit(jacfwd(lambda x: logLike_jit(x, data)))
elif speed == "fast":
    gradSig = jit((lambda x: gradientSig(x, data)))
    gradF = jit(grad(lambda x: logLike_jit(x, data)))

solution0 = estimKnitro(VAR, data, speed)

# back to scripts/estimation so save correctly
os.chdir("../../")

df_route.to_csv(OUTPUT + market + ".csv")
pd.DataFrame(solution0).to_csv(
    OUTPUT + market + "_params.csv",
    header = None,
    index = None
)
pd.DataFrame(np.array(Pt)).to_csv(
    OUTPUT + market + "_Pt.csv",
    header = None,
    index = None
)
pd.DataFrame(np.array(prices)).to_csv(
    OUTPUT + market + "_prices.csv",
    header = None,
    index = None
)

LLN = float(logLike_jit(solution0, data))
tbl = [LLN, numFlights, numDDates, numObs]
pd.DataFrame(np.array(tbl)).to_csv(
    OUTPUT + market + "_info.csv",
    header = None,
    index = None
)
