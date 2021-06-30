"""
    This script report the results for stochastic limit counterfactual in
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

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import os
from estim_markets import *

# --------------------------------------------------------------------------------
# Set path for data and logs
# --------------------------------------------------------------------------------
INPUT = "../../data/"

os.chdir(INPUT)

X = []
for m in mkts:
    print(m)
    tmp = pd.read_parquet(
        INPUT + "estimation/" + m + "/" + \
            m + "stochLim_counterfactuals.parquet"
    )
    tmp = tmp.loc[tmp.initC != 0]
    tmp = tmp.groupby(["initC", "fl"]).salesD.sum().reset_index()
    tmp = tmp.groupby("initC").salesD.mean()
    pctchange = tmp.pct_change()
    thres = np.argmax((np.abs(pctchange) < .005) & (pctchange > 0))
    if thres == 0:
        thres = len(pctchange)
    data = pd.read_csv(INPUT + "estimation/" + m + "/" + m + ".csv")
    data = data.loc[data.tdate == 0]
    X.extend([(data.seats > thres).values])

np.sum([x.sum() for x in X]) / np.sum([len(x)for x in X])
