"""
    This script estimates the dynamic substitution regressions that appear in
    "The Welfare Effects of Dynamic Pricing:Evidence from Airline Markets".
--------------------------------------------------------------------------------
change log:
    v0.0.1  Mon 14 Jun 2021
-------------------------------------------------------------------------------
notes:
    `stata` must be an executable.
    You might need to add a symbolic links for `stata` to work.
--------------------------------------------------------------------------------
contributors:
    Kevin:
        name:   Kevin Williams
        email:  kevin.williams@yale.edu
--------------------------------------------------------------------------------
Copyright 2021 Yale University
"""

import numpy as np
import pandas as pd
import subprocess
import os

INPUT = OUTPUT = "../../data"
# so that AB, BA both become A-B
def determine_OD_Pair(D, O):
    return "_".join(sorted([D, O]))

df = pd.read_parquet(f"{INPUT}/efdata_clean.parquet")
df_n = pd.read_parquet(f"{INPUT}/asdata_clean.parquet")

df_n["lf"] = (df_n.capY- df_n.sY) / df_n.capY # adjust to look at coach only

common_cols = ["origin", "dest", "flightNum", "tdate", "ddate", "fare", "seats", "lf"]
df = df[common_cols]
df_n["seats"] = df_n.sY # adjust to look at coach only
df = df.append(df_n[common_cols])

dfR = pd.read_csv(f"{INPUT}/airline_routes.csv", sep = "\t", header = None)
dfR[0] = dfR[0].str.strip()
dfR[1] = dfR[1].str.strip()
dfR[0] = dfR[0].astype("str")
dfR[1] = dfR[1].astype("str")
dfR .rename(
    columns = {0 : "origin", 1 : "dest", 2 : "year", 3 : "comp"},
    inplace = True
)

df = df.merge(dfR, on = ["origin", "dest"], how = "left")

# replace time until departure variable from -60,0 to 0,60
df["ttdate"] = -df["tdate"] + 60

cols = ["origin", "dest", "ddate", "flightNum", "tdate"]

df = df.sort_values(cols, ascending = False).reset_index(drop = True)

cols = ["origin", "dest", "flightNum", "ddate"]
df["difS"] = df.groupby(cols).seats.shift(-1) - df.seats
df["difP"] = df.groupby(cols).fare.shift(-1) - df.fare

df["route"] = np.vectorize(determine_OD_Pair)(df["origin"], df["dest"])

df["DDmonth"] = df.ddate.dt.month
df["DOWS"] = df.ddate - pd.to_timedelta(df["tdate"], unit = "d")
df["DOWS"] = df.DOWS.dt.dayofweek
df["DOWD"] = df.ddate.dt.dayofweek

df["APD3"] = 1 * (df.tdate == 3)
df["APD7"] = 1 * (df.tdate == 7)
df["APD14"] = 1 * (df.tdate == 14)
df["APD21"] = 1 * (df.tdate == 21)

df.to_csv(f"{OUTPUT}/subregs.csv")

dofile = f"{OUTPUT}/bunching_subRegressions.do"
cmd = ["stata", "-b", "do", dofile, "&"]
subprocess.call(cmd)

os.remove(f"{OUTPUT}/subregs.csv")
