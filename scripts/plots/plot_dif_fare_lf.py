"""
    This script plots the difference in fare and difference in load factor
    for when a carrier operates two nonstop flights a day in
    "The Welfare Effects of Dynamic Pricing:Evidence from Airline Markets"
--------------------------------------------------------------------------------
change log:
    v0.0.1  Mon 14 Jun 2021
-------------------------------------------------------------------------------
notes:
    
--------------------------------------------------------------------------------
contributors:
    Kevin:
    name:   Kevin Williams
    email:  kevin.williams@yale.edu
--------------------------------------------------------------------------------
Copyright 2021 Yale University
"""

# -------------------------------------------------------------------------------
# IMPORT REQUIRED PACKAGES
# -------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as sm
import plot_setup


# paths to read/write data
INPUT = "../../data"
OUTPUT = "../../output"

# -------------------------------------------------------------------------------
# OPEN THE DATA / CLEAN OBS
# -------------------------------------------------------------------------------

df = pd.read_parquet(f"{INPUT}/efdata_clean.parquet")
df_n = pd.read_parquet(f"{INPUT}/asdata_clean.parquet")

# adjust to look at coach only
df_n["lf"] = (df_n ["capY"]- df_n ["sY"]) / df_n ["capY"]

df = df[[
    "origin", "dest", "flightNum", "tdate", "ddate",
    "fare", "seats", "lf", "acode"
]]
df_n["seats"] = df_n ["sY"] # adjust to look at coach only
df_n["acode"] = "AS"
df = df.append(
    df_n[[
        "origin", "dest", "flightNum", "tdate", "ddate",
        "fare", "seats", "lf", "acode"
    ]]
)

dfR  = pd.read_csv(f"{INPUT}/airline_routes.csv", sep = "\t", header = None)
dfR[0] = dfR[0].str.strip()
dfR[1] = dfR[1].str.strip()
dfR[0] = dfR[0].astype("str")
dfR[1] = dfR[1].astype("str")
ddfR.rename(
    columns = {
        0: "origin",
        1: "dest",
        2: "year",
        3: "comp"
    },
    inplace = True
)
df = df.merge(dfR, on = ["origin", "dest"], how = "left")

df["numJ"] = df.groupby(
    ["origin", "dest", "tdate", "ddate", "acode"]
).flightNum.transform("nunique")
df = df.loc[df["numJ"] == 2]

df["f"] = df.groupby(
    ["origin", "dest", "tdate", "ddate", "acode"]
).flightNum.transform("first")
df["f"] = (df["f"] == df["flightNum"]).astype("int")

df1 = df.pivot(
    index = ["origin", "dest", "tdate", "ddate", "acode"],
    columns = "f",
    values = ["lf", "fare"]
)
df1 = df1.reset_index()
df1["difF"] = df1["fare"][0] - df1["fare"][1]
df1["difLF"] = 100 * (df1["lf"][0] - df1["lf"][1])

df1 = df1.loc[df1.difF.notnull()].reset_index(drop = True)
df1.loc[df1.difLF < 0, "difF"] = -df1.loc[df1.difLF < 0].difF.values
df1.loc[df1.difLF < 0, "difLF"] = -df1.loc[df1.difLF < 0].difLF.values

# -------------------------------------------------------------------------------
# RUN THE POLY REGRESSION AND PLOT THE RESULTS
# -------------------------------------------------------------------------------

result = sm.ols(
    formula = "difF ~ difLF + I(difLF**2) + I(difLF**3) + I(difLF**4)",
    data=df1
).fit()
betahat = result.params.values
predict = np.ones((30,5))
predict[:, 1] = np.arange(30)
predict[:, 2] = predict[:, 1] * np.arange(30)
predict[:, 3] = predict[:, 2] * np.arange(30)
predict[:, 4] = predict[:, 3] * np.arange(30)

xb = predict.dot(betahat)

fig = plt.figure(figsize = FIG_SIZE)
plt.plot(
    range(0,30),
    xb,
    label = "Percent Difference in Fare",
    color = PALETTE[3],
    linewidth = LINE_WIDTH,
    linestyle = "-"
)

plt.setp(plt.legend().texts, family = FONT, fontsize = FONT_SIZE - 2)
plt.xlabel("Difference in Load Factor", **CSFONT)
plt.ylabel("Percent Difference in Fare", **CSFONT)
plt.yticks(fontname = FONT, fontsize = FONT_SIZE) 
plt.xticks(fontname = FONT, fontsize = FONT_SIZE) 

plt.savefig(
    f"{OUTPUT}/dif_fare_dif_lf.pdf",
    bbox_inches = "tight",
    format = FIG_FORMAT,
    dpi = 600
)
plt.close()
