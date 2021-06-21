"""
  This script plots the percentage fare in changes over time for each
  market separately in
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

import numpy as np
import csv
import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from plot_setup import * # improt constants

# paths to read/write data
INPUT = "../../data"
OUTPUT = "../../output"

# -------------------------------------------------------------------------------
# OPEN THE DATA / CLEAN OBS
# -------------------------------------------------------------------------------

df = pd.read_parquet(f"{INPUT}/efdata_clean.parquet")
df_n = pd.read_parquet(f"{INPUT}/asdata_clean.parquet")

df_n["lf"] = (df_n["capY"]- df_n["sY"]) / df_n["capY"] # adjust to look at coach only

df = df[[
    "origin", "dest", "flightNum" ,"tdate", "ddate", "fare", "seats", "lf"
]]
df_n["seats"] = df_n["sY"] # adjust to look at coach only
df = df.append(
    df_n[["origin", "dest", "flightNum", "tdate", "ddate", "fare", "seats", "lf"]]
)

dfR = pd.read_csv(f"{INPUT}/airline_routes.csv", sep = "\t", header = None)
dfR[0] = dfR[0].str.strip()
dfR[1] = dfR[1].str.strip()
dfR[0] = dfR[0].astype("str")
dfR[1] = dfR[1].astype("str")
dfR.rename(
    columns = {
        0: "origin",
        1: "dest",
        2: "year",
        3: "comp"
    },
    inplace = True
)

df = df.merge(dfR, on = ["origin", "dest"], how = "left")

# replace time until departure variable from -60, 0 to 0, 60
df["ttdate"] = -df["tdate"] + 60

cols = ["origin", "dest", "ddate", "flightNum", "tdate"]
df["ddate"] = df["ddate"].astype("category").cat.codes

df = df.sort_values(cols, ascending = False).reset_index(drop = True)

cols = ["origin", "dest", "flightNum", "ddate"]
df["difS"] = df.groupby(cols).seats.shift(-1) - df.seats
df["difP"] = df.groupby(cols).fare.shift(-1) - df.fare


df = df.loc[df.difS.notnull()]
df["seatC"] = 0
df .loc[df.difS < 0, "seatC"] = 1

df["route"] = df[["origin", "dest"]].min(axis = 1) + \
    df[["origin", "dest"]].max(axis = 1)

# -------------------------------------------------------------------------------
# CREATE PLOT GRID AND PLOT THE RESULTS
# -------------------------------------------------------------------------------

ncols = 4
nrows = int(np.ceil(df["route"].nunique() / ncols))
fig, axs = plt.subplots(
    nrows, ncols, figsize = (ncols * FIG_SIZE[0], nrows * FIG_SIZE[1])
)
fig.subplots_adjust(hspace = .4)
counter = 0
markets = sorted(list(df["route"].unique()))
for i in range(nrows - 1):
    for j in range(ncols):
        df1 = df.loc[df.route == markets[counter]]
        axs[i, j].plot(
            100 * df1.groupby("ttdate").fare.mean().pct_change(),
            color = PALETTE[4]
        )
        axs[i, j].set_ylim((-10, 50))
        axs[i, j].set_title(markets[counter], fontsize = FONT_SIZE, fontname = FONT)
        counter += 1

remain = len(markets) - counter
for j in range(remain):
    df1 = df.loc[df.route == markets[counter]]
    axs[nrows - 1, j].plot(
        100 * df1.groupby("ttdate").fare.mean().pct_change(),
        color = PALETTE[4]
    )
    axs[nrows - 1, j].set_ylim((-10, 50))
    axs[nrows - 1, j].set_title(markets[counter], fontsize = FONT_SIZE, fontname = FONT)
    counter += 1

for ax in axs.flat:
    ax.set(xlabel = "Booking Horizon", ylabel = "Fare % Change")
    ax.xaxis.get_label().set_fontsize(FONT_SIZE)
    ax.yaxis.get_label().set_fontsize(FONT_SIZE)
    ax.xaxis.get_label().set_fontname(FONT)
    ax.yaxis.get_label().set_fontname(FONT)

numDelete = ncols - remain
for j in range(numDelete):
    fig.delaxes(axs[nrows-1][ncols-j-1])

plt.yticks(fontname = FONT, fontsize = FONT_SIZE)
plt.xticks(fontname = FONT, fontsize = FONT_SIZE)

plt.savefig(
    f"{OUTPUT}/avg_fare_pct_change_allRoutes.pdf",
    bbox_inches = "tight",
    format = FIG_FORMAT,
    dpi = DPI
)
plt.close()
