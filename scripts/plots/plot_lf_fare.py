"""
    This script creates the average load factor and price plot in
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
import matplotlib.pyplot as plt
import seaborn as sns
import plot_setup

# paths to read/write data
INPUT = "../../data"
OUTPUT = "../../output"

# figure additional setup
AX_RANGE = range(0, 61)

# -------------------------------------------------------------------------------
# READ DATA
# -------------------------------------------------------------------------------

df = pd.read_parquet(f"{INPUT}/efdata_clean.parquet"})
df_n = pd.read_parquet(f"{INPUT}/asdata_clean.parquet")

 # adjust to look at coach only
df_n["lf"] = (df_n["capY"]- df_n["sY"]) / df_n["capY"]

df = df[["origin", "dest", "flightNum" ,"tdate", "ddate", "fare", "seats", "lf"]]
df_n["seats"] = df_n[sY] # adjust to look at coach only
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
# compute mean by day before departure
loadfactor = 100 * df.groupby(["ttdate"])["lf"].mean()
meanfare = df.groupby(["ttdate"])["fare"].mean()

# -------------------------------------------------------------------------------
# CREATE PLOT
# -------------------------------------------------------------------------------
fig , ax1 = plt.subplots(figsize = FIG_SIZE)
ax2 = ax1.twinx()
ax1.plot(
    AX_RANGE,
    meanfare,
    label = "Mean Fare",
    color = PALETTE[4],
    linewidth = LINE_WIDTH,
    linestyle = "--")
ax2.plot(
    AX_RANGE,
    loadfactor,
    label = "Mean Load Factor",
    color = PALETTE[0],
    linewidth = LINE_WIDTH,
    linestyle = "-")

# set ylim for fare axis
ax1.set_ylim(np.min(meanfare) - 25, np.max(meanfare) + 25)
ax2.set_ylim(30,100) # adjust lf ylim
ax1.set_xlabel("Booking Horizon", **CSFONT)
ax1.set_ylabel("Mean Fare", color = PALETTE[4], **CSFONT)
ax2.set_ylabel("Mean Load Factor", color = PALETTE[0], **CSFONT)
plt.setp(
    ax1.legend(loc = 2, frameon = False).texts,
    family = FONT,
    fontsize = FONT_SIZE - 2
)
plt.setp(
    ax2.legend(loc = 1,frameon = False).texts,
    family = FONT,
    fontsize = FONT_SIZE - 2)
for x in [53, 46, 39]:
    plt.axvline(
        x = x,
        color = PALETTE[2],
        linewidth = LINE_WIDTH - 1,
        linestyle = ":"
    )
plt.xticks(fontname = FONT, fontsize = FONT_SIZE)
ax1.tick_params(axis = "both", which = "major", labelsize = FONT_SIZE)
ax2.tick_params(axis = "both", which = "major", labelsize = FONT_SIZE)

plt.savefig(
    f"{OUTPUT}/lf_fare_overT.pdf",
    bbox_inches = "tight",
    format = FIG_FORMAT,
    dpi = DPI)
plt.close()
