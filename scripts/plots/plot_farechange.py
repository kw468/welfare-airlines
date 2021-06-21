"""
 This script plots the frequency and magnitude of fare changes in
 "The Welfare Effects of Dynamic Pricing:Evidence from Airline Markets"
 Both monopoly and duopoly markets are considered
--------------------------------------------------------------------------------
change log:
 v0.0.1 Mon 14 Jun 2021
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
from plot_setup import * # improt constants

# paths to read/write data
INPUT = "../../data"
OUTPUT = "../../output"

# figure additional setup
AX_RANGE = range(0, 60)

# -------------------------------------------------------------------------------
# OPEN THE DATA / CLEAN OBS
# -------------------------------------------------------------------------------

df 	= pd.read_parquet(f"{INPUT}/efdata_clean.parquet")
df_n = pd.read_parquet(f"{INPUT}/asdata_clean.parquet")

df_n["lf"] = (df_n["capY"]- df_n["sY"]) / df_n["capY"] # adjust to look at coach only

df = df[["origin", "dest", "flightNum" ,"tdate", "ddate", "fare", "seats", "lf"]]
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
df = df.sort_values(cols, ascending = False).reset_index(drop = True)
cols = ["origin", "dest", "flightNum", "ddate"]
df["difS"] 	= df.groupby(cols).seats.shift(-1) - df.seats
df["difP"] 	= df.groupby(cols).fare.shift(-1) - df.fare

df = df.loc[df.difP.notnull()]
df["fareC"] = 0
df.loc[df.difP > 0, "fareC"] = 1
df.loc[df.difP < 0, "fareC"] = 2

# -------------------------------------------------------------------------------
# CREATE THE PLOT
# -------------------------------------------------------------------------------
def plotFareChange(df, comp):
    df1 = df.loc[df["comp"] == comp]
    df1 = df1.groupby(["ttdate", "fareC"])["fare"].count().reset_index()
    df1 = pd.pivot_table(
        df1, values = "fare", index = ["ttdate"], columns = ["fareC"]
    )
    df1["total"] = df1[0] + df1[1] + df1[2]
    df1[0] = df1[0] / df1["total"]
    df1[1] = df1[1] / df1["total"]
    df1[2] = df1[2] / df1["total"]

    df2 = df.loc[df["comp"] == comp].copy().groupby(["ttdate", "fareC"])["difP"].mean().reset_index()
    df2 = pd.pivot_table(
        df2, values = "difP", index = ["ttdate"], columns = ["fareC"])
    # NOW PLOT THE RESULTS
    if comp:
        name = "lffarechange_comp"
    elif comp == 0:
        name = "lffarechange"
    fig = plt.figure(figsize = FIG_SIZE)
    plt.plot(
        AX_RANGE,
        100 * df1[2].values,
        label = "Fare Declines",
        color = PALETTE[0],
        linewidth = LINE_WIDTH,
        linestyle = "--"
    )
    plt.plot(
        AX_RANGE,
        100 * df1[1],
        label = "Fare Increases",
        color = PALETTE[4],
        linewidth = LINE_WIDTH,
        linestyle = "-"
    )
    plt.setp(plt.legend().texts, family = FONT, fontsize = FONT_SIZE)
    plt.xlabel("Booking Horizon", **CSFONT)
    plt.ylabel("Frequency (%)", **CSFONT)
    for x in [54, 47, 40]:
        plt.axvline(
            x = x,
            color = PALETTE[2],
            linewidth = LINE_WIDTH - 1,
            linestyle = ":"
        )
    plt.yticks(fontname = FONT, fontsize = FONT_SIZE)
    plt.xticks(fontname = FONT, fontsize = FONT_SIZE)
    plt.savefig(
        f"{OUTPUT}/{name}0.pdf",
        bbox_inches = "tight",
        format = FIG_FORMAT,
        dpi = DPI
    )
    plt.close()
    #
    fig = plt.figure(figsize = FIG_SIZE)
    plt.plot(
        AX_RANGE,
        df2[2],
        label = "Fare Declines",
        color = PALETTE[0],
        linewidth = LINE_WIDTH,
        linestyle = "--"
    )
    plt.plot(
        AX_RANGE,
        df2[1],
        label = "Fare Increases",
        color = PALETTE[4],
        linewidth = LINE_WIDTH,
        linestyle = "-"
    )
    plt.setp(plt.legend().texts, family = FONT, fontsize = FONT_SIZE - 2)
    plt.xlabel("Booking Horizon", **CSFONT)
    plt.ylabel("Magnitude ($)", **CSFONT)
    for x in [54, 47, 40]:
        plt.axvline(
            x = x,
            color = PALETTE[2],
            linewidth = LINE_WIDTH - 1,
            linestyle = ":"
        )
    plt.yticks(fontname = FONT, fontsize = FONT_SIZE)
    plt.xticks(fontname = FONT, fontsize = FONT_SIZE)
    plt.savefig(
        f"{OUTPUT}/{name}1.pdf",
        bbox_inches = "tight",
        format = FIG_FORMAT,
        dpi = DPI
    )
    plt.close()

#run the program
plotFareChange(df,1)
plotFareChange(df,0)
