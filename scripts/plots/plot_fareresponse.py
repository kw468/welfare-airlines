"""
    This code processes the Expert Flyer and Yapta data in Williams (2021)
    Inputs:
        * Bucket availability and seat maps are pulled from the expertflyer.com API
        * Prices come from queries to the Yapta API.
    Operations:
        * The code constructs enplanement totals from xml files
        * Prices are gathered from queries on yapta.
    Output:
        * Nonstop data
        * Onestop data
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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plot_setup import * # improt constants

# -------------------------------------------------------------------------------
# DEFINE PATHS
# -------------------------------------------------------------------------------

# paths to read/write data
INPUT = "../../data"
OUTPUT = "../../output"

# figure additional setup
AX_RANGE = range(0, 60)

# -------------------------------------------------------------------------------
# OPEN THE DATA / CLEAN OBS
# -------------------------------------------------------------------------------

df = pd.read_parquet(f"{INPUT}/efdata_clean.parquet")
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
df["ddate"] = df["ddate"].astype("category").cat.codes

df = df.sort_values(cols, ascending = False).reset_index(drop = True)

cols = ["origin", "dest", "flightNum", "ddate"]
df["difS"] = df.groupby(cols)["seats"].shift(-1) - df["seats"]
df["difP"] = df.groupby(cols)["fare"].shift(-1) - df["fare"]

df = df.loc[df.difS.notnull()]
df["seatC"] = 0
df .loc[df.difS < 0, "seatC"] = 1

# -------------------------------------------------------------------------------
# CREATE THE PLOTS
# -------------------------------------------------------------------------------

# this function plots the fare response of nonstop bookings on nonstop fares
# both monopoly and duopoly markets are considered.
def plotFareResponse(df, comp):
    if comp:
        name = "fareresponse_comp.pdf"
    elif comp == 0:
        name = "fareresponse.pdf"
    df1 = df.loc[df["comp"] == comp]
    df1 = df1.groupby(["ttdate", "seatC"])["difP"].mean().reset_index()
    df1 = pd.pivot_table(
        df1, values = "difP", index = ["ttdate"], columns = ["seatC"]
    )
    fig = plt.figure(figsize = FIG_SIZE)
    plt.plot(
        AX_RANGE,
        df1[0],
        label = "No Sales",
        color = PALETTE[0],
        linewidth = LINE_WIDTH,
        linestyle = "--"
    )
    plt.plot(
        AX_RANGE,
        df1[1],
        label = "Positive Sales",
        color = PALETTE[4],
        linewidth = LINE_WIDTH,
        linestyle = "-"
    )
    plt.setp(plt.legend().texts, family = FONT, fontsize = FONT_SIZE - 2)
    plt.xlabel("Booking Horizon",**CSFONT)
    plt.ylabel("Fare Response ($)",**CSFONT)
    # adjust vline for dif() in data construction
    for x in [54, 47, 40]:
        plt.axvline(
            x = x,
            color = PALETTE[2],
            linewidth = LINE_WIDTH - 1,
            linestyle = ":"
        )
    plt.xticks(fontname = FONT, fontsize = FONT_SIZE)
    plt.yticks(fontname = FONT, fontsize = FONT_SIZE)
    plt.axhline(y = 0, color = PALETTE[-1], linewidth = LINE_WIDTH)
    plt.savefig(
        f"{OUTPUT}/{name}",
        bbox_inches = "tight",
        format = FIG_FORMAT,
        dpi = DPI)
    plt.clf()

# this function plots the fare response of first class bookings on first class fares.
def plotFareResponseFirst(df_n):
    # # replace time until departure variable from -60, 0 to 0, 60
    df_n["ttdate"] = -df_n["tdate"] + 60
    df_n["seats"] = df_n.sF
    cols = ["origin", "dest", "ddate", "flightNum", "tdate"]
    df["ddate"] = df_n["ddate"].astype("category").cat.codes
    df_n = df_n.sort_values(cols, ascending = False).reset_index(drop = True)

    cols = ["origin", "dest", "flightNum", "ddate"]
    df_n["difS"] = df_n.groupby(cols)["seats"].shift(-1) - df_n["seats"]
    df_n["difP"] = df_n.groupby(cols).firstFare.shift(-1) - df_n.firstFare
    df_n = df_n.loc[df_n.difS.notnull()]
    df_n["seatC"] 			= 0
    df_n.loc[df_n.difS < 0, "seatC"] = 1

    df2 = df_n.copy()
    df2 = df2.groupby(["ttdate", "seatC"])["difP"].mean().reset_index()
    df2 = pd.pivot_table(
        df2, values = ["difP"], index = ["ttdate"], columns = ["seatC"]
    )

    fig = plt.figure(figsize = FIG_SIZE)
    plt.plot(
        AX_RANGE,
        df2["difP"][0],
        label = "No Sales",
        color = PALETTE[0],
        linewidth = LINE_WIDTH,
        linestyle = "--"
    )
    plt.plot(
        AX_RANGE,
        df2["difP"][1],
        label = "Positive Sales",
        color = PALETTE[4],
        linewidth = LINE_WIDTH,
        linestyle = "-")
    plt.setp(plt.legend().texts, family = FONT, fontsize = FONT_SIZE - 2)
    plt.xlabel("Booking Horizon", **CSFONT)
    plt.ylabel("Fare Response ($)", **CSFONT)
    # adjust vline for dif() in data construction
    for x in [54, 47, 40]:
        plt.axvline(
            x = x,
            color = PALETTE[2],
            linewidth = LINE_WIDTH - 1,
            linestyle = ":"
        )
    plt.yticks(fontname = FONT, fontsize = FONT_SIZE)
    plt.xticks(fontname = FONT, fontsize = FONT_SIZE)
    plt.axhline(y = 0, color = PALETTE[-1], linewidth = LINE_WIDTH)
    plt.savefig(
        f"{OUTPUT}/fareresponse_firstC.pdf",
        bbox_inches = "tight",
        format = FIG_FORMAT,
        dpi = DPI
    )
    plt.close()

plotFareResponse(df, 0)
plotFareResponse(df, 1)
plotFareResponseFirst(df_n)
