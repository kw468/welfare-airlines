"""
    This script investigates the price response of nonstop bookings
    on connecting fares in
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
from plot_setup import * # improt constants

# paths to read/write data
INPUT = "../../data"
OUTPUT = "../../output"

# figure additional setup
AX_RANGE = range(0, 60)

# -------------------------------------------------------------------------------
# READ DATA
# -------------------------------------------------------------------------------

df1 = pd.read_parquet(f"{INPUT}/asdata_cleanCon.parquet")
df1 = df1.groupby(["origin", "dest", "sdate", "ddate"]).fare.mean()
df1 = df1.reset_index(drop = False)

df = pd.read_parquet(f"{INPUT}/asdata_clean.parquet")
df = df.drop(columns = "fare")


df = df.merge(df1, on = ["origin", "dest", "sdate", "ddate"], how = "inner")

# replace time until departure variable from -60,0 to 0,60
df["ttdate"] = -df["tdate"] + 60

cols = ["origin", "dest", "ddate", "flightNum", "tdate"]
df = df.sort_values(cols, ascending = False).reset_index(drop = True)

df["seats"] = df["sY"] + df["sF"]
df.loc[df.capY == 76, "seats"] = df["sY"]

cols = ["origin", "dest", "flightNum", "ddate"]
df["difS"] = df.groupby(cols)["seats"].shift(-1) - df["seats"]
df["difP"] = df.groupby(cols)["fare"].shift(-1) - df["fare"]


df = df.loc[df["difS"].notnull()]
df["seatC"] = 0
df.loc[df["difS"] < 0, "seatC"] = 1

df2 = df.copy()
df2 = df2.groupby(["ttdate", "seatC"])["difP"].mean().reset_index()
df2 = pd.pivot_table(
    df2, values = ["difP"], index = ["ttdate"], columns = ["seatC"]
)

# -------------------------------------------------------------------------------
# CREATE THE PLOT
# -------------------------------------------------------------------------------

def plotFareResponse(df2):
    fig = plt.figure(figsize = FIG_SIZE)
    plt.plot(
        AX_RANGE,
        df2["difP"][0].values,
        label = "No Sales",
        color = PALETTE[0],
        linewidth = LINE_WIDTH,
        linestyle = "--")
    plt.plot(
        AX_RANGE,
        df2["difP"][1].values,
        label = "Positive Sales",
        color = PALETTE[4],
        linewidth = LINE_WIDTH,
        linestyle = "-"
    )
    plt.setp(plt.legend().texts, family = FONT, fontsize = FONT_SIZE - 2)
    plt.xlabel("Booking Horizon", **CSFONT)
    plt.ylabel("Fare Response ($)", **CSFONT)
    # adjust vlines by 1 because of dif() in data creation
    for x in [40, 47, 54]:
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
        f"{OUTPUT}/fareresponse_1stop.pdf",
        bbox_inches = "tight",
        format = FIG_FORMAT,
        dpi = DPI
    )
    plt.clf()

plotFareResponse(df2)
