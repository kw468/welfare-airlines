"""
    This script plots fare availability over different fare classes
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
# OPEN THE DATA / CLEAN OBS
# -------------------------------------------------------------------------------

# only AS data has first class observations, so we only use those data
df = pd.read_parquet(f"{INPUT}/asdata_clean.parquet")
df = df.loc[df.nonstop == 1] # no connecting obs

# mark availability of each fare type
df.loc[df.mainFare.notnull(), "mainFare"] = 1
df.loc[df.saverFare.notnull(), "saverFare"] = 1
df.loc[df.refundYFare.notnull(), "refundYFare"] = 1

df.loc[df.mainFare.isnull(), "mainFare"] = 0
df.loc[df.saverFare.isnull(), "saverFare"] = 0
df.loc[df.refundYFare.isnull(), "refundYFare"] = 0

df.loc[(df.firstFare.notnull()) & (df.capF.notnull()), "firstFare"] = 1
df.loc[(df.refundFFare.notnull()) & (df.capF.notnull()), "refundFFare"] = 1

df.loc[(df.firstFare.isnull()) & (df.capF.notnull()), "firstFare"] = 0
df.loc[(df.refundFFare.isnull()) & (df.capF.notnull()), "refundFFare"] = 0

# replace time until departure variable from -60, 0 to 0, 60
df["ttdate"] = -df["tdate"] + 60

# this code obtains the fraction of flights that have first class
df1 = df.groupby(["flightNum", "ddate", "origin", "dest"]).sF.max().reset_index()
frac = df1.sF.notnull().sum() / \
    df[["flightNum", "ddate", "origin", "dest"]].drop_duplicates().shape[0]

# -------------------------------------------------------------------------------
# CREATE THE PLOT
# -------------------------------------------------------------------------------

fig = plt.figure(figsize = FIG_SIZE)
plt.plot(
    AX_RANGE,
    100 * df.groupby("ttdate")["saverFare"].mean(),
    label = "Saver Economy",
    color = PALETTE[0],
    linewidth = LINE_WIDTH,
    linestyle = ":"
)
plt.plot(
    AX_RANGE,
    100 * df.groupby("ttdate")["mainFare"].mean(),
    label = "Economy Class",
    color = PALETTE[1],
    linewidth = LINE_WIDTH,
    linestyle="-."
)
plt.plot(
    AX_RANGE,
    frac * 100 * df.groupby("ttdate")["firstFare"].mean(),
    label = "First Class",
    color = PALETTE[3],
    linewidth = LINE_WIDTH,
    linestyle = "-"
)
plt.setp(plt.legend().texts, family = FONT, fontsize = FONT_SIZE - 2)
plt.xlabel("Booking Horizon", **CSFONT)
plt.ylabel("Percent of Flights with Available Fares", **CSFONT)
for x in [53, 46, 39]:
    plt.axvline(
        x = x,
        color = PALETTE[2],
        linewidth = LINE_WIDTH - 1,
        linestyle = ":"
    )
plt.yticks(fontname = FONT, fontsize = FONT_SIZE)
plt.xticks(fontname = FONT, fontsize = FONT_SIZE)

plt.savefig(
    f"{OUTPUT}/fareavail.pdf",
    bbox_inches = "tight",
    format = FIG_FORMAT,
    dpi = DPI)
plt.close()
