"""
    This script plots average fares across fare classes in
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

# -------------------------------------------------------------------------------
# DEFINE READ DATA FUNCTIONS
# -------------------------------------------------------------------------------

# first class is only observed in the AS data, so load asdata.
df_n = pd.read_parquet(f"{INPUT}/asdata_clean.parquet")

# replace time until departure variable from -60,0 to 0,60
df_n["ttdate"] = -df_n["tdate"] + 60
df_n = df_n.loc[df_n.nonstop == 1]

# compute mean by day before departure
mainFare = df_n.groupby(["ttdate"])["mainFare"].mean()
saverFare = df_n.groupby(["ttdate"])["saverFare"].mean()
refundYFare = df_n.groupby(["ttdate"])["refundYFare"].mean()
firstFare = df_n.groupby(["ttdate"])["firstFare"].mean()
refundFFare = df_n.groupby(["ttdate"])["refundFFare"].mean()


df_n["FF"] = df_n[["firstFare", "refundFFare"]].min(axis = 1)
FF = df_n.groupby(["ttdate"])["FF"].mean()

# -------------------------------------------------------------------------------
# CREATE THE PLOT
# -------------------------------------------------------------------------------
fig = plt.figure(figsize = FIG_SIZE)
plt.plot(
    saverFare,
    label = "Saver Economy",
    color = PALETTE[0],
    linewidth = LINE_WIDTH,
    linestyle = ":"
)
plt.plot(
    mainFare,
    label = "Economy",
    color = PALETTE[1],
    linewidth = LINE_WIDTH,
    linestyle = "-"
)
plt.plot(
    refundYFare,
    label = "Unrestricted Economy",
    color = PALETTE[2],
    linewidth = LINE_WIDTH,
    linestyle = "-."
)
plt.plot(
    firstFare,
    label = "First Class",
    color = PALETTE[3],
    linewidth = LINE_WIDTH,
    linestyle = "-")
plt.plot(
    refundFFare,
    label = "Unrestricted First Class",
    color = PALETTE[4],
    linewidth = LINE_WIDTH,
    linestyle = "-."
)
plt.setp(plt.legend().texts, family = FONT, fontsize = FONT_SIZE - 2)
plt.xlabel("Booking Horizon", **CSFONT)
plt.ylabel("Average Fare", **CSFONT)
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
    f"{OUTPUT}/fyfares.pdf",
    bbox_inches = "tight",
    format = FIG_FORMAT,
    dpi = DPI
)
plt.close()
