"""
    This script calculates the measurement error in the UA seat map data in
    measurement error is at the flight, day before departure level.
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
import statsmodels.api as sm
from plot_setup import *
lowess = sm.nonparametric.lowess

# paths to read/write data
INPUT = "../../data"
OUTPUT = "../../output"

# -------------------------------------------------------------------------------
# DEFINE READ DATA FUNCTIONS
# -------------------------------------------------------------------------------

X = pd.read_csv(f"{INPUT}/united_data.csv")

# convert data to pd
# replace time until departure variable from -60, 0 to 0, 60
X["ttdate"] = X["ttdate"] + 60
# compute mean by day before departure
meanerror = 100 * X.groupby(["ttdate"])["pc2"].mean()
# compute poly fit of measurement error
coefs = np.polynomial.polynomial.polyfit(X["ttdate"], X["pc2"], 4)
ffit = np.polynomial.polynomial.Polynomial(coefs) # instead of np.poly1d

# NOW PLOT THE RESULTS
fig = plt.figure(figsize = FIG_SIZE)
meanerror.plot(
    style = ".",
    ms = 10,
    color = PALETTE[4],
    label = "Estimated Measurement Error"
)
plt.plot(range(61), 100 * ffit(range(61)), lw = LINE_WIDTH, color = PALETTE[1])
L = plt.legend()
plt.setp(L.texts, family=FONT, fontsize = FONT_SIZE - 2)
plt.xlabel("Booking Horizon", **CSFONT)
plt.ylabel("Percentage (0%-100%)", **CSFONT)
plt.yticks(fontname = FONT, fontsize = FONT_SIZE)
plt.xticks(fontname = FONT, fontsize = FONT_SIZE)
plt.savefig(
    f"{OUTPUT}/UASeatMapError.pdf",
    bbox_inches = "tight",
    format = FIG_FORMAT,
    dpi = DPI)
plt.close()
