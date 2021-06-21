"""
    This script stores the common parameters used in creating figures for
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

import matplotlib.pyplot as plt
import seaborn as sns

FONT = "Liberation Serif"
FONT_SIZE = 20
CSFONT = {"fontname": FONT, "fontsize": FONT_SIZE}
PALETTE = ["#FF6700", "#FCB07E", "#6B717E", "#3A6EA5", "#004E98", "#070707"]
LINE_WIDTH = 3
sns.set(style = "white", color_codes = False)
plt.rcParams["font.family"] = FONT
FIG_SIZE = (1.5 * 6.4, 1.1 * 4.8)
DPI = 600
FIG_FORMAT = "PDF"
