"""
    Thi script matches monthly enplanements using my seat maps aggregated on
    the day of departure with actual monthly enplanements reported in the T100
    Segment tables with assumptions described in the paper.
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
from plot_setup import *

# paths to read/write data
INPUT = "../../data"
OUTPUT = "../../output"

# -------------------------------------------------------------------------------
# OPEN THE DATA / CLEAN OBS
# -------------------------------------------------------------------------------

df_n = pd.read_parquet(f"{INPUT}/asdata_clean.parquet")
df_n = df_n.loc[df_n.tdate == 0]
df_n["month"] = df_n.ddate.dt.month
df_n["occupied"] = df_n["capY"] - df_n["sY"] + \
    (df_n["capF"] - df_n["sF"]).fillna(0)
df_n["capacity"] = df_n["capY"] + (df_n["capF"]).fillna(0)
df_n["group"] = df_n.groupby(["flightNum", "ddate"]).ngroup()

# common columns that will be used throughout the code
main_cols = ["origin", "dest", "month"]

df_n = df_n.groupby(main_cols) \
    .agg({"occupied" : "sum", "capacity" : "sum", "group" : "nunique"})
df_n = df_n.reset_index(drop = False)

t100 = pd.read_csv(f"{INPUT}/2019_T100D_SEGMENT_US_CARRIER_ONLY.csv")
t100 = t100.loc[t100.CLASS == "F"]

t100.rename(
    columns = {"ORIGIN" : "origin", "DEST" : "dest", "MONTH" : "month"},
    inplace = True
)
t100 = t100.loc[(t100.CARRIER == "AS") | (t100.CARRIER == "QX")]
t100 = t100.groupby(main_cols)["SEATS", "PASSENGERS", "DEPARTURES_PERFORMED"].sum()
t100 = t100.reset_index(drop = False)

df_n = df_n.merge(t100, on = main_cols, how = "left")


### OPERATE ON 2012 DATA

df_m = pd.read_parquet(f"{INPUT}/efdata_clean.parquet")
df_m = df_m.loc[df_m.tdate == 0]
df_m["month"] = df_m.ddate.dt.month
df_m["group"] = df_m.groupby(["flightNum", "ddate"]).ngroup()

df_m = df_m.groupby(main_cols).agg(
    {"occupied" : "sum", "capacity" : "sum", "group" : "nunique"}
)
df_m = df_m.reset_index(drop = False)

t100 = pd.read_csv(f"{INPUT}/2012_T100D_SEGMENT_US_CARRIER_ONLY.csv")
t100 = t100.loc[t100.CLASS == "F"]
t100["fclass"] = 0
t100.loc[(t100.AIRCRAFT_TYPE == 614) | (t100.AIRCRAFT_TYPE == 634), "fclass"] = 16
t100.loc[(t100.AIRCRAFT_TYPE == 673) | (t100.AIRCRAFT_TYPE == 638), "fclass"] = 12
t100["fclass"] = t100.fclass*t100.DEPARTURES_PERFORMED
t100["PASSENGERS"] = t100["PASSENGERS"] - t100["fclass"]

t100 = t100.loc[t100.UNIQUE_CARRIER != "DL"]
t100 = t100.loc[t100.UNIQUE_CARRIER != "CP"]
t100 = t100.loc[t100.UNIQUE_CARRIER != "9E"]

t100.rename(
    columns = {"ORIGIN" : "origin", "DEST" : "dest", "MONTH" : "month"},
    inplace = True
)

t100 = t100.groupby(main_cols)["SEATS", "PASSENGERS", "DEPARTURES_PERFORMED"].sum()
t100 = t100.reset_index(drop = False)

df_m = df_m.merge(t100, on = main_cols, how = "left")

df = df_m.append(df_n)
df = df.reset_index(drop = True)
df["dep_dif"] = df.DEPARTURES_PERFORMED - df.group

df["avgPaxD"] = df.occupied / df.group
df["avgPaxT"] = df.PASSENGERS / df.DEPARTURES_PERFORMED
df["paxAdjust"] = df.occupied.copy()
df.loc[df.dep_dif > 0, "paxAdjust"] = df.paxAdjust + df.dep_dif * df.avgPaxT
df.loc[df.dep_dif < 0, "paxAdjust"] = df.paxAdjust + df.dep_dif * df.avgPaxD

# NOW PLOT THE RESULTS
fig = plt.figure(figsize = FIG_SIZE)
plt.scatter(df.PASSENGERS, df.paxAdjust, s = 40, color = PALETTE[4])
plt.plot(
    df.PASSENGERS,
    df.PASSENGERS,
    color = PALETTE[1],
    linewidth = LINE_WIDTH,
    linestyle = "-"
)
plt.xlabel("Estimated Enplanements using T100 Tables", **CSFONT)
plt.ylabel("Estimated Enplanements using Seat Maps", **CSFONT)
plt.yticks(fontname = FONT, fontsize = FONT_SIZE)
plt.xticks(fontname = FONT, fontsize = FONT_SIZE)
plt.savefig(
    f"{OUTPUT}/T100Error.pdf",
    bbox_inches = "tight",
    format = FIG_FORMAT,
    dpi = DPI
)
