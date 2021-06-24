"""
    This script creates figures, tables and numbers related to DB1B apppeared in
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

import pandas as pd
import numpy as np
import math
import glob
from multiprocessing import Pool
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.weightstats import ttest_ind

INPUT = "../../data"
OUTPUT = "../../output"

# figure settings
COLORS = ["windows blue", "greyish", "black", "red"]
DPI = 600
FIG_FORMAT = "PDF"
sns.set(style = "white", palette = sns.xkcd_palette(COLORS), color_codes = False)

def gatherAirports(f):
    qtr = pd.read_csv(f, usecols = ["origin"])
    return qtr.drop_duplicates()

def gatherPaxNums(f):
    qtr = pd.read_csv(f, usecols = ["origin", "paxx"])
    return qtr.groupby("origin").paxx.sum().reset_index(drop = False)

def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    All args must be of equal length.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * \
        np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km

def airportListNums(files):
    # run in parallel
    with Pool as p:
        results = p.map(gatherAirports, files)
        p.join()
    df = pd.concat(results)

    # run in parallel
    with Pool as p:
        results = p.map(gatherPaxNums, files)
        p.join()

    nums = pd.concat(results).groupby("origin").paxx.sum().reset_index(drop = False)

    db1b_airports = df[["origin"]].drop_duplicates()
    db1b_airports = db1b_airports.rename(columns = {"origin" : "airport"})
    return db1b_airports, nums


def determineClosebyAirports(files):
    db1b_airports,nums = airportListNums(files)
    df = pd.read_csv(f"{INPUT}/airports.dat", header = None)
    df["ones"] = 1

    cross = df[[6, 7, 4, "ones"]]
    cross = cross.loc[df[4] != "\\N"]
    cross = cross.rename(columns = {6 : "lat", 7 : "lon", 4 : "airport"})
    cross = cross.merge(db1b_airports, on = "airport", how = "right")

    cross = cross.loc[cross.lat.notnull()]

    cross = cross.merge(cross, on = "ones", how = "outer")
    cross = cross.loc[cross["lat_x"] != cross["lat_y"]]

    cross["distance"] = haversine_np(
        cross["lon_x"], cross["lat_x"], cross["lon_y"], cross["lat_y"]
    )

    cross = cross.loc[cross.distance < 60]

    # search over all closeby airports
    X = list(cross.airport_x.unique())
    Y = []
    for x in X:
        y = set([x])
        sub = list(cross.loc[cross.airport_x == x].airport_y)
        y.update(set(sub))
        z = set([])
        for s in sub:
            sub2 = list(cross.loc[cross.airport_x == s].airport_y)
            z.update(set(sub2))
        if z.difference(y)!= set([]):
            sub = list(z.difference(y))
            y.update(z)
            z = set([])
            for s in sub:
                sub2 = list(cross.loc[cross.airport_x == s].airport_y)
                z.update(set(sub2))
            if z.difference(y) != set([]):
                sub = list(z.difference(y))
                y.update(z)
                z = set([])
                for s in sub:
                    sub2 = list(cross.loc[cross.airport_x == s].airport_y)
                    z.update(set(sub2))
                if z.difference(y) != set([]):
                    print(x)
        Y.append(list(y))

    name = []
    res = list(set(tuple(sorted(sub)) for sub in Y))
    for r in res:
        paxT = nums.loc[nums.origin.isin(r)]
        name.extend([paxT.loc[paxT.paxx.idxmax()]["origin"]])
    return res, name

def sumSegmentPath(df):
    ##### NOW AGG PAX TOTALS
    # nonstop only
    df1 = df.loc[df.connect1.isnull()][["origin", "dest", "paxx", "quarter", "year"]]
    # now 1 connection
    df1 = df1.append(
        df.loc[(df.connect1.notnull()) & (df.connect2.isnull())][["origin", "connect1", "paxx", "quarter", "year"]].rename(columns = {"connect1" : "dest"})
    )
    df1 = df1.append(
        df.loc[(df.connect1.notnull()) & (df.connect2.isnull())][["connect1", "dest", "paxx", "quarter", "year"]].rename(columns = {"connect1" : "origin"})
    )
    # now 2 connections
    df1 = df1.append(
        df.loc[(df.connect2.notnull()) & (df.connect3.isnull())][["origin", "connect1", "paxx", "quarter", "year"]].rename(columns = {"connect1" : "dest"})
    )
    df1 = df1.append(
        df.loc[(df.connect2.notnull()) & (df.connect3.isnull())][["connect1", "connect2", "paxx", "quarter", "year"]].rename(columns = {"connect1" : "origin", "connect2" : "dest"})
    )
    df1 = df1.append(
        df.loc[(df.connect2.notnull()) & (df.connect3.isnull())][["connect2", "dest", "paxx", "quarter", "year"]].rename(columns = {"connect2" : "origin"})
    )
    # now 3 connections
    df1 = df1.append(
        df.loc[(df.connect3.notnull()) & (df.connect4.isnull())][["origin", "connect1", "paxx", "quarter", "year"]].rename(columns = {"connect1" : "dest"})
    )
    df1 = df1.append(
        df.loc[(df.connect3.notnull()) & (df.connect4.isnull())][["connect1", "connect2", "paxx", "quarter", "year"]].rename(columns = {"connect1" : "origin", "connect2" : "dest"})
    )
    df1 = df1.append(
        df.loc[(df.connect3.notnull()) & (df.connect4.isnull())][["connect2", "connect3", "paxx", "quarter", "year"]].rename(columns = {"connect2" : "origin", "connect3" : "dest"})
    )
    df1 = df1.append(
        df.loc[(df.connect3.notnull()) & (df.connect4.isnull())][["connect3", "dest", "paxx", "quarter", "year"]].rename(columns = {"connect3" : "origin"})
    )
    # now 4 connections
    df1 = df1.append(
        df.loc[df.connect4.notnull()][["origin", "connect1", "paxx", "quarter", "year"]].rename(columns = {"connect1" : "dest"})
    )
    df1 = df1.append(
        df.loc[df.connect4.notnull()][["connect1", "connect2", "paxx", "quarter", "year"]].rename(columns = {"connect1" : "origin", "connect2" : "dest"})
    )
    df1 = df1.append(
        df.loc[df.connect4.notnull()][["connect2", "connect3", "paxx", "quarter", "year"]].rename(columns = {"connect2" : "origin", "connect3" : "dest"})
    )
    df1 = df1.append(
        df.loc[df.connect4.notnull()][["connect3", "connect4", "paxx", "quarter", "year"]].rename(columns = {"connect3" : "origin", "connect4" : "dest"})
    )
    df1 = df1.append(
        df.loc[df.connect4.notnull()][["connect4", "dest", "paxx", "quarter", "year"]].rename(columns = {"connect4" : "origin"})
    )
    return df1

files = glob.glob(f"{INPUT}/*")
f1 = [f for f in files if "2012" in f]
files = f1 + [f for f in files if "2019" in f]
res, name = determineClosebyAirports(files)

df = pd.DataFrame()
for f in files:
    print(f)
    qtr = pd.read_csv(f)
    df = df.append(qtr)

df = df.reset_index(drop = True)

cols = ["origin", "dest", "connect1", "connect2", "connect3", "connect4"]
for r in range(len(res)):
    for c in cols:
        print(c + " " + name[r])
        df.loc[df[c].isin(res[r]), c] = name[r]


df = df.loc[df.foreign != 1]
df = df.loc[df.legtype != "V"]
df = df.loc[df.interline != 1]
df = df.loc[df.fare > 20].reset_index(drop = True)
df = df.loc[df.fare < df.fare.quantile(.9995)].reset_index(drop = True)

df.to_csv(f"{OUTPUT}/db1b_jmp.csv")


df1 = sumSegmentPath(df)


cols = ["origin", "dest", "quarter", "year"]

totalPax = df1.groupby(cols).paxx.sum().reset_index(drop = False)
directPax = df.loc[df.connect1.isnull()][cols + ["paxx"]] \
    .groupby(cols).paxx.sum().reset_index(drop = False)
connectPax = df.loc[df.connect1.notnull()][cols + ["paxx"]] \
    .groupby(cols).paxx.sum().reset_index(drop = False)
ODPax = df.groupby(cols).paxx.sum().reset_index(drop = False)

result = directPax.rename(columns = {"paxx" : "directPax"})
result = result \
    .merge(connectPax.rename(
        columns = {"paxx" : "connectPax"}), on = cols, how = "left"
    ) \
    .merge(totalPax.rename(
        columns = {"paxx" : "totalPax"}), on = cols, how = "left"
    ) \
    .merge(ODPax.rename(columns = {"paxx" : "odPax"}), on = cols, how = "left") \
    .fillna(0)

result["fracDirect"] = result.directPax / result.odPax
result["nonconnectingTraffic"] = result.directPax / result.totalPax
result["monthly"] = result.totalPax / 3


nsCarrier = df.loc[df.connect1.isnull()]
# half a 50 seater plane, at least 8 times a month (weekend only), and there are 3 months
nsCarrier = nsCarrier.merge(
    totalPax.loc[totalPax.paxx > .5 * 50 * 8 * 3][cols],
    on = cols,
    how = "inner"
)
nsCarrier = nsCarrier.groupby(cols + ["mktcar1"])["paxx"] \
    .sum().reset_index(drop = False)
nsCarrier = nsCarrier.loc[nsCarrier.paxx >= .5 * 50 * 8 * 3]
nsCarrier["totalOD"] = nsCarrier.groupby(cols).paxx.transform("sum")
nsCarrier["fracOD"] = nsCarrier.paxx / nsCarrier.totalOD
nsCarrier = nsCarrier.loc[nsCarrier.fracOD > .01]
nsCarrier = nsCarrier[cols + ["mktcar1"]].drop_duplicates()
nsCarrier = nsCarrier.groupby(cols)["mktcar1"].unique()
nsCarrier = nsCarrier.reset_index(drop = False)
nsCarrier["numCar"] = nsCarrier.mktcar1.str.len()

result = result.merge(nsCarrier, on = cols, how="inner")

# potential markets to explore
print("potential markets to explore")
df[["origin", "dest"]].drop_duplicates().shape
#(73301, 2)

df.loc[df.connect1.isnull()][["origin", "dest"]].drop_duplicates().shape
#(9767, 2)

nsCarrier.loc[nsCarrier.numCar == 1][["origin", "dest"]].drop_duplicates().shape
#(3980, 2)

# calc frac above
# frac of total OD traffic
result.loc[result.mktcar1.str.len() ==1 ].odPax.sum() / df.paxx.sum()
#0.14451198971334225

# rev per quarter
tmp = df.loc[df.connect1.isnull()].merge(
    result.loc[result.mktcar1.str.len() == 1][["origin", "dest", "quarter", "year"]] \
        .drop_duplicates()
)
tmp["rev"] = tmp.paxx * tmp.fare
tmp.groupby(["year", "quarter"]).rev.sum().mean() / 1000000000
del(tmp)

print("all markets, means, mono vs duo, frac and non-con")
a = result.fracDirect.mean()
b = result.nonconnectingTraffic.mean()
c = result.loc[result.numCar > 1].fracDirect.mean()
d = result.loc[result.numCar > 1].nonconnectingTraffic.mean()

a, b, c, d
#(0.7611249297018865, 0.5701224977020173, 0.8278477209110113, 0.6137059164076264)


print("all markets, medians, mono vs duo, frac and non-con")
a = result.fracDirect.median()
b = result.nonconnectingTraffic.median()
c = result.loc[result.numCar > 1].fracDirect.median()
d = result.loc[result.numCar > 1].nonconnectingTraffic.median()

a, b, c, d
#(0.8237999425122161, 0.5609815950920245, 0.8776774958094059, 0.6231270613609098)

print("potential markets, medians, mono, frac and non-con, after cleaning")
potMarkets = result.loc[result.numCar == 1]
potMarkets = potMarkets.loc[potMarkets.monthly < 15000]

a = potMarkets.fracDirect.median()
b = potMarkets.nonconnectingTraffic.median()

a, b
#(0.7600798904896322, 0.49670404675347246)


potMarkets = potMarkets.merge(
    df[["origin", "dest", "nstopdist"]].drop_duplicates(),
    on = ["origin", "dest"],
    how = "left"
)
potMarkets[["nonconnectingTraffic", "fracDirect", "nstopdist"]].corr()

#                       nonconnectingTraffic  fracDirect  nstopdist
# nonconnectingTraffic              1.000000   -0.330061   0.238462
# fracDirect                       -0.330061    1.000000  -0.519039
# nstopdist                         0.238462   -0.519039   1.000000


potMarkets[["origin", "dest"]].drop_duplicates().shape
# (3906, 2)

print("here are the best markets")
potMarkets.loc[
    (potMarkets.year == 2019) & (potMarkets.fracDirect > .95) & \
    (potMarkets.nonconnectingTraffic > .95)
].mktcar1.str[0].value_counts().sum()
#556
potMarkets.loc[
    (potMarkets.year == 2019) & (potMarkets.fracDirect > .95) & \
    (potMarkets.nonconnectingTraffic > .95)
].mktcar1.str[0].value_counts()
# G4    368
# NK    155
# AS     18
# B6     15
# Name: mktcar1, dtype: int64


dfR = pd.read_csv(f"{INPUT}/airline_routes.csv", sep = "\t", header = None)
dfR[0] = dfR[0].str.strip()
dfR[1] = dfR[1].str.strip()
dfR[0] = dfR[0].astype("str")
dfR[1] = dfR[1].astype("str")
dfR.rename(columns = {0 : "origin", 1 : "dest", 2 : "year", 3 : "comp"}, inplace = True)

OD = result.merge(dfR, on = ["origin", "dest", "year"], how = "right")
OD_hist = OD.groupby(["origin", "dest"])[["nonconnectingTraffic", "fracDirect"]] \
    .mean().reset_index(drop = False)

#potMarkets = potMarkets.loc[potMarkets.mktcar1.str[0] != "NK"]
#potMarkets = potMarkets.loc[potMarkets.mktcar1.str[0] != "G4"]

# how do the selected markets compare in char space
print("how markets compare in char space")
distCompare = potMarkets \
    .groupby(["origin", "dest"])[["nonconnectingTraffic", "fracDirect"]] \
    .mean().reset_index(drop = False)
distCompare["nonconnectingTraffic"] = 1 - distCompare.nonconnectingTraffic
distCompare["fracDirect"] = 1 - distCompare.fracDirect
distCompare["L1"] = distCompare.fracDirect + distCompare.nonconnectingTraffic
distCompare["L2"] = \
    np.sqrt(distCompare.fracDirect ** 2 + distCompare.nonconnectingTraffic ** 2)
distRoutes = distCompare.merge(dfR, on = ["origin", "dest"], how = "right")

D1 = [
    stats.percentileofscore(distCompare.L1,distRoutes.L1.values[x]) \
    for x in range(len(distRoutes.L1.values))
]
D2 = [
    stats.percentileofscore(distCompare.L2,distRoutes.L2.values[x]) \
    for x in range(len(distRoutes.L2.values))
]
np.nanmean(D1)
np.nanmean(D2)
stats.percentileofscore(distCompare.L1, distCompare.L1.mean())
stats.percentileofscore(distCompare.L2, distCompare.L2.mean())
# >>> np.nanmean(D1)
# 28.921067404466903
# >>> np.nanmean(D2)
# 25.283692208989933
# >>> stats.percentileofscore(distCompare.L1,distCompare.L1.mean())
# 40.695296523517385
# >>> stats.percentileofscore(distCompare.L2,distCompare.L2.mean())
# 39.97955010224949

print("how markets compare in char space, without NK and G4")
distCompare = potMarkets.copy()
distCompare = distCompare.loc[distCompare.mktcar1.str[0] != "NK"]
distCompare = distCompare.loc[distCompare.mktcar1.str[0] != "G4"]
distCompare = distCompare \
    .groupby(["origin", "dest"])[["nonconnectingTraffic", "fracDirect"]] \
    .mean().reset_index(drop=False)
distCompare["nonconnectingTraffic"] = 1 - distCompare.nonconnectingTraffic
distCompare["fracDirect"] = 1 - distCompare.fracDirect
distCompare["L1"] = distCompare.fracDirect + distCompare.nonconnectingTraffic
distCompare["L2"] = \
    np.sqrt(distCompare.fracDirect ** 2 + distCompare.nonconnectingTraffic ** 2)
distRoutes = distCompare.merge(dfR, on = ["origin", "dest"], how = "right")

D1 = [
    stats.percentileofscore(distCompare.L1,distRoutes.L1.values[x]) \
    for x in range(len(distRoutes.L1.values))
]
D2 = [
    stats.percentileofscore(distCompare.L2,distRoutes.L2.values[x]) \
    for x in range(len(distRoutes.L2.values))
]
np.nanmean(D1)
np.nanmean(D2)
stats.percentileofscore(distCompare.L1,distCompare.L1.mean())
stats.percentileofscore(distCompare.L2,distCompare.L2.mean())
# >>> np.nanmean(D1)
# 18.03487979958568
# >>> np.nanmean(D2)
# 15.697114226525992
# >>> stats.percentileofscore(distCompare.L1,distCompare.L1.mean())
# 43.24324324324324
# >>> stats.percentileofscore(distCompare.L2,distCompare.L2.mean())
# 40.29484029484029

# get ready to create plots
X = potMarkets.groupby(["origin", "dest"])["nonconnectingTraffic"] \
    .mean().reset_index(drop = False)
Y = potMarkets.groupby(["origin", "dest"])["fracDirect"] \
    .mean().reset_index(drop = False)

fig = plt.figure()
ax = fig.add_subplot(111)    # The big subplot
plt.scatter(
    100 * Y.fracDirect,
    100 * X.nonconnectingTraffic,
    s = 1,
    color = sns.xkcd_palette(COLORS)[0],
    label = "DB1B data"
)
plt.scatter(
    100 * OD_hist.fracDirect,
    100 * OD_hist.nonconnectingTraffic,
    s = 7,
    color = sns.xkcd_palette(COLORS)[3],
    marker = "s",
    label = "Selected Routes"
)
plt.xlabel("Nonstop Traffic Percentage")
plt.ylabel("Non-connecting Traffic Percentage")
plt.axvline(
    100 * Y.fracDirect.mean(),
    linestyle = "-.",
    linewidth = 2,
    color = sns.xkcd_palette(COLORS)[1],
    label = "Mean"
)
plt.axhline(
    100 * X.nonconnectingTraffic.mean(),
    linestyle = "-.",
    linewidth = 2,
    color = sns.xkcd_palette(COLORS)[1]
)

reg = sm.OLS(
    100 * Y.fracDirect,
    sm.add_constant(100 * X.nonconnectingTraffic)
).fit()
X_plot = np.linspace(0, 100, 100)
plt.plot(
    X_plot, reg.params[0] + X_plot * reg.params[1],
    linestyle = "-",
    linewidth = 2,
    color = sns.xkcd_palette(COLORS)[2],
    label = "Regression Line"
)
plt.legend()

plt.savefig(
    f"{OUTPUT}/db1b_routes.pdf",
    bbox_inches = "tight",
    format = FIG_FORMAT,
    dpi = DPI)
plt.clf()


potMarkets = potMarkets.loc[potMarkets.mktcar1.str[0] != "NK"]
potMarkets = potMarkets.loc[potMarkets.mktcar1.str[0] != "G4"]
X = potMarkets.groupby(["origin", "dest"])["nonconnectingTraffic"] \
    .mean().reset_index(drop = False)
Y = potMarkets.groupby(["origin", "dest"])["fracDirect"] \
    .mean().reset_index(drop = False)

fig = plt.figure()
ax = fig.add_subplot(111) # The big subplot
plt.scatter(
    100 * Y.fracDirect,
    100 * X.nonconnectingTraffic,
    s = 1,
    color = sns.xkcd_palette(COLORS)[0],
    label = "DB1B data"
)
plt.scatter(
    100 * OD_hist.fracDirect,
    100 * OD_hist.nonconnectingTraffic,
    s = 7,
    color = sns.xkcd_palette(COLORS)[3],
    marker = "s",
    label = "Selected Routes"
)
plt.xlabel("Nonstop Traffic Percentage")
plt.ylabel("Non-connecting Traffic Percentage")
plt.axvline(
    100 * Y.fracDirect.mean(),
    linestyle = "-.",
    linewidth = 2,
    color = sns.xkcd_palette(COLORS)[1],
    label = "Mean"
)
plt.axhline(
    100 * X.nonconnectingTraffic.mean(),
    linestyle = "-.",
    linewidth = 2,
    color=sns.xkcd_palette(COLORS)[1]
)
reg = sm.OLS(
    100 * Y.fracDirect,
    sm.add_constant(100 * X.nonconnectingTraffic)
).fit()
X_plot = np.linspace(0, 100, 100)
plt.plot(
    X_plot,
    reg.params[0] + X_plot * reg.params[1],
    linestyle = "-",
    linewidth = 2,
    color = sns.xkcd_palette(COLORS)[2],
    label = "Regression Line"
)
plt.legend()

plt.savefig(
    f"{OUTPUT}/db1b_routes_small.pdf",
    bbox_inches = "tight",
    format = FIG_FORMAT,
    dpi = DPI
)
plt.clf()

# then run
#scp -p /mnt/data0/bts_airline/resources/db1b_routes.pdf kw468@bfg:/home/kw468/Projects/airlines_jmp/output/
#scp -p /mnt/data0/bts_airline/resources/db1b_routes.pdf kw468@bfg:/home/kw468/Projects/airlines_jmp/output/

tbl_fares = df.loc[df.connect1.isnull()] \
    .merge(dfR, on = ["origin", "dest", "year"], how = "inner")
tbl_fares.loc[tbl_fares.year == 2012, "fare"] = tbl_fares.fare * 1.12
tbl_fares["fare"] = tbl_fares.fare * tbl_fares.paxx
tbl_fares["tpax"] = tbl_fares.groupby(["origin", "dest"]).paxx.transform("sum")
tbl_fares["fare"] = tbl_fares["fare"] / tbl_fares["tpax"]
tbl_fares = tbl_fares.groupby(["origin", "dest"]).fare.sum().reset_index(drop = False)

TBL1 = OD.groupby(["origin", "dest"])["directPax", "connectPax", "totalPax"].mean()
TBL1 = TBL1.reset_index(drop = False)
TBL1 = TBL1.merge(tbl_fares, on = ["origin", "dest"], how = "inner")
TBL1[["directPax", "connectPax", "totalPax", "fare"]] = \
    TBL1[["directPax", "connectPax", "totalPax", "fare"]].astype("int")
TBL1["route"] = TBL1[["origin", "dest"]].min(axis = 1) + \
    TBL1[["origin", "dest"]].max(axis = 1)
TBL1 = TBL1.sort_values(["route", "origin"]).reset_index(drop = False)
TBL11 = TBL1.copy().drop_duplicates("route", keep="last")
TBL11.rename(columns = {"origin" : "d"}, inplace = True)
TBL11.rename(columns = {"dest" : "origin"}, inplace = True)
TBL11.rename(columns = {"d" : "dest"}, inplace = True)
TBL1 = TBL1.merge(TBL11, on = ["origin", "dest"], how="inner")
TBL1 = TBL1[[
    "origin", "dest", "directPax_x", "directPax_y", "totalPax_x",
    "totalPax_y", "connectPax_x", "connectPax_y", "fare_x", "fare_y"
]]

tbl_fares1 = df.loc[(df.connect1.notnull()) & (df.connect2.isnull())] \
    .merge(dfR, on = ["origin", "dest", "year"], how = "inner")
tbl_fares1.loc[tbl_fares1.year == 2012, "fare"] = tbl_fares1.fare * 1.12
tbl_fares1["route"] = tbl_fares1[["origin", "dest"]].min(axis = 1) + \
    tbl_fares1[["origin", "dest"]].max(axis = 1)
tbl_fares1["fare"] = tbl_fares1.fare*tbl_fares1.paxx
tbl_fares1["tpax"] = tbl_fares1.groupby(["route"]).paxx.transform("sum")
tbl_fares1["fare"] = tbl_fares1["fare"] / tbl_fares1["tpax"]
tbl_fares1 = tbl_fares1.groupby(["route"]).fare.sum() \
    .astype("int").reset_index(drop = False)

tbl_fares1["origin"] = tbl_fares1.route.str[0:3]
tbl_fares1["dest"] = tbl_fares1.route.str[3:6]
tbl_fares1.rename(columns = {"fare" : "connectfare"}, inplace = True)

TBL1 = TBL1.merge(tbl_fares1[["origin", "dest", "connectfare"]])

def f1(x):
    return f"{x:,}"

with open(f"{OUTPUT}/routeSummary.tex", "w") as tf:
    tf.write(TBL1.to_latex(index = False, formatters = [None, None] + [f1] * 9))

#scp -p /mnt/data0/bts_airline/resources/routeSummary.tex kw468@bfg:/home/kw468/Projects/airlines_jmp/output/
