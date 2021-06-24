"""
    This script calculates the number of passengers per booking in
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

import glob
import pandas as pd
import json
import os
import re
import numpy as np
import multiprocessing as mp

INTPUT = OUTPUT = "../../data"

NUM_PROCESSES = 24

# -------------------------------------------------------------------------------
# DEFINE FUNCTIONS
# -------------------------------------------------------------------------------

def processASFile(fname):
    #print(fname)
    with open(fname) as json_data:
        data = json.load(json_data,)
    # process the json file
    X = pd.DataFrame()
    for d in data:
        if d["stops"] == "Nonstop":
            try:
                flightInfo = [
                    d["flights"][0]["flight_number"],
                    d["flights"][0]["depart_full_time"],
                    d["flights"][0]["arrive_full_time"]
                ]
                sdate = searchDate(fname)
                ddate = depDateRoute(fname)
                info = ddate + sdate + flightInfo
                seatMap = determineSeatAvail(d)
                seatMap = pd.concat(
                    (seatMap, pd.DataFrame([info]*seatMap.shape[0])),
                    axis = 1
                )
                X = X.append(seatMap)
            except KeyError:
                print("error with file" + fname)
                pass
    return X

def searchDate(fname):
    date=re.split("/",fname)[6]
    return re.split("-|\_",date)[0:3]

def depDateRoute(fname):
    date=re.split("/",fname)[-1]
    return re.split("\_|.json",date)[0:6]

def determineSeatAvail(d):
    X = []
    for d1 in d["seat_map"]["SeatMapModels"]:
        for d2 in d1["Rows"]:
            for d3 in d2["RowItems"]:
                if d3["SeatId"] != None:
                    #if d3["IsSeatAvailable"] == True:
                    X.extend([d3["SeatTitle"].split(":")])
    df = pd.DataFrame(X)
    return df


# -------------------------------------------------------------------------------
# PROCESS THE DATA
# -------------------------------------------------------------------------------

files = glob.glob(f"{INTPUT}/*/alaskaair/*.json")
with mp.Pool(NUM_PROCESSES) as p:
    df = p.map(processASFile, files)
    df = pd.concat(df)
    p.join()

df.columns = list(range(df.shape[1]))
df.rename(
    columns = {
        0 : "seat",
        1 : "status",
        2 : "ddate_y",
        3 : "ddate_m",
        4 : "ddate_d",
        5 : "origin",
        6 : "to",
        7 : "dest",
        8 : "sdate_y",
        9 : "sdate_m",
        10 : "sdate_d",
        11 : "flightNum",
        12 : "timeIn",
        13 : "timeOut",
    },
    inplace = True
)

df.drop(columns = "to", inplace = True)
df = df.reset_index(drop = True)


df.to_parquet(f"{OUTPUT}/seatmap_as_raw.parquet")

df["flight"] = df.groupby(["origin", "dest", "flightNum"]).ngroup()
df["ddate"] = pd.to_datetime(
    df["ddate_y"] + df["ddate_m"] + df["ddate_d"],
    format = "%Y%m%d"
)
df["sdate"] = pd.to_datetime(
    df["sdate_y"] + df["sdate_m"] + df["sdate_d"],
    format = "%Y%m%d"
) #df.groupby(["sdate_y", "sdate_m", "sdate_d"]).ngroup()
df["avail"] = 0
df.loc[df.status.str.contains("Unavailable"), "avail"] = 1
df["row"] = df.seat.str.extract("(\d+)")
df["seatPos"] = df.seat.str.replace("\d+", "")

df = df[["flight", "sdate", "ddate", "avail", "row", "seatPos"]]

df = df.drop_duplicates(["flight", "sdate", "ddate", "row", "seatPos"])

df1 = pd.pivot_table(
    df,
    values = "avail",
    index = ["flight", "sdate", "ddate", "row"],
    columns=["seatPos"]
)

df1 = df1.reset_index(drop = False)
df1["tdate"] = (df1.ddate-df1.sdate).dt.days

df1 = df1.sort_values(["flight", "ddate", "row", "tdate"]).reset_index(drop = True)
df1["numBookedRow"] = df1[["A", "B", "C", "D", "E", "F"]].sum(axis = 1)
df1["s_numBookedRow"] = df1.groupby(["flight", "ddate", "row"])["numBookedRow"].shift(-1)
df1["dif"] = df1.numBookedRow - df1.s_numBookedRow #dif > 0 is a sale


##########################################################################################
# MULTI BOOKING
##########################################################################################

df2 = df1.loc[df1.dif == 2]
idx = df2.index + 1
df2 = df2.append(df1.iloc[idx,:])
df2 = df2.sort_values(["flight", "ddate", "row", "tdate"]).reset_index(drop = True)
df2["AA"] = df2.groupby(["flight", "ddate", "row"]).A.shift(1)
df2["BB"] = df2.groupby(["flight", "ddate", "row"]).B.shift(1)
df2["CC"] = df2.groupby(["flight", "ddate", "row"]).C.shift(1)
df2["DD"] = df2.groupby(["flight", "ddate", "row"]).D.shift(1)
df2["EE"] = df2.groupby(["flight", "ddate", "row"]).E.shift(1)
df2["FF"] = df2.groupby(["flight", "ddate", "row"]).F.shift(1)

df3 = df2.iloc[[k for k in range(df2.shape[0]) if k %2],:]
df3["aa"] = df3.AA - df3.A
df3["bb"] = df3.BB - df3.B
df3["cc"] = df3.CC - df3.C
df3["dd"] = df3.DD - df3.D
df3["ee"] = df3.EE - df3.E
df3["ff"] = df3.FF - df3.F

X = df3[["aa","bb","cc","dd","ee","ff"]].values
counter = 0
y = np.zeros(len(X))
for x in range(len(X)):
    try:
        y = X[x]
        y = y[~np.isnan(y)]
        y = list(np.where(y == 1)[0])
        if y[-1] - y[0] == 1:
            counter += 1
    except: pass


print("fraction of 2 bookings to include")
counter/len(X)
#0.8202318829207047

print("row stats")
df1.loc[df1.dif >= 0].dif.value_counts()
# >>> df1.loc[df1.dif >= 0].dif.value_counts()
# 0.0    14464729
# 1.0      340481
# 2.0      101948
# 3.0       21267
# 4.0       14588
# 6.0        1227
# 5.0        1010
# Name: dif, dtype: int64
# >>>
df1.loc[df1.dif >= 3].dif.count() / df1.loc[df1.dif >= 1].dif.count()
#0.079272289868705

print("fraction of <= 2 ppl that are single")
df4 = df1.loc[df1.dif >= 0].dif.value_counts().reset_index(drop=False)
df4["exp"] = df4["index"] * df4.dif
df4.loc[df4["index"] <= 2].exp.sum() / df4.exp.sum()
#0.8018019212244935
# percent less than equal to 2

print("1 - fraction of single pax and bookings")
### UPPER BOUNDS
(df4.exp.sum() - df4.loc[df4["index"] <= 1].exp.sum() - 2 * (len(X) - counter)) / \
    df4.exp.sum()
#0.4445254528369139
(df4.dif.sum() - df4.loc[df4["index"] <= 1].dif.sum() - (len(X) - counter)) / \
    df4.loc[( df4["index"]> 0)].dif.sum()
#0.253293820665486

print("avg pax per booking")
# upper bound on the number of passengers
(df4.exp.sum() - (len(X) - counter)) / df4.loc[(df4["index"] > 0)].dif.sum()
#1.3747890310725233

byday = df1.loc[df1.dif >= 0]
# >>> byday.groupby("tdate").dif.mean()
# tdate
# 0     0.568709
# 1     0.109737
# 2     0.065602
# 3     0.044413
# 4     0.043522
# 5     0.043903
# 6     0.051340
# 7     0.045002
# 8     0.040926
# 9     0.039188
# 10    0.037251
# 11    0.035103
# 12    0.041550
# 13    0.055914
# 14    0.045504
# 15    0.042147
# 16    0.041600
# 17    0.039487
# 18    0.037539
# 19    0.040607
# 20    0.048203
# 21    0.044184
# 22    0.041371
# 23    0.041837
# 24    0.040992
# 25    0.039200
# 26    0.040708
# 27    0.042304
# 28    0.039906
# 29    0.039246
# 30    0.038651
# 31    0.038533
# 32    0.036207
# 33    0.037155
# 34    0.036765
# 35    0.034667
# 36    0.034692
# 37    0.032405
# 38    0.034114
# 39    0.031295
# 40    0.031874
# 41    0.031893
# 42    0.029221
# 43    0.028882
# 44    0.028394
# 45    0.030349
# 46    0.027056
# 47    0.027282
# 48    0.027694
# 49    0.025466
# 50    0.025203
# 51    0.025478
# 52    0.026843
# 53    0.024409
# 54    0.022212
# 55    0.022598
# 56    0.021003
# 57    0.020792
# 58    0.021554
# 59    0.036731
# Name: dif, dtype: float64
# >>>
