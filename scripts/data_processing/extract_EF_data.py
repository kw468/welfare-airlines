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

import pandas as pd
import numpy as np

INPUT = OUTPUT = "../../data"

# --------------------------------------------------------------------------------
# Define function definitions
# --------------------------------------------------------------------------------

df = pd.read_csv(f"{INPUT}/Master_SM.txt", header = None)
df = df.loc[df[14] != " "]
df[14] = df[14].str.strip()
df.loc[df[14] == "blocked", 14] = "occupied"
df.loc[df[14] != "occupied", 14] = "available"

df["ones"] = 1
df = df.groupby([0, 1, 2, 3, 9, 10, 11, 14]).ones.sum().reset_index()
df = df.pivot_table(
    values = "ones",
    index = [0, 1, 2, 3, 9, 10, 11],
    columns = 14,
    aggfunc = "mean"
).reset_index()

# label and parse columns
df[["origin", "dest"]] 	= df[0].str.split(" ", expand = True)
df["ddate"] = pd.to_datetime(df[1])
df["sdate"] = pd.to_datetime(dict(year = df[9], month = df[10], day = df[11]))
df["tdate"] = (df.ddate - df.sdate).dt.days
df["acode"] = df[2]
df["flightNum"] = df[3]
df = df[[
    "sdate", "ddate", "tdate", "flightNum", "acode",
    "origin", "dest", "available", "occupied"
]]
df = df.fillna(0)
df["capacity"] = df.available + df.occupied
df["acode"] = df.acode.str.strip()

dfP = pd.read_csv(f"{INPUT}/Max_Yapta_price.txt", header = None)

# label and parse columns
dfP = dfP.loc[dfP[8].isnull()]
dfP["sdate"] = pd.to_datetime(dfP[0])
dfP["origin"] = dfP[1]
dfP["dest"] = dfP[2]
dfP["acode"] = dfP[3]
dfP["flightNum"] = dfP[4]
dfP["fare"] = dfP[15]
dfP["ddate"] = pd.to_datetime(dict(year = dfP[5], month = dfP[6], day = dfP[7]))
dfP["tdate"] = (dfP.ddate - dfP.sdate).dt.days
merge_key = ["sdate", "ddate", "tdate", "flightNum", "acode", "origin", "dest"]
dfP = dfP[merge_key + ["fare"]]

# combine data
df = df.merge(dfP, on = merge_key, how = "inner")

# create flight grid for ffill() and then merge data, and then ffill()
cols = ["acode", "flightNum", "ddate", "tdate", "origin", "dest"]
ss = df[cols].drop_duplicates().copy()
dfTimes = pd.DataFrame()
# 0 is departure date
dfTimes["tdate"] = np.arange(ss.tdate.min(), ss.tdate.max() + 1, 1)
cols = ["acode", "flightNum", "ddate", "origin", "dest"]
dfIndex = ss[cols].copy().drop_duplicates()
dfTimes["ones"] = dfIndex["ones"] = 1
dfIndex = dfTimes.merge(dfIndex, on = "ones")
dfIndex.drop(columns = "ones", inplace = True)

ss["ind"] = 1
cols = ["acode", "flightNum", "ddate", "tdate", "origin", "dest"]
dfIndex = dfIndex.merge(ss, on = cols, how = "left")
cols = ["acode", "flightNum", "ddate", "origin", "dest"]
tmp = dfIndex.loc[dfIndex["ind"].notnull() == True] \
    .groupby(cols).tdate.max().reset_index(drop = False)
tmp.rename(columns = {"tdate" : "maxT"}, inplace = True)
dfIndex = dfIndex.merge(tmp, on = cols, how = "left")
dfIndex.shape
#(226859, 8)

# trim the data at the lower bound -- the first time we see the flight
dfIndex = dfIndex.loc[dfIndex.tdate <= dfIndex.maxT].reset_index(drop = True)
dfIndex.drop(columns = "maxT", inplace = True)
dfIndex.shape
#(205670, 7)

tmp = dfIndex.loc[dfIndex["ind"].notnull() == True] \ 
    .groupby(cols).tdate.min().reset_index(drop = False)
tmp.rename(columns = {"tdate" : "minT"}, inplace = True)
dfIndex = dfIndex.merge(tmp, on = cols, how = "left")

# do not trim the data at the upper bound -- the last time we see the flight
# this is because it may have sold out so let"s check it later
dfIndex.drop(columns = "ind", inplace = True)

# sort the data in descending error for ffill and bfill
cols = ["origin", "dest", "ddate", "acode", "flightNum", "tdate"]
df = df.merge(dfIndex, on = cols, how = "right")
df = df.sort_values(cols, ascending = False).reset_index(drop = True)

# cut the data; drop obs such that upper bound is more than a week before flight leaves
df = df.loc[(df.tdate >= df.minT) | (df.minT <= 7)]
df.shape
#(201850, 12)

# execute ffill and bfill -- this takes a while
cols = ["origin", "dest", "ddate", "acode", "flightNum"]
df = df.groupby(cols, as_index = False).apply(lambda group: group.ffill())
#df = df.groupby(cols, as_index = False).apply(lambda group: group.bfill())

df = df.loc[df.available.notnull()]
df.shape
#(201850, 12)

df["lf"] = df.occupied / df.capacity

cols = ["origin", "dest", "ddate"]
df["numCar"] = df.groupby(cols).acode.transform("nunique")

df = df.loc[df.minT <= 2] # must track flight within 2 days of departure

cols = ["origin", "dest", "ddate", "flightNum"]
df["ones"] = 1
df["numObs"] = df.groupby(cols)["ones"].transform("sum")
df = df.loc[df.numObs >= 59].reset_index(drop = True)
# remove flights that are tracked for less than 59 days

df.rename(columns =  {"available" : "seats"}, inplace = True)

df["fare"] = df["fare"] * 1.12 #adjustment for inflation between sample periods

df = df[[
    "sdate", "ddate", "tdate", "flightNum", "acode", "origin", "dest",
    "seats", "occupied", "capacity", "fare", "lf", "numCar"
]]

df.to_parquet(f"{OUTPUT}/efdata_clean.parquet")
