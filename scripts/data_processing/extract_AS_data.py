"""
    This code processes the Alaska Air Lines data in Williams (2021)
    Inputs:
        * Prices and seat maps scraped from alaskaair.com
        * Bucket availability from BCD Travel.
    Operations:
        * The code constructs enplanement totals from json files
        * Prices are extracted from from the menu for every flight
        * Merge results to bucket availability (censored)
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

import glob
import pandas as pd 
import json
import os
import re
import numpy as np
import multiprocessing as mp

INPUT = OUTPUT = "../../data"

NUM_PROCESSES = 24

# --------------------------------------------------------------------------------
# Define function definitions
# --------------------------------------------------------------------------------

# Open AS json file and process flight number, travel times, prices, and capacities
def processASFile(fname):
    #print(fname)
    with open(fname) as json_data:
        data = json.load(json_data) 
    # process the json file
    X = []
    for d in data:
        if d["stops"] == "Nonstop":
            try:
                flightInfo = [
                    d["flights"][0]["flight_number"],
                    d["flights"][0]["depart_full_time"],
                    d["flights"][0]["arrive_full_time"]
                ]
                duration = [d["total_duration"]]
                sdate = searchDate(fname)
                ddate = depDateRoute(fname)
                prices = extractPrices(d)
                capLoad = determineSeatCount(d)
                capacity1 = [capLoad[0]]
                seatM1 = [capLoad[1]]
                capacity2 = [capLoad[2]]
                seatM2 = [capLoad[3]]
                X.extend([
                    flightInfo + duration + sdate + ddate + prices + \
                    capacity1 + seatM1 + capacity2 + seatM2 + [1]
                ])
            except KeyError:
                print("error with file" + fname)
                pass    
        else:
            try:
                numFlights = len(d["flights"])
                flightInfo = [
                    "_".join(d["flights"][i]["flight_number"] for i in range(numFlights)),
                    d["flights"][0]["depart_full_time"],
                    d["flights"][-1]["arrive_full_time"]
                ]
                duration = [np.nan]
                sdate = searchDate(fname)
                ddate = depDateRoute(fname)
                prices = extractPrices(d)
                capacity1 = [np.nan]
                seatM1 = [np.nan]
                capacity2 = [np.nan]
                seatM2 = [np.nan]
                X.extend([
                    flightInfo + duration + sdate + ddate + prices + \
                    capacity1 + seatM1 + capacity2 + seatM2 + [0]]
                )  
            except KeyError:
                print("error with file" + fname)
                pass
    return X


# extract seats remaining from the seat maps
def extractSeatsRemain(jsonData):
    length = len(jsonData["seat_map"]["SeatMapModels"])
    seats  = [jsonData["seat_map"]["SeatMapModels"][i]["AvailableSeats"] for i in range(length)]
    return sum(seats)

# File records the search data
def searchDate(fname):
    date = fname.split("_utc")[0].split("/")[-1]
    return re.split("-|\_", date)[0:3]

# File records the route
def depDateRoute(fname):
    date = re.split("/", fname)[-1]
    return re.split("\_|.json", date)[0:6]


# Extract pricing menu; we"ll obtain as many fares as we can
def extractPrices(jsonData):
    prices = [
        "lowest_fare_saver_fare",
        "lowest_fare_main_fare",
        "lowest_fare_first_fare",
        "main_fare_refundable_fare",
        "refundable_fare_first_fare"
    ]
    priceResults = [
        re.search(
            "\$(\d+)", jsonData[p].replace(",", ""), re.IGNORECASE
        ) for p in prices
    ]
    return [p.group(1) if p != None else "" for p in priceResults]


# Iterate through the seat map to find seat IDs which are available
def determineSeatCount(d):
    caps = [0] * 4
    it = 0
    for d1 in d["seat_map"]["SeatMapModels"]:
        counter = 0
        counter2 = 0
        for d2 in d1["Rows"]:
            for d3 in d2["RowItems"]:
                if d3["SeatId"] != None:
                    counter += 1
                    if d3["IsSeatAvailable"] == True:
                        counter2 += 1
        if it == 0:
            caps[0] = counter
            caps[1] = counter2
        if it == 1:
            caps[2] = counter
            caps[3] = counter2 
        it += 1       
    return caps

# Glob all files and process them, output data frame
def processASData():
    files = glob.glob(f"{INPUT}/*/alaskaair/*.json")
    assert len(files) > 0, f"""
        Did not find any JSON files in {INPUT}/*/alaskaair.
        Please make sure to unzip the files.
    """

    
    with mp.Pool(NUM_PROCESSES) as p:
        df = p.map(processASFile, files)
        df = [y for x in df for y in x]
        df = pd.DataFrame(df)
    
    df["flightNum"] = df[0].astype(str)
    df["ddate"] = pd.to_datetime(df[7] + df[8] + df[9], format = "%Y%m%d")
    df["sdate"] = pd.to_datetime(df[4] + df[5] + df[6], format = "%Y%m%d")
    df["capacity1"] = df[18].astype(float)
    df["seats1"] = df[19].astype(float)
    df["capacity2"] = df[20].astype(float)
    df["seats2"] = df[21].astype(float)
    df["nonstop"] = df[22].astype(float)
    df["origin"] = df[10]
    df["dest"] = df[12]
    df["saverFare"] = df[13].replace("", np.nan, regex = True).astype(float)
    df["mainFare"] = df[14].replace("", np.nan, regex = True).astype(float)
    df["firstFare"] = df[15].replace("", np.nan, regex = True).astype(float)
    df["refundYFare"] = df[16].replace("", np.nan, regex = True).astype(float)
    df["refundFFare"] = df[17].replace("", np.nan, regex = True).astype(float)
    df["tdate"] = (df.ddate-df.sdate).dt.days
    df = df.drop(df.columns[list(range(23))], axis = 1)
    df["fare"] = df[
        df.columns[df.columns.to_series().str.contains("Fare")]
    ].min(axis = 1)
    return df

# Open BCD file and extract key flight parameters
def processBCDFile(fname):
    with open(fname) as json_data:
        data = json.load(json_data,) 
    # process the json file
    X = []
    for d in data:
        if d["operated_by"] != "":
            try:
                flightInfo = d["airline_code"]
                flightAvail = d["seat_availability"]
                sdate = searchDate(fname)
                ddate = depDateRoute(fname)
                X.extend([[flightInfo] + sdate + ddate + [flightAvail]])
            except KeyError:
                print("error with file" + fname)
                pass    
    return X


# Process BCD bucket file; dictionary buckets and fill in bucket availability
def processBuckets():
    files = glob.glob(f"{INPUT}/*/bcdtraval/*.json")
    assert len(files) > 0, f"""
        Did not find any JSON files in {INPUT}/*/bcdtraval.
        Please make sure to unzip the files.
    """

    with mp.Pool(NUM_PROCESSES) as p:
        df = p.map(processBCDFile,files)
        df = [y for x in df for y in x]
        df = pd.DataFrame(df)

    dftemp = pd.DataFrame(
        df[10].apply(lambda x: dict(z.replace(" ", "").split("-") for z in x)) \
            .values.tolist()
    )
    df["bucket_" + dftemp.columns] = dftemp
    df.drop(10, axis = 1, inplace = True)
    
    df["flightNum"] = df[0]
    df["ddate"] = pd.to_datetime(df[4] + df[5] + df[6], format = "%Y%m%d")
    df["sdate"] = pd.to_datetime(df[1] + df[2] + df[3], format = "%Y%m%d")
    df["origin"] = df[7]
    df["dest"] = df[9]
    df = df.drop(df.columns[list(range(10))], axis = 1)
    
    filter_col = [col for col in df if col.startswith("bucket_")]
    for col in filter_col:
        df[col] = df[col].astype("float")
        df["temp"] = df.groupby(["flightNum", "ddate"])[col].transform("max")
        df.loc[(df["temp"].isnull() == False) & (df[col].isnull() == True), col] = 0
    
    df["maxBucket"] = df[filter_col].max(axis = 1)
    df["tdate"] = (df.ddate-df.sdate).dt.days
    df.drop("temp", axis = 1, inplace = True)
    df["carrier"] = df["flightNum"].astype("str").str[:2]
    df["flightNum"] = df["flightNum"].astype("str").str[3:].astype(float)
    return df

# --------------------------------------------------------------------------------
# Run pooled processes
# --------------------------------------------------------------------------------

bdf = processBuckets()
df = processASData()
 
df.to_parquet(f"{OUTPUT}/as_raw.parquet")
bdf.to_parquet(f"{OUTPUT}/bcd_raw.parquet")

df = pd.read_parquet(f"{INPUT}/as_raw.parquet")
df.shape
# (4814609, 17)
bdf = pd.read_parquet(f"{INPUT}/bcd_raw.parquet")
bdf.shape
# (894621, 34)

# map connecting flight numbers to string
df["fn"] = df.flightNum.str.split("_").map(lambda x: x[0]).astype("float")
bdf.rename(columns = {"flightNum" : "fn"}, inplace = True)

bcd_sellouts = bdf.copy()
bcd_sellouts["lastObs"] = \
     bcd_sellouts.groupby(["origin", "dest", "fn", "ddate"]).tdate.transform("min")
bcd_sellouts = bcd_sellouts.loc[
    (bcd_sellouts.lastObs > 0) & (bcd_sellouts.lastObs == bcd_sellouts.tdate)
]
bcd_sellouts = bcd_sellouts.loc[(bcd_sellouts.lastObs < 7)]
bcd_sellouts = bcd_sellouts[["origin", "dest", "fn", "ddate", "lastObs"]]
bcd_sellouts.shape
# (2118, 5)

asY_sellouts = df.loc[(df.refundYFare.isnull()) & (df.nonstop == 1)].copy()
asY_sellouts["lastObs"] = \
    asY_sellouts.groupby(["origin", "dest", "fn", "ddate"]).tdate.transform("min")
asY_sellouts = asY_sellouts.loc[
    (asY_sellouts.lastObs > 0) & (asY_sellouts.lastObs == asY_sellouts.tdate)
]
asY_sellouts = asY_sellouts[["origin", "dest", "fn", "ddate", "lastObs"]]
asY_sellouts.shape
# (576, 5)

asF_sellouts = df.loc[(df.refundFFare.isnull()) & (df.nonstop == 1)].copy()
asF_sellouts["lastObs"] = \
    asF_sellouts.groupby(["origin", "dest", "fn", "ddate"]).tdate.transform("min")
asF_sellouts = asF_sellouts.loc[
    (asF_sellouts.lastObs > 0) & (asF_sellouts.lastObs == asF_sellouts.tdate)
]
asF_sellouts = asF_sellouts[["origin", "dest", "fn", "ddate", "lastObs"]]
asF_sellouts.shape
# (3296, 5)


# merge files bcd travel and as data together
df = df.merge(
    bdf, on = ["fn", "ddate", "tdate", "sdate", "origin", "dest"], how = "left"
)

# organize F and Y cabins
df["capY"] = np.nan
df["capF"] = np.nan
df["sY"] = np.nan
df["sF"] = np.nan
df["capY"] = df[["capacity1", "capacity2"]].max(axis = 1)
df["capF"] = df[["capacity1", "capacity2"]].min(axis = 1)
df.loc[df.capY < 64, "capF"] = df.capY # the min Y capacity in the data is 64
df.loc[df.capY < 64, "capY"] = 0 # hence, switch F and Y accordingly

# assign seats to proper F and Y cabins
df.loc[df.capY == df.capacity1, "sY"] = df.seats1
df.loc[df.capY == df.capacity2, "sY"] = df.seats2
df.loc[df.capF == df.capacity1, "sF"] = df.seats1
df.loc[df.capF == df.capacity2, "sF"] = df.seats2

# manual adjustment for improper plane size
# the other 177 obs are errors
df.loc[(df.capF == 8) & (df.capY == 177), "capY"] = 141

# create flight grid for ffill() and then merge data, and then ffill()
cols = ["flightNum", "ddate", "tdate", "origin", "dest"]
ss = df[cols].drop_duplicates().copy()
dfTimes = pd.DataFrame()
# 0 is departure date
dfTimes["tdate"] = np.arange(ss.tdate.min(), ss.tdate.max() + 1, 1)
cols = ["flightNum", "ddate", "origin", "dest"]
dfIndex = ss[cols].copy().drop_duplicates()
dfTimes["ones"] = 1
dfIndex["ones"] = 1
dfIndex = dfTimes.merge(dfIndex, on = "ones")
dfIndex.drop(columns = "ones", inplace = True)

ss["ind"] = 1
cols = ["flightNum", "ddate", "tdate", "origin", "dest"]
dfIndex = dfIndex.merge(ss, on = cols, how = "left")
cols = ["flightNum", "ddate", "origin", "dest"]
tmp = dfIndex.loc[dfIndex["ind"].notnull() == True].groupby(cols) \
    .tdate.max().reset_index(drop = False)
tmp.rename(columns = {"tdate" : "maxT"}, inplace = True)
dfIndex = dfIndex.merge(tmp, on = cols, how = "left")
dfIndex.shape
# (7970260, 7)
# trim the data at the lower bound -- the first time we see the flight
dfIndex = dfIndex.loc[dfIndex.tdate <= dfIndex.maxT].reset_index(drop = True)
dfIndex.drop(columns = "maxT", inplace = True)
dfIndex.shape
# (6745117, 6)

tmp = dfIndex.loc[dfIndex["ind"].notnull() == True].groupby(cols) \
    .tdate.min().reset_index(drop = False)
tmp.rename(columns = {"tdate" : "minT"}, inplace = True)
dfIndex = dfIndex.merge(tmp, on = cols, how = "left")

# do not trim the data at the upper bound -- the last time we see the flightffi
# this is because it may have sold out so let"s check it later

dfIndex.drop(columns = "ind", inplace = True)

# sort the data in descending error for ffill and bfill
cols = ["origin", "dest", "ddate", "flightNum", "tdate"]
df = df.merge(dfIndex, on = cols, how = "right")
df = df.sort_values(cols, ascending = False).reset_index(drop = True)
cols = ["origin", "dest", "ddate", "flightNum"]

# we can"t have 0 capacity, it"s either nan or > 0; we"ll use the sell out information in a bit
df.loc[df.capY == 0, "capY"] = np.nan
df.loc[df.capF == 0, "capF"] = np.nan

df.loc[df.sY == 0, "sY"] = np.nan
df.loc[df.sF == 0, "sF"] = np.nan

# define the proper sdate for the grid
df["sdate"] = df.ddate - pd.to_timedelta(df["tdate"], unit = "d")
# cut searches at the end of the sample; we can keep ddates after.
# df = df.loc[df.sdate <= "2019-08-31"]

# cut the data; drop obs such that upper bound is more than a week before flight leaves
df = df.loc[(df.tdate >= df.minT) | (df.minT <= 7)]
df.shape
# (5406252, 51)

print("coach adjustments due to buckets")
print(df.loc[(df.bucket_Y < 7)].shape)
# (26225, 51)
df.loc[(df.bucket_Y < 7), "sY"] = df.bucket_Y

print(df.loc[(df.maxBucket < 7) & (df.capY != 76)].shape)
# (18624, 51)
df.loc[(df.bucket_F < 7) & (df.capY != 76), "sF"] = df.bucket_F


df = df.merge(asY_sellouts, on = ["origin", "dest", "fn", "ddate"], how = "left")
print("coach adjustments due to pricing")
print(df.loc[(df.lastObs.notnull()) & (df.tdate < df.lastObs)].shape)
# (1042, 52)
df.loc[(df.lastObs.notnull()) & (df.tdate < df.lastObs), "sY"] = 0
df.drop(columns = "lastObs", inplace = True)


df = df.merge(asF_sellouts, on = ["origin", "dest", "fn", "ddate"], how = "left")
print("first adjustments due to pricing")
print(df.loc[
    (df.lastObs.notnull()) & (df.tdate < df.lastObs) & (df.capF.notnull())
].shape)
# (2614, 52)
df.loc[(df.lastObs.notnull()) & (df.tdate < df.lastObs) & (df.capF.notnull()), "sF"] = 0
df.drop(columns = "lastObs", inplace = True)

cols = ["origin", "dest", "ddate", "flightNum"]
# execute ffill -- this takes a while
df["idx"] = df.groupby(cols).cumcount()+1
for c in ["saverFare", "mainFare", "firstFare", "refundYFare", "refundFFare"]:
    df["idxInd"] = df[c].notnull()*df.idx
    df["maxT"] = df.groupby(cols).idxInd.transform("max")
    df[c] = df.groupby(cols, as_index = False)[c].ffill()
    df.loc[df.idx > df.maxT, c] = np.nan

fcols = [
    "capacity1", "seats1", "capacity2",
    "seats2", "capY", "capF", "sY", "sF",
    "fn", "carrier", "refundYFare", "nonstop"
] #"fare"
df[fcols] = df.groupby(cols, as_index = False)[fcols].apply(lambda group: group.ffill())

print("nonstop shape")
print(df.loc[df.nonstop == 1].shape)
# (798808, 54)

dfC = df.loc[df.nonstop == 0].reset_index(drop = True)
dfC["fare"] = dfC[dfC.columns[dfC.columns.to_series().str.contains("Fare")]].min(axis = 1)

df = df.loc[df.capY.notnull()]
df = df.loc[~((df.capY == 76) & (df.capF == 12))].reset_index(drop = True)
df.shape
# (798566, 54)

df.loc[df.sY == 0, "mainFare"] = np.nan
df.loc[df.sY == 0, "saverFare"] = np.nan
df.loc[df.sY == 0, "refundYFare"] = np.nan
df.loc[df.sF == 0, "firstFare"] = np.nan
df.loc[df.sF == 0, "refundFFare"] = np.nan

df["fare"] = df[df.columns[df.columns.to_series().str.contains("Fare")]].min(axis = 1)

df.loc[df.fare.isnull(), "sY"] = 0
df.loc[df.fare.isnull(), "sF"] = 0

df["lf"] = (df.capY + df.capF - df.sY - df.sF) / (df.capY+df.capF)
df .loc[df.capY == 76, "lf"] = (df.capY- df.sY) / df.capY

cols = ["origin", "dest", "ddate", "flightNum"]
df["ones"] = 1
df["numObs"] = df.groupby(cols)["ones"].transform("sum")
df = df.loc[df.numObs >= 59].reset_index(drop = True)

df = df[[
    "flightNum", "ddate", "sdate", "capacity1", "seats1", "capacity2",
    "seats2", "nonstop", "origin", "dest", "saverFare", "mainFare",
    "firstFare", "refundYFare", "refundFFare", "tdate", "fare",
    "fn","carrier", "capY", "capF", "sY", "sF", "lf"
]]

df.to_parquet(f"{OUTPUT}/asdata_clean.parquet")
dfC.to_parquet(f"{OUTPUT}/asdata_cleanCon.parquet")
