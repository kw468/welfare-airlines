"""
    This script creates the route map in 
    "The Welfare Effects of Dynamic Pricing:Evidence from Airline Markets"
--------------------------------------------------------------------------------
change log:
    v0.0.1  Mon 14 Jun 2021
-------------------------------------------------------------------------------
notes: This code must be run on Honu using
            conda activate py_ocr
    
--------------------------------------------------------------------------------
contributors:
    Kevin:
        name:       Kevin Williams
        email:      kevin.williams@yale.edu
--------------------------------------------------------------------------------
Copyright 2021 Yale University
"""

# -------------------------------------------------------------------------------
# IMPORT REQUIRED PACKAGES
# -------------------------------------------------------------------------------

import pandas as pd
import plotly.graph_objects as go

# -------------------------------------------------------------------------------
# IMPORT ROUTES AND SELECT ODS
# -------------------------------------------------------------------------------

# this file contains the ODs studied in the paper
dfR                     = pd.read_csv("/mnt/data0/bts_airline/airline_jmp_routes.csv", sep="\t", header=None)
dfR[0]                  = dfR[0].str.strip()
dfR[1]                  = dfR[1].str.strip()
dfR[0]                  = dfR[0].astype("str")
dfR[1]                  = dfR[1].astype("str")
dfR                     .rename(columns = {0 : "origin", 1 : "dest", 2 : "year", 3 : "comp"}, inplace=True )

# this file contains the lats and lons of airports
airports                = pd.read_csv("/mnt/data0/bts_airline/resources/airports.dat", header=None)
airports                .rename(columns = {4 : "origin", 6 : "lat", 7 : "lon"}, inplace=True )

plot                    = dfR.merge(airports[["origin", "lat", "lon"]], on = "origin", how="left")
airports                .rename(columns = {"origin" : "dest"}, inplace=True )
plot                    = plot.merge(airports[["dest", "lat", "lon"]], on = "dest", how="left")


df_airports             = plot[["origin", "lat_x", "lon_x"]].drop_duplicates()


df_flight_paths         = plot.copy()
df_flight_paths["od"]   = df_flight_paths["origin"] + df_flight_paths["dest"]
df_flight_paths["do"]   = df_flight_paths["dest"] + df_flight_paths["origin"]
df_flight_paths["od"]   = df_flight_paths[["od", "do"]].min(axis=1) # create OD pair
df_flight_paths         = df_flight_paths.drop_duplicates("od")
df_flight_paths         = df_flight_paths.reset_index(drop=True)

df_flight_paths1        = df_flight_paths.loc[df_flight_paths.comp == 0].reset_index(drop=True)
df_flight_paths2        = df_flight_paths.loc[df_flight_paths.comp == 1].reset_index(drop=True)


# -------------------------------------------------------------------------------
# CREATE THE FIGURE
# -------------------------------------------------------------------------------


fig                     = go.Figure()

fig.add_trace(go.Scattergeo(
    locationmode = 'USA-states',
    lon = plot['lon_x'],
    lat = plot['lat_x'],
    mode = 'markers',
    marker = dict(
        size = 8,
        color = 'rgb(0, 0, 0)',
        line = dict(
            width = 8,
            color = 'rgba(100, 100, 100, 0)'
        )
    )))

flight_paths = []
for i in range(len(df_flight_paths1)):
    fig.add_trace(
        go.Scattergeo(
            locationmode = 'USA-states',
            lon = [df_flight_paths1['lon_x'][i], df_flight_paths1['lon_y'][i]],
            lat = [df_flight_paths1['lat_x'][i], df_flight_paths1['lat_y'][i]],
            mode = 'lines',
            line = dict(width = 1,color = 'rgb(70, 120, 186)'),
        )
    )

flight_paths = []
for i in range(len(df_flight_paths2)):
    fig.add_trace(
        go.Scattergeo(
            locationmode = 'USA-states',
            lon = [df_flight_paths2['lon_x'][i], df_flight_paths2['lon_y'][i]],
            lat = [df_flight_paths2['lat_x'][i], df_flight_paths2['lat_y'][i]],
            mode = 'lines',
            line = dict(width = .5,color = 'rgb(167, 164, 151)'),
        )
    )


fig.update_layout(
    geo = dict(
        scope='usa',
        projection=go.layout.geo.Projection(type = 'albers usa'),
        showlakes=True, # lakes
        lakecolor='rgb(255, 255, 255)'),
        showlegend=False,
)


fig.write_image("/mnt/data0/bts_airline/resources/routemap.pdf")
fig.close()
