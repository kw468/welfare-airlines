"""
    This script reports the counterfactual results in
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

#srun --nodelist=c3 --cpus-per-task=128 --pty bash -i

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import os
from plot_setup import *
from estim_markets import *

CSFONT = {"fontname": FONT, "fontsize": FONT_SIZE - 2}

# --------------------------------------------------------------------------------
# Set path for data and logs
# --------------------------------------------------------------------------------
INTPUT = "../../data/"
OUTPUT = "../../output/"

os.chdir(INTPUT)

df = pd.DataFrame()
for m in mkts:
    print(m)
    df = df.append(pd.read_parquet(INTPUT  + "estimation/" + m + "/" + m + "_counterfactuals.parquet"))

df["difD"] = df.groupby(["market", "fl"]).priceD.diff()
df["difS"] = df.groupby(["market", "fl"]).priceS.diff()
df["indD"] = (df.difD != 0)
df["indS"] = (df.difS != 0)

def cf_table(df):
    a = df.loc[df.t != 60][["priceD", "priceA", "priceU"]].mean().values
    b = df.loc[df.t == 60][["LF_D", "LF_A", "LF_U"]].mean().values
    c1 = 100.0 * len(df[(df.t == 60) & (df.capD == 0)]) / \
        (df.market.nunique() * df.fl.nunique())
    c2 = 100.0 * len(df[(df.t == 60) & (df.capA == 0)]) / \
        (df.market.nunique() * df.fl.nunique())
    c3 = 100.0 * len(df[(df.t == 60) & (df.capU == 0)]) / \
        (df.market.nunique() * df.fl.nunique())
    c = [c1, c2, c3]
    d1 = (df.groupby(["market", "fl"])["revD"].sum()).mean()
    d2 = (df.groupby(["market", "fl"])["revA"].sum()).mean()
    d3 = (df.groupby(["market", "fl"])["revU"].sum()).mean()
    d = [d1, d2, d3]
    e = [
        100 * (df.groupby(["market", "fl"])["CS_D_L"].sum()).mean(),
        100 * (df.groupby(["market", "fl"])["CS_A_L"].sum()).mean(),
        100 * (df.groupby(["market", "fl"])["CS_U_L"].sum()).mean()
    ]
    f = [
        100 * (df.groupby(["market", "fl"])["CS_D_B"].sum()).mean(),
        100 * (df.groupby(["market", "fl"])["CS_A_B"].sum()).mean(),
        100 * (df.groupby(["market", "fl"])["CS_U_B"].sum()).mean()
    ]
    g = [
        100 * (df.groupby(["market", "fl"])["CS_D_ALL"].sum()).mean(),
        100 * (df.groupby(["market", "fl"])["CS_A_ALL"].sum()).mean(),
        100 * (df.groupby(["market", "fl"])["CS_U_ALL"].sum()).mean()
    ]
    h1 = (
        100 * df.groupby(["market", "fl"]).CS_D_ALL.sum() + \
            df.groupby(["market", "fl"]).revD.sum()
    ).mean()
    h2 = (
        100 * df.groupby(["market", "fl"]).CS_A_ALL.sum() + \
            df.groupby(["market", "fl"]).revA.sum()
    ).mean()
    h3 = (
        100 * df.groupby(["market", "fl"]).CS_U_ALL.sum() + \
            df.groupby(["market", "fl"]).revU.sum()
    ).mean()
    h=[h1, h2, h3]
    F = open(OUTPUT + "cf_table.tex","w")
    F.write("   &   \\underline{Fare}    &   \\underline{Load Factor} &   \\underline{Sell Outs}   &   \\underline{Revenue} &   \\underline{$CS_L$}    &   \\underline{$CS_B$}  &   \\underline{$CS$}    &   \\underline{Welfare} \\\\\n")
    F.write( "Dynamic&"+ "&".join([str(round(z, 1)) for z in [a[0], b[0], c[0], d[0], e[0], f[0], g[0], h[0]]]) + "\\\\\n")
    F.write( "IPD&"+ "&".join([str(round(z, 1)) for z in [a[1], b[1], c[1], d[1], e[1], f[1], g[1], h[1]]]) + "\\\\\n")
    F.write( "Uniform&"+ "&".join([str(round(z, 1)) for z in [a[2], b[2], c[2], d[2], e[2], f[2], g[2], h[2]]]) + "\\\\\n")
    F.write("\\midrule\n")
    F.close()

def q25(x):
    return x.quantile(.25)

def q75(x):
    return x.quantile(.75)

def makePricePlot(df):
    stat1 = {"priceD": ["mean", q25, q75]}
    stat2 = {"priceA": ["mean", q25, q75]}
    stat3 = {"priceU": ["mean", q25, q75]}
    df1 = df.loc[df.t != 60].groupby("t").agg(stat1)
    df2 = df.loc[df.t != 60].groupby("t").agg(stat2)
    df3 = df.loc[df.t != 60].groupby("t").agg(stat3)
    df1.columns = ["avg", "q25", "q75"]
    df2.columns = ["avg", "q25", "q75"]
    df3.columns = ["avg", "q25", "q75"]
    fig = plt.figure(figsize = FIG_SIZE)
    plt.plot(
        df1.avg,
        color=PALETTE[4],
        linewidth = LINE_WIDTH,
        label = "Dynamic Pricing"
    )
    plt.plot(
        df2.avg,
        color = PALETTE[0],
        linewidth = LINE_WIDTH,
        label = "IPD",
        ls = "-."
    )
    plt.plot(
        df3.avg,
        color = PALETTE[2],
        linewidth = LINE_WIDTH,
        label = "Uniform Pricing",
        ls = ":"
    )
    # plt.plot(df1.q25,color=PALETTE[4], linestyle="--",linewidth = 1,label="_nolegend_")
    # plt.plot(df2.q25,color=PALETTE[0], linestyle="--",linewidth = 1,label="_nolegend_")
    # plt.plot(df3.q25,color=PALETTE[2], linestyle="--",linewidth = 1,label="_nolegend_")
    # plt.plot(df2.q75,color=PALETTE[0], linestyle="--",linewidth = 1, label="_nolegend_")
    # plt.plot(df1.q75,color=PALETTE[4], linestyle="--",linewidth = 1,label="_nolegend_")
    # plt.plot(df3.q75,color=PALETTE[2], linestyle="--",linewidth = 1,label="_nolegend_")
    plt.ylabel("Mean Price", **CSFONT)
    plt.xlabel("Booking Horizon", **CSFONT)
    plt.setp(plt.legend().texts, family = FONT, fontsize = FIG_SIZE - 2)
    plt.yticks(fontname = FONT, fontsize = FIG_SIZE - 2)
    plt.xticks(fontname = FONT, fontsize = FIG_SIZE - 2)
    plt.savefig(
        OUTPUT + "cf_mean_price_plot.pdf",
        bbox_inches = "tight",
        format= FIG_FORMAT,
        dpi = DPI
    )
    plt.close()

def makeLFPlot(df):
    df1 = df.groupby("t").LF_D.mean()
    df2 = df.groupby("t").LF_A.mean()
    df3 = df.groupby("t").LF_U.mean()
    df2 = 100 * df2.values / df1.values
    df3 = 100 * df3.values / df1.values
    fig = plt.figure(figsize = FIG_SIZE)
    #plt.plot(df1,color=PALETTE[4],linewidth = LINE_WIDTH,label="Dynamic Pricing")
    plt.plot(
        df2,
        color = PALETTE[0],
        linewidth = LINE_WIDTH,
        label = "IPD",
        ls = "-."
    )
    plt.plot(
        df3,
        color = PALETTE[2],
        linewidth = LINE_WIDTH,
        label = "Uniform Pricing",
        ls = ":"
    )
    plt.ylabel("Load Factor, Relative to DP", **CSFONT)
    plt.xlabel("Booking Horizon", **CSFONT)
    plt.setp(plt.legend().texts, family = FONT, fontsize = FONT_SIZE - 2)
    plt.yticks(fontname = FONT, fontsize = FONT_SIZE - 2)
    plt.xticks(fontname = FONT, fontsize = FONT_SIZE - 2)
    plt.axhline(100, color = PALETTE[4], lw = 2, ls = "-")
    plt.savefig(
        OUTPUT + "cf_mean_lf.pdf",
        bbox_inches = "tight",
        format = FIG_FORMAT,
        dpi = DPI
    )
    plt.close()

def makeSelloutPlot(df):
    df["selloutD"] = df.LF_D == 100
    df["selloutA"] = df.LF_A == 100
    df["selloutU"] = df.LF_U == 100
    df1 = 100 * df.groupby("t").selloutD.mean()
    df2 = 100 * df.groupby("t").selloutA.mean()
    df3 = 100 * df.groupby("t").selloutU.mean()
    fig = plt.figure(figsize = FIG_SIZE)
    plt.plot(
        df1,
        color = PALETTE[4],
        linewidth = LINE_WIDTH,
        label = "Dynamic Pricing"
    )
    plt.plot(
        df2,
        color = PALETTE[0],
        linewidth = LINE_WIDTH,
        label = "IPD",
        ls="-."
    )
    plt.plot(
        df3,
        color = PALETTE[2],
        linewidth = LINE_WIDTH,
        label = "Uniform Pricing",
        ls = ":"
    )
    plt.ylabel("CDF of Sellouts", **CSFONT)
    plt.xlabel("Booking Horizon", **CSFONT)
    plt.setp(plt.legend().texts, family = FONT, fontsize = FONT_SIZE - 2)
    plt.yticks(fontname = FONT, fontsize = FONT_SIZE - 2)
    plt.xticks(fontname = FONT, fontsize = FONT_SIZE - 2)
    plt.savefig(
        OUTPUT + "cf_sellouts.pdf",
        bbox_inches = "tight",
        format = FIG_FORMAT,
        dpi = DPI
    )
    plt.close()


def makeWPlot(df):
    df1 = df.loc[df.t != 60].groupby("t")[["revD", "revU", "revA"]].sum().reset_index()
    df1["revD"] = df1.revD.cumsum()
    df1["revA"] = df1.revA.cumsum()
    df1["revU"] = df1.revU.cumsum()
    df2 = 100 * df1.revA.values / df1.revD.values
    df3 = 100 * df1.revU.values / df1.revD.values
    fig = plt.figure(figsize = FIG_SIZE)
    #plt.plot(df1,color=PALETTE[4],linewidth = LINE_WIDTH,label="Dynamic Pricing")
    plt.plot(
        df2,
        color = PALETTE[0],
        linewidth = LINE_WIDTH,
        label = "IPD - Revenue"
    )
    plt.plot(
        df3,
        color = PALETTE[2],
        linewidth = LINE_WIDTH,
        label="Uniform Pricing - Revenue"
    )
    # now do CS
    df1 = df.loc[df.t != 60].groupby("t")[["CS_D_ALL", "CS_U_ALL", "CS_A_ALL"]].sum().reset_index()
    df1["CS_D_ALL"] = df1.CS_D_ALL.cumsum()
    df1["CS_A_ALL"] = df1.CS_A_ALL.cumsum()
    df1["CS_U_ALL"] = df1.CS_U_ALL.cumsum()
    df2 = 100 * df1.CS_A_ALL.values / df1.CS_D_ALL.values
    df3 = 100 * df1.CS_U_ALL.values / df1.CS_D_ALL.values
    plt.plot(
        df2,
        color = PALETTE[0],
        linewidth = LINE_WIDTH,
        label = "IPD - CS",
        ls = ":"
    )
    plt.plot(
        df3,
        color = PALETTE[2],
        linewidth = LINE_WIDTH,
        label = "Uniform Pricing - CS",
        ls = ":"
    )
    plt.ylabel("Cumulative %, Relative to DP", **CSFONT)
    plt.xlabel("Booking Horizon", **CSFONT)
    plt.setp(plt.legend().texts, family = FONT, fontsize = FONT_SIZE - 2)
    plt.yticks(fontname = FONT, fontsize = FONT_SIZE - 2)
    plt.xticks(fontname = FONT, fontsize = FONT_SIZE - 2)
    plt.axhline(100, color = PALETTE[4], lw = 2, ls = "-")
    plt.savefig(
        OUTPUT + "cf_welfare_compare.pdf",
        bbox_inches = "tight",
        format = FIG_FORMAT,
        dpi = DPI
    )
    plt.close()

def makeStaticPPlot(df):
    stat1 = {"priceD": ["mean", q25, q75]}
    stat3 = {"priceS": ["mean", q25, q75]}
    df1 = df.loc[df.t != 60].groupby("t").agg(stat1)
    df3 = df.loc[df.t != 60].groupby("t").agg(stat3)
    df1.columns = ["avg", "q25", "q75"]
    df3.columns = ["avg", "q25", "q75"]
    csfont = {"fontname": FONT, "fontsize": FONT_SIZE}
    fig = plt.figure(figsize = FIG_SIZE)
    plt.plot(
        df1.avg,
        color = PALETTE[4],
        linewidth = LINE_WIDTH,
        label="Dynamic Pricing"
    )
    plt.plot(
        df3.avg,
        color = PALETTE[0],
        linewidth = LINE_WIDTH,
        label = "Static Pricing",
        ls="-."
    )
    # plt.plot(df1.q25,color=PALETTE[4], linestyle="--",linewidth = 1,label="_nolegend_")
    # plt.plot(df3.q25,color=PALETTE[0], linestyle="--",linewidth = 1,label="_nolegend_")
    # plt.plot(df1.q75,color=PALETTE[4], linestyle="--",linewidth = 1,label="_nolegend_")
    # plt.plot(df3.q75,color=PALETTE[0], linestyle="--",linewidth = 1,label="_nolegend_")
    plt.setp(plt.legend().texts, family = FONT, fontsize = FONT_SIZE - 2)
    plt.ylabel("Mean Fare", **csfont)
    plt.xlabel("Booking Horizon", **csfont)
    plt.yticks(fontname = FONT, fontsize = FONT_SIZE)
    plt.xticks(fontname = FONT, fontsize = FONT_SIZE)
    plt.savefig(
        OUTPUT + "cf_dp_v_static_p.pdf",
        bbox_inches = "tight",
        format = FIG_FORMAT,
        dpi = DPI
    )

def makeRoutePlot(df1):
    fig = plt.figure(figsize = FIG_SIZE)
    plt.scatter(
        df1.fracGains,
        df1.fracU_w,
        color=PALETTE[4],
        s = 60,
        marker = "o",
        label = "Market"
    )
    plt.axhline(1, color = PALETTE[0], linewidth = LINE_WIDTH - 1)
    plt.ylabel("Welfare of DP / Welfare of Uni.", **CSFONT)
    plt.xlabel("(Rev IPD - Rev Uni.) / (Rev DP - Rev Uni.)", **CSFONT)
    plt.setp(plt.legend().texts, family = FONT, fontsize = FONT_SIZE - 2)
    plt.yticks(fontname = FONT, fontsize = FONT_SIZE - 2)
    plt.xticks(fontname = FONT, fontsize = FONT_SIZE - 2)
    plt.savefig(
        OUTPUT + "cf_route_plot.pdf",
        bbox_inches = "tight",
        format = FIG_FORMAT,
        dpi = DPI
    )
    plt.close()

def makePriceDecomp(df):
    df3 = df.loc[df.difD.notnull()].groupby("t").indD.mean()
    df4 = df.loc[df.difS.notnull()].groupby("t").indS.mean()
    df3 = df3[:59]
    df4 = df4[:59]
    fig = plt.figure(figsize = FIG_SIZE)
    plt.plot(
        100 * df3,
        color = PALETTE[4],
        linewidth = LINE_WIDTH,
        label = "Dynamic Pricing"
    )
    plt.plot(
        100 * df4,
        color=PALETTE[0],
        linewidth = LINE_WIDTH,
        label = "Static Pricing",
        ls = "-."
    )
    plt.ylabel("% of Flights with Price Changes", **CSFONT)
    plt.xlabel("Booking Horizon", **CSFONT)
    plt.setp(plt.legend().texts, family = FONT, fontsize = FONT_SIZE - 2)
    plt.yticks(fontname = FONT, fontsize = FONT_SIZE - 2)
    plt.xticks(fontname = FONT, fontsize = FONT_SIZE - 2)
    plt.ylim((0, 25))
    plt.savefig(
        OUTPUT + "cf_price_decomp.pdf",
        bbox_inches = "tight",
        format = FIG_FORMAT,
        dpi = DPI
    )

cf_table(df)
makePricePlot(df)
makeLFPlot(df)
makeSelloutPlot(df)
makeWPlot(df)
makeStaticPPlot(df)
makePriceDecomp(df)


df1 = df.groupby(["market"])[[
    "revU", "revD", "revA", "CS_D_ALL", "CS_A_ALL", "CS_U_ALL"
]].sum().reset_index()
df1["fracU_rev"] = df1.revU / df1.revD
df1["fracA_rev"] = df1.revA / df1.revD
df1["fracU_cs"] = df1.CS_U_ALL / df1.CS_D_ALL
df1["fracA_cs"] = df1.CS_A_ALL / df1.CS_D_ALL
df1["fracA_w"] = (100 * df1.CS_D_ALL + df1.revD) / (100 * df1.CS_A_ALL + df1.revA)
df1["fracU_w"] = (100 * df1.CS_D_ALL + df1.revD) / (100 * df1.CS_U_ALL + df1.revU)

df1["fracGains"] = (df1.revA - df1.revU) / (df1.revD - df1.revU)

df1["colors"] = ["#FF6700" if x < 1 else "#3A6EA5" for x in df1["fracU_w"]]

df1 = df1.sort_values(by = "fracU_w").reset_index(drop = True)
df["so"] = df.LF_D == 100
df2 = df.loc[df.t == 60].groupby("market").so.mean().reset_index()
df1 = df1.merge(df2, on = "market")

makeRoutePlot(df1)

#then
#scp -p /gpfs/home/kw468/airlines_jmp/output/*.pdf kw468@72.89.51.250:/home/kw468/Projects/airlines_jmp/output/
#scp -p /gpfs/home/kw468/airlines_jmp/output/*.tex kw468@72.89.51.250:/home/kw468/Projects/airlines_jmp/output/

CSFONT1 = {"fontname":FONT, "fontsize":12}

# Draw plot
fig, ax = plt.subplots(figsize=(10, 10), dpi = DPI)
plt.hlines(
    y = df1.index,
    xmin = 1,
    xmax = df1["fracU_w"],
    color = df1["colors"],
    lw = 4
)
for x, y, tex in zip(df1["fracU_w"], df1.index, df1["fracU_w"]):
    t = plt.text(
        x, y, str(round(100 * tex, 1)) + "%",
        horizontalalignment = "right" if x < 1 else "left",
        verticalalignment = "center",
        fontdict = {
            "color":PALETTE[2] if x < 1 else PALETTE[2],
            "size":12
        },
        **CSFONT1
    )

# Decorations
plt.yticks(df1.index, df1["market"], fontsize = 10,fontname = FONT)
plt.grid(linestyle = "--", alpha = 0.5)
minX = np.round(df1.fracU_w.min(), 2) - .03
maxX = np.round(df1.fracU_w.max(), 2) + .02
plt.xlim(minX, maxX)
plt.xlabel("Percentange of Flights that Sellout & Welfare of Dynamic Pricing Relative to Uniform Pricing", **CSFONT1)
plt.ylabel("Route", **CSFONT1)

plt.xlim((.92, 1.09))

plt.hlines(
    y = df1.index,
    xmin = .92,
    xmax = (.92 + df1["so"]*.02),
    color = df1["colors"],
    lw = 12
)
for x, y, tex in zip((.92 + df1["so"] * .02), df1.index, df1["so"]):
    t = plt.text(
        x,
        y,
        str(round(100 * tex, 1)) + "%",
        horizontalalignment = "right" if x < 0 else "left",
        verticalalignment = "center",
        fontdict = {"color": PALETTE[2] if x < 0 else PALETTE[2], "size":12},
        **CSFONT1
    )

plt.axvline(x = .94, lw = 2, color = "black")
fig.canvas.draw()
labels = [item.get_text() for item in ax.get_xticklabels()]
labels[0] = "0"
labels[1] = "100"
ax.set_xticklabels(labels)

plt.savefig(
    OUTPUT + "cf_route_welfare.pdf",
    bbox_inches = "tight",
    format = FIG_FORMAT,
    dpi = DPI
)
plt.close()

df.priceD.mean() / df.priceU.mean()
df.salesD.sum() / df.salesU.sum()
df.loc[(df.LF_U == 100)].arrivals_B.sum() /  df.arrivals_B.sum()
df.loc[(df.LF_D == 100)].arrivals_B.sum() /  df.arrivals_B.sum()
df.salesD.sum() / df.salesU.sum()
1.0250837184018144

(df.revA.sum() - df.revU.sum()) / (df.revD.sum() - df.revU.sum())
0.6648010893885555

df.revD.sum() / df.revU.sum()
df.priceD.mean() / df.priceS.mean()
df.CS_D_L.sum() / df.CS_U_L.sum()
1 - df.CS_D_ALL.sum() / df.CS_U_ALL.sum()
df.loc[(df.t != 60) & (df.difD.notnull())].indD.sum() / \
    df.loc[(df.t != 60) & (df.difS.notnull())].indS.sum()

df.salesD.sum() / df.salesS.sum()
df.salesD_B.sum() / df.salesS_B.sum()
1 - df.salesD_L.sum() / df.salesS_L.sum()

(df.salesD_B.sum() - df.salesU_B.sum()) / df.salesD_B.sum()
(df.salesD_L.sum() - df.salesU_L.sum()) / df.salesD_L.sum()
