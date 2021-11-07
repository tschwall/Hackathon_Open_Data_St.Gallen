# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 14:25:26 2021

@author: marius.baumann
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

df = pd.read_csv("verkehrszahlung-bundesstrassen-kanton-stgallen-astra.csv", delimiter=";")

def splitStringDate(string):
    string = str(string)
    return pd.to_datetime(f"{string[0:4]}.{string[4:6]}.{string[6:8]}.{string[8:10]}")  

def splitStringDate4(string):
    string = str(string)
    return pd.to_datetime(f"{string[0:4]}.{string[4:6]}.{string[6:8]}")  

wetter_temp_df = pd.read_csv('Wetter_Temperatur.csv',sep=";")

stg_temp_df =  wetter_temp_df[wetter_temp_df['stn']=='STG'].copy(deep=True)
stg_temp_df["timer"] = stg_temp_df["time"].apply(splitStringDate)
stg_temp_df["date"] = stg_temp_df["time"].apply(splitStringDate4)

stg_temp_df =  stg_temp_df[(stg_temp_df['timer'].dt.hour >= 8) & (stg_temp_df['timer'].dt.hour <=18)]

stg_temp_df =  stg_temp_df[wetter_temp_df['tre200h0']!='-'].copy(deep=True)
stg_temp_df["tre200h0"] = pd.to_numeric(stg_temp_df["tre200h0"])

temp_mean = stg_temp_df.groupby(['date'],as_index=False).agg({'tre200h0': 'mean'})



# wetter_df = pd.read_csv("Wetter_Daten_Stündlich.csv",sep=";")
# wetter_df = wetter_df[wetter_df["stn"]=="STG"].copy(deep=True)
# wetter_df["date"] = wetter_df["time"].apply(splitStringDate)
# mapping = {"-":0}
# wetter_df = wetter_df.replace({"rre024i0":mapping})
# wetter_df["rre024i0"] = pd.to_numeric(wetter_df["rre024i0"])
# wetter_df = wetter_df.sort_values("time")

# for i in df.columns:
#     print(i)
    
# df_filtered = df[df["Zählstellen-Bezeichnung"].str.contains("ST. GALLEN, HARZBUECHEL")].sort_values("JJMMTT")

# df_filtered.to_csv("filtered.csv", sep=";", encoding="utf-8")

# df["count"] = 1
# df_count = df.groupby("Zählstellen-Bezeichnung").count()["count"]
# print(df_count)

# grouped_df = df[["JJMMTT", "Zählstellen-Bezeichnung", "NAME_D", "Richtung 1", "Richtung 2", "Total Richtung 1", "Total Richtung 2", "Total Beide Richtungen"]]
# grouped_df = grouped_df.groupby(["Zählstellen-Bezeichnung", "NAME_D"])
# for name in grouped_df.groups:
#     print(name)
#     group = grouped_df.get_group(name).sort_values("JJMMTT")
#     # group.to_csv(f"data/{name[0][0:4]}.csv", sep=";")
#     selected = group[["JJMMTT", "Total Beide Richtungen"]]
#     print(selected)

# example = grouped_df.get_group(list(grouped_df.groups.keys())[0]).sort_values("JJMMTT")
# example.to_csv(f"data/example.csv", sep=";")

# example = example[["JJMMTT", "Total Beide Richtungen"]]
# plt.hist(example["Total Beide Richtungen"], example["JJMMTT"])

vehicleType = "Motorrad"
df_1 = df.loc[(df["Richtung 1"] == "ST. GALLEN") & (df["NAME_D"] == vehicleType)]
df_1["JJMMTT"] = pd.to_datetime(df_1["JJMMTT"])

# COLUMNS = [f"R1H{i:02}" for i in range(24)]
# col_name_mapping = {name:hour for hour, name in enumerate(COLUMNS)}
# df_temp = df_1[COLUMNS].rename(columns = col_name_mapping).stack().reset_index().drop('level_0', axis=1)
# df_temp.rename(columns={'level_1': 'time', 0: 'count'}, inplace=True)
df_temp = df_1[["JJMMTT", "Total Richtung 1"]].copy(deep=True)
df_temp = df_temp.rename(columns={"JJMMTT":"date", "Total Richtung 1":"count"})
df_temp = df_temp.sort_values("date")
# df_temp['time'] = df_temp['time'].astype('timedelta64[ns]')

# df_time = df_1["JJMMTT"]
# df_time = df_time.repeat(24).reset_index(drop=True)

# df_temp['time'] = df_temp['time'] + df_time
# print(df_temp.head)
wetter_df = temp_mean[(temp_mean["date"] >= pd.to_datetime("2019.01.01")) & (temp_mean["date"] < pd.to_datetime("2021.01.01")) ]

total_df = pd.merge(df_temp, wetter_df)

# reg = LinearRegression().fit(np.column_stack((wetter_df["rre024i0"],np.array(wetter_df["rre024i0"])**2)),df_temp["count"])
print(total_df.head())
mod = smf.ols(formula="count ~ tre200h0 + np.power(tre200h0, 2)", data=total_df)
res = mod.fit()
print(res.summary())
# total_df.plot(x="tre200h0", y="count",kind="scatter")
# df_richtSG2 = df.groupby(["Richtung 2"]).get_group("ST. GALLEN").sort_values("JJMMTT"))

fig, ax = plt.subplots()
ax.set_xlabel("date")
plt.xticks((pd.to_datetime("2019.01.01"), pd.to_datetime("2020.01.01"), pd.to_datetime("2021.01.01")))
ax.plot(df_temp["date"], df_temp["count"], label=f"Motorcycle driving to St. Gallen")
ax.legend(loc="best")
# df_temp.plot(x="date", y=)


fig, ax = plt.subplots()
T = np.linspace(-5, 30, 1000)
off, lin, lg = res.params
y = off + T*lin + T**2*lg
ax.set_xlabel("Temperature in C°")
ax.plot(T, y, label="Model prediction")
ax.plot(total_df["tre200h0"], total_df["count"], ".", label="Motorcycle / temperature")
ax.legend(loc="best")


# COLUMS = ["JJMMTT", "Wochentag", "Total Richtung 1", "Total Richtung 2", "Total Beide Richtungen"]
# df_richtSGSum = df_richtSG[COLUMS]
# df_richtSGSum["TOT"] = df_richtSG.groupby(["JJMMTT"]).sum()["Total Beide Richtungen"]
# df_richtSGSum.to_csv("data/df_richtSGSum.csv", sep=";")

# df_mr = df_richtSG.groupby(["NAME_D"]).get_group("Motorrad")
# df_mr.to_csv("data/richtSG_motorrad.csv", sep=";")

# df_pkw = df_richtSG.groupby(["NAME_D"]).get_group("Personenwagen")
# df_pkw.to_csv("data/richtSG_pkw.csv", sep=";")

# COLUMS = ["JJMMTT", "Wochentag", "Total Richtung 1", "Total Richtung 2", "Total Beide Richtungen"]
# df_temp = df_pkw.groupby("JJMMTT").sum()
# print(df_temp)
