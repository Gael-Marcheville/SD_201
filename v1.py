import pandas as pd 
import datetime as dt
# import numpy as np
# from scipy import stats
# import matplotlib.pyplot as plt

df_2015 = pd.read_csv("./dataset/2015.csv")
df_2016 = pd.read_csv("./dataset/2016.csv")
df_2017 = pd.read_csv("./dataset/2017.csv")
df_2018 = pd.read_csv("./dataset/2018.csv")
df_2019 = pd.read_csv("./dataset/2019.csv")
# df = df.assign(year = 2015)
# df_temp = pd.read_csv("./dataset/2016.csv")
# df_temp = df.assign(year = 2016)
# df = pd.concat([df, df_temp])
df_country = pd.DataFrame({'Country' : df_2015['Country']})
print(df_country)
df_country = pd.merge(df_country, pd.DataFrame({'Country' : df_2016['Country']}), how='inner', on=['Country'])
print(df_country)
df_country = pd.merge(df_country, pd.DataFrame({'Country' : df_2017['Country']}), how='inner', on=['Country'])
print(df_country)
df_country = pd.merge(df_country, pd.DataFrame({'Country' : df_2018['Country or region']}), how='inner', on=['Country'])
print(df_country)
df_country = pd.merge(df_country, pd.DataFrame({'Country' : df_2019['Country or region']}), how='inner', on=['Country'])
print(df_country)
#df_country = df_country.join(pd.read_csv("./dataset/2017.csv")['Country'],on = 'Country',how='inner')
# df_temp = pd.read_csv("./dataset/2018.csv")
# df_temp = df.assign(year = 2018)
# df = pd.concat([df, df_temp])
# df_temp = pd.read_csv("./dataset/2019.csv")
# df_temp = df.assign(year = 2019)
# df = pd.concat([df, df_temp])
