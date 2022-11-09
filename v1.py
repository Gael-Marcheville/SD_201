import pandas as pd 
import datetime as dt
# import numpy as np
# from scipy import stats
# import matplotlib.pyplot as plt

### import data
df_2015 = pd.read_csv("./dataset/2015.csv")
df_2016 = pd.read_csv("./dataset/2016.csv")
df_2017 = pd.read_csv("./dataset/2017.csv")
df_2018 = pd.read_csv("./dataset/2018.csv")
df_2019 = pd.read_csv("./dataset/2019.csv")
df_gdp = pd.read_csv("./dataset/gdp_year.csv",error_bad_lines=False)

### clean data
dfc_2015 = df_2015.rename(columns={"Country": "country","Happiness Rank" : "happiness_rank","Happiness Score" : "happiness_score","Economy (GDP per Capita)" : "gdp_per_capita","Freedom" : "freedom_choices","Generosity" : "generosity","Health (Life Expectancy)" : "health_life_expectancy","Family" : "social_support"})
dfc_2015 = dfc_2015.drop(['Region'], axis=1)
dfc_2016 = df_2016.rename(columns={"Country": "country","Happiness Rank" : "happiness_rank","Happiness Score" : "happiness_score","Economy (GDP per Capita)" : "gdp_per_capita","Freedom" : "freedom_choices","Generosity" : "generosity","Health (Life Expectancy)" : "health_life_expectancy","Family" : "social_support"})
dfc_2016 = dfc_2016.drop(['Region'], axis=1)
dfc_2017 = df_2017.rename(columns={"Country": "country","Happiness.Rank" : "happiness_rank","Happiness.Score" : "happiness_score","Economy..GDP.per.Capita." : "gdp_per_capita","Freedom" : "freedom_choices","Generosity" : "generosity","Health..Life.Expectancy." : "health_life_expectancy","Family" : "social_support"})
dfc_2018 = df_2018.rename(columns={"Country or region": "country","Overall rank" : "happiness_rank","Score" : "happiness_score","GDP per capita" : "gdp_per_capita","Freedom to make life choices" : "freedom_choices","Generosity" : "generosity","Healthy life expectancy" : "health_life_expectancy","Social support" : "social_support"})
dfc_2019 = df_2019.rename(columns={"Country or region": "country","Overall rank" : "happiness_rank","Happiness Score" : "happiness_score","GDP per capita" : "gdp_per_capita","Freedom to make life choices" : "freedom_choices","Generosity" : "generosity","Healthy life expectancy" : "health_life_expectancy","Social support" : "social_support"})

dfc_country = pd.DataFrame({'country' : dfc_2015['country']})
dfc_country = pd.merge(dfc_country, pd.DataFrame({'country' : dfc_2016['country']}), how='inner', on=['country'])
dfc_country = pd.merge(dfc_country, pd.DataFrame({'country' : dfc_2017['country']}), how='inner', on=['country'])
dfc_country = pd.merge(dfc_country, pd.DataFrame({'country' : dfc_2018['country']}), how='inner', on=['country'])
dfc_country = pd.merge(dfc_country, pd.DataFrame({'country' : dfc_2019['country']}), how='inner', on=['country'])

dfc_2015 = pd.merge(dfc_country, dfc_2015, how='inner', on=['country'])
print(dfc_2015)

dfc_all = dfc_2015.assign(year = 2015)

# df_temp = pd.read_csv("./dataset/2016.csv")
# df_temp = df.assign(year = 2016)
# df = pd.concat([df, df_temp])
#df_country = df_country.join(pd.read_csv("./dataset/2017.csv")['Country'],on = 'Country',how='inner')
# df_temp = pd.read_csv("./dataset/2018.csv")
# df_temp = df.assign(year = 2018)
# df = pd.concat([df, df_temp])
# df_temp = pd.read_csv("./dataset/2019.csv")
# df_temp = df.assign(year = 2019)
# df = pd.concat([df, df_temp])
