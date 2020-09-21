#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from plotly import graph_objects as go
import plotly as py

from datetime import datetime
from datetime import timedelta


# In[ ]:


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


# In[ ]:


df_merged = pd.read_csv('../../../data/cleandata/Info pluviometricas/Merged Data/merged.csv', sep = ';')
df_owm = pd.read_csv('../../../data/cleandata/Info pluviometricas/Merged Data/OpenWeatherMapSantoAndre.csv')


# In[ ]:


df_merged['Data_Hora'] = pd.to_datetime(df_merged['Data_Hora'])
df_owm['Data_Hora'] = pd.to_datetime(df_owm['dt_iso'].str[:-10])
df_owm['Data_Hora'] = df_owm['Data_Hora'] - pd.Timedelta(hours = 3)
df_owm_f = df_owm[df_owm['Data_Hora'] >= datetime.strptime('2010-01-01', '%Y-%m-%d')]
df_owm_f


# In[ ]:


df_owm.isna().sum()


# In[ ]:


df_owm_f.loc[~df_owm_f['rain_1h'].isna(), ['Data_Hora', 'rain_1h', 'weather_description']]


# In[ ]:


df = df_merged.merge(df_owm_f, on = 'Data_Hora')
df.loc[~df['rain_1h'].isna(), ['Data_Hora'] + [c for c in df.columns if 'Precipitacao' in c] + ['rain_1h']].corr()


# In[ ]:


df['Data'] = df['Data_Hora'].dt.strftime('%Y-%m-%d')
df.loc[~df['rain_1h'].isna(), ['Data'] + [c for c in df.columns if 'Precipitacao' in c] + ['rain_1h']].groupby('Data').mean()


# In[ ]:


df['Data_Hora'].dt.strftime('%Y-%m-%d')


# # blablabla

# In[ ]:


df_est = pd.read_csv('../../../ordens.csv', sep=';')
df_est['Data'] = pd.to_datetime(df_est['Data'], yearfirst=True)
df_est.head()


# In[ ]:


print(df_est['Data'].min())
print(df_est['Data'].max())


# In[ ]:


#df_owm.isna().sum()
df_o = df_owm[(datetime.strptime('2019-08-30', '%Y-%m-%d') >= df_owm['Data_Hora']) & (df_owm['Data_Hora'] >= datetime.strptime('2010-01-01', '%Y-%m-%d'))].copy()
df_o = df_o.drop(columns = ['sea_level', 'grnd_level', 'rain_3h', 'snow_1h', 'snow_3h'])
df_o = df_o.fillna(0)


# In[ ]:


df_o['Data_Hora'].unique().shape


# In[ ]:


df_owm.apply(lambda x: x['Data_Hora'] + pd.Timedelta(hours = x['timezone'] / 3600), axis = 1)

