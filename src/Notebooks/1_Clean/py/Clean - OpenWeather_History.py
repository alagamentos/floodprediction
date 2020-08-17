#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
df = pd.read_csv('/home/felipe/Documents/TCC/data/rawdata/openweather/history_bulk.csv')


# In[2]:


df.insert(0, 'Data_Hora', np.nan)

df['Data_Hora'] = pd.to_datetime(df['dt_iso'].str[:-10])
df['Data_Hora'] = df.apply(lambda x: x['Data_Hora'] +
                                   pd.Timedelta(hours = x['timezone'] / 3600), axis = 1)

df = df[df['Data_Hora'] > '2010-01-01']


# In[4]:


drop_cols = ['dt', 'dt_iso', 'timezone', 'city_name',
            'lat', 'lon', 'weather_main', 'weather_id', 'weather_icon',
            'snow_1h','snow_3h','rain_3h','sea_level','grnd_level' ]

rename_cols = {'pressure': 'PressaoAtmosferica',
         'humidity': 'UmidadeRelativa',
         'wind_speed': 'VelocidadeDoVento',
         'wind_deg': 'DirecaoDoVento',
         'rain_1h': 'Precipitacao',
         'feels_like': 'SensacaoTermica',
         'temp': 'TemperaturaDoAr'}

df = df.drop(columns = drop_cols).rename(columns=rename_cols)
df['Precipitacao'] = df['Precipitacao'].fillna(0)


# In[ ]:


df.to_csv('../../../data/cleandata/OpenWeather/history_bulk.csv',
          sep = ';', index = False)


# In[ ]:


df.Data_Hora

