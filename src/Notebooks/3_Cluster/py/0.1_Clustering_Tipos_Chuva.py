#!/usr/bin/env python
# coding: utf-8

# In[55]:


path = 'https://raw.githubusercontent.com/tbrugz/geodata-br/master/geojson/geojs-35-mun.json'

from urllib.request import urlopen
import json
with urlopen(path) as response:
    counties = json.load(response)
    
SA = [ i for i in counties['features'] if i['properties']['name'] == 'Santo André' ][0]


# In[56]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from plotly import graph_objects as go
import plotly as py

from datetime import datetime
from datetime import timedelta

py.offline.init_notebook_mode()


# In[57]:


df = pd.read_csv('../../../data/cleandata/Ordens de serviço/Enchentes_LatLong.csv',
                 sep = ';')

est = pd.read_csv('../../../data/cleandata/Estacoes/lat_lng_estacoes.csv', sep = ';')


# In[72]:


def Calculate_Dist(lat1, lon1, lat2, lon2):
    r = 6371
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2)**2 + np.cos(phi1) *        np.cos(phi2) *   np.sin(delta_lambda / 2)**2
    res = r * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))
    return np.round(res, 2)

def get_distances(estacoes, ord_serv):
    for index, row in ord_serv.iterrows():
        dist = estacoes.apply(lambda x: 
                           Calculate_Dist(row['lat'], row['lng'],
                                          x['lat'],   x['lng']),
                           axis=1)
        ord_serv.loc[index,'Distance'], arg = dist.min(), dist.argmin()
        ord_serv.loc[index,'Est. Prox'] = estacoes.iloc[arg,0]

    return ord_serv

def reverse_ohe(df, features, ignoredFeatures, featuresLength, prefix, suffix = ''):
    all_features = ignoredFeatures + features
    df_pivot = pd.DataFrame(columns = all_features)

    for i in range(featuresLength):
        cols_aux = [f'{feature}{prefix}{i}{suffix}' for feature in features]
        df_aux = df[ignoredFeatures + cols_aux].copy()
        df_aux.columns = all_features
        df_pivot = pd.concat([df_pivot, df_aux])

    return df_pivot.sort_values(by='Data_Hora').copy()

def round_date(date_string):
    left = date_string[:-5]
    minute = date_string[-5:-3]
    minute = str(round(int(minute)/15) * 15)
    minute = '00' if minute == '0' else minute
    if minute == '60':
        minute = '00'
        date_concat = left + minute + ':' + '00'
        date_concat = datetime.strptime(date_concat, '%d/%m/%Y %H:%M:%S')
        date_concat = date_concat + timedelta(hours = 1)
        date_concat = date_concat.strftime('%d/%m/%Y %H:%M:%S')
    else:
        date_concat = left + minute + ':' + '00'

    return date_concat


# In[59]:


ord_serv = get_distances(est, df)
ord_serv.loc[ord_serv['Distance'] > 4.5, 'Est. Prox'] = 'Null'


# In[60]:


ord_serv = ord_serv[['lat','lng','Data', 'Hora', 'Est. Prox']]
#ord_serv.loc[:,'Data'] = pd.to_datetime(ord_serv.loc[:,'Data'])
ord_serv = ord_serv.sort_values(['Data', 'Hora'])

ord_serv['pos'] = ord_serv['lat'].astype(str).str.rstrip() +                   ord_serv['lng'].astype(str).str.rstrip() 

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(ord_serv['pos'])
ord_serv['pos'] = le.transform(ord_serv['pos'])


# In[61]:


ord_serv['Data_Hora'] = ord_serv['Data'] + ' ' + ord_serv['Hora']


# In[129]:


df_merged = pd.read_csv('../../../data/cleandata/Info pluviometricas/Merged Data/merged.csv', sep = ';')
df_repaired = pd.read_csv('../../../data/cleandata/Info pluviometricas/Merged Data/repaired.csv', sep = ';')
#df_merged['Data_Hora'] = pd.to_datetime(df_merged['Data_Hora'])
#df_repaired['Data_Hora'] = pd.to_datetime(df_repaired['Data_Hora'])
display(df_merged)
display(df_repaired)


# In[110]:


df_ord = ord_serv[['Est. Prox', 'Data_Hora']]
df_ord['Data_Hora'] = pd.to_datetime(df_ord['Data_Hora'], format='%d/%m/%Y %H:%M:%S').dt.strftime('%d/%m/%Y %H:%M:%S').apply(round_date)
df_ord['Data_Hora'] = pd.to_datetime(df_ord['Data_Hora'], format='%d/%m/%Y %H:%M:%S')
df_ord['Data'] = df_ord['Data_Hora'].dt.strftime('%d/%m/%Y')
#df_ord = df_ord.drop(columns = 'Data_Hora')
df_ord


# In[95]:


# my_map = dict(zip(ord_serv['pos'], ord_serv['Est. Prox']))
# df['Estacao'] = df.index.map(my_map)
# df = df[~(df['Estacao'] == 'Null')]

# df_est = pd.DataFrame(columns=list(my_cols))

df_est = pd.DataFrame(columns=['Data'] + list(df_ord['Est. Prox'].unique()))

for index, row in df_ord.iterrows():
    if (df_est['Data'] == row['Data']).any():
        df_est.loc[df_est['Data'] == row['Data'], row['Est. Prox']] = df_est.loc[df_est['Data'] == row['Data'], row['Est. Prox']] + 1
    else:
        df_est.loc[df_est.shape[0]] = [row['Data'], 0, 0, 0, 0, 0, 0]
        df_est.loc[df_est['Data'] == row['Data'], row['Est. Prox']] = 1

df_est


# In[111]:


df_merged_n = df_merged.drop(columns = [c for c in df_merged.columns.values if 'Sensacao' in c]).dropna().copy()
df_repaired_n = df_repaired.drop(columns = [c for c in df_repaired.columns.values if 'Sensacao' in c]).dropna().copy()

df = df_merged_n.merge(df_repaired_n, on='Data_Hora')
#df['Data_Hora'] = pd.to_datetime(df['Data_Hora'], format='%d/%m/%Y %H:%M:%S')
#df['Data'] = df['Data_Hora'].dt.strftime('%d/%m/%Y')
# df = df.merge(df_est, on='Data', how = 'outer')
# df['Data_Hora'] = pd.to_datetime(df['Data_Hora'])
# df = df.sort_values(by = 'Data_Hora')
# df


# In[128]:


#pd.to_datetime(pd.Series(['08/05/2019', '31/05/2019']), dayfirst=True)
df_merged


# df_merged[df_merged['Data'] == '08/05/2019']

# In[66]:


print(df[(~df['Paraiso'].isna())].shape)
print(df[(~df['Paraiso'].isna()) & df['index'].isna()].shape)


# In[67]:


df[(~df['Paraiso'].isna()) & df['index'].isna()]


# In[97]:


#df[df['Data'] == '08/05/2019']
df[df['index'] == 73933]
#ord_serv.loc[100]['Data_Hora']
#df


# In[155]:


ord_serv


# In[163]:




