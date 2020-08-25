#!/usr/bin/env python
# coding: utf-8

# In[ ]:


path = 'https://raw.githubusercontent.com/tbrugz/geodata-br/master/geojson/geojs-35-mun.json'

from urllib.request import urlopen
import json
with urlopen(path) as response:
    counties = json.load(response)
    
SA = [ i for i in counties['features'] if i['properties']['name'] == 'Santo André' ][0]


# In[ ]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from plotly import graph_objects as go
import plotly as py

py.offline.init_notebook_mode()


# In[ ]:


df = pd.read_csv('../../../data/cleandata/Ordens de serviço/Enchentes_LatLong.csv',
                 sep = ';')

est = pd.read_csv('../../../data/cleandata/Estacoes/lat_lng_estacoes.csv', sep = ';')


# In[ ]:


est=est.iloc[1:-1]


# In[ ]:


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


# In[ ]:


ord_serv = get_distances(est, df)
ord_serv.loc[ord_serv['Distance'] > 4.5, 'Est. Prox'] = 'Null'


# In[ ]:


fig = go.Figure()

colors = dict(zip(ord_serv['Est. Prox'].unique(),
                  ['black', 'green', 'yellow', 'teal', 'orange', 'blue']) )

fig.add_trace(go.Scatter(x=ord_serv['lng'],
                         y= ord_serv['lat'],
                         marker=dict(
                                    size=7,
                                    color=ord_serv['Est. Prox'].apply(lambda x: colors[x]), #set color equal to a variable
                                    showscale=False
                                ),
                    showlegend = False,
                    mode='markers',
                    name='markers'))

fig.add_trace(go.Scatter(x = est['lng'],
                         y = est['lat'],
                         marker_symbol = 'x',
                         marker=dict(
                                    size=10,
                                    color='red', #set color equal to a variable
                                    showscale=False
                                ),
                    showlegend = False,
                    mode='markers',
                    name='markers'))

fig.show()


# In[ ]:


ord_serv = ord_serv[['lat','lng','Data', 'Est. Prox']]
ord_serv.loc[:,'Data'] = pd.to_datetime(ord_serv.loc[:,'Data']) 
ord_serv = ord_serv.sort_values('Data')
ord_serv = ord_serv[ord_serv['Data'] >= '2011-01-13']

ord_serv['pos'] = ord_serv['lat'].astype(str).str.rstrip() +                   ord_serv['lng'].astype(str).str.rstrip() 

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(ord_serv['pos'])
ord_serv['pos'] = le.transform(ord_serv['pos'])


# In[ ]:


my_index = np.sort(ord_serv['pos'].unique())
my_cols = ord_serv.Data.dt.strftime('%Y-%m-%d').unique()

df = pd.DataFrame(columns=list(my_cols), index = list(my_index))
df.loc[:,:] = 0


# In[ ]:


from datetime import datetime
from datetime import timedelta

day_delta = 4
for d in df.columns:

    lim_dates = [datetime.strptime(d, '%Y-%m-%d') + timedelta(days=-day_delta),
                 datetime.strptime(d, '%Y-%m-%d') + timedelta(days=day_delta)]

    selected_dates = ord_serv[(ord_serv['Data'] > lim_dates[0]) &
                        (ord_serv['Data'] <= lim_dates[1])]
        
    df.loc[df.index.isin(selected_dates.pos),d] = 1


# In[ ]:


ord_serv.head(1)


# In[ ]:


my_map = dict(zip(ord_serv['pos'], ord_serv['Est. Prox']))
df['Estacao'] = df.index.map(my_map)
#df = df[~(df['Estacao'] == 'Null')]

df_est = pd.DataFrame(columns=list(my_cols))

for est in ord_serv['Est. Prox'].unique():
    df_est.loc[est,:] =  df[df['Estacao'] == est].drop(columns = ['Estacao']).sum(axis = 0)    


# In[ ]:


import plotly.express as px

df_plot = df_est.T
df_plot['Date'] = df_plot.index
df_plot['Date'] = pd.to_datetime(df_plot['Date'])

fig = px.line(df_plot, x="Date", y=list(df_plot.columns)[:-1],
              title='Ordens de Serviço')
fig.show()


# In[ ]:


merged = pd.read_csv('../../../data/cleandata/Info pluviometricas/Merged Data/merged.csv',
                    sep = ';')

local_cols = [i for i in merged.columns if 'Local' in i]
merged[local_cols].head(1)


# In[ ]:


precipitacao_cols = ['Data_Hora'] + [i for i in merged.columns if 'Precipitacao' in i]
#merged[merged['Data'] == '08/05/19'][precipitacao_cols].sum()


# In[ ]:


merged.head(1)


# In[ ]:




fig = px.line(merged, x="Data_Hora", y=precipitacao_cols[1:],
              title='Precipitacao')
fig.show()


# In[ ]:


precipitacao = merged[precipitacao_cols]

precipitacao['Data'] = pd.to_datetime(precipitacao['Data_Hora']).dt.strftime('%Y-%m-%d')
precipitacao = precipitacao.drop(columns = 'Data_Hora')

# Selectionar somente dias com chamadas de ordem de serviço
precipitacao = precipitacao[precipitacao.Data.isin(my_cols)]


# In[ ]:


rename_dict = dict(zip(['Precipitacao_0','Precipitacao_1','Precipitacao_2','Precipitacao_3','Precipitacao_4'],
                       ['Camilopolis','Erasmo','Paraiso','RM','Vitoria']))

precipitacao = precipitacao.groupby('Data').mean()
precipitacao = precipitacao.rename(columns = rename_dict)


# In[ ]:


precipitacao_plot = precipitacao.copy()
precipitacao_plot['Date'] = pd.to_datetime(precipitacao.index)

fig = px.line(precipitacao_plot, x="Date", y=list(precipitacao_plot.columns)[:-1],
              title='Precipitacao')
fig.show()


# In[ ]:


t_precipitacao = precipitacao.T.add_suffix('_precipitacao').copy()

df_est = df_est.reset_index().merge(t_precipitacao.reset_index(), on = 'index')


# In[ ]:


df_est = df_est.dropna(axis = 1)
df_est = df_est.set_index('index')


# In[ ]:


t_df_est = df_est.T.copy()
t_df_est = t_df_est[t_df_est.columns.sort_values()].astype(float)


# In[ ]:


df_plot[df_plot.columns.sort_values()].drop(columns = ['Date']).astype(float).corr()


# In[ ]:


t_df_est.corr()


# In[ ]:


precipitacao_plot.corr()

