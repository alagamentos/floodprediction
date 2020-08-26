#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from plotly import graph_objects as go
import plotly.express as px
import plotly as py
py.offline.init_notebook_mode()


# In[ ]:


df = pd.read_csv('../../../data/cleandata/Ordens de serviço/Enchentes_LatLong.csv',
                 sep = ';')

est = pd.read_csv('../../../data/cleandata/Estacoes/lat_lng_estacoes.csv', sep = ';')
est = est.iloc[:-1] # Remove OpenWeather


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


# ### Cluster by location (lat, lng) 
# 
# create subregions

# In[ ]:


lat_lng = ord_serv[['lat','lng']]

from sklearn.cluster import KMeans

n_clusters = 30

import matplotlib.pyplot as plt
import numpy as np

clusterer = KMeans(n_clusters = n_clusters).fit(lat_lng)
ord_serv['cluster'] = clusterer.labels_


# In[ ]:


fig = go.Figure()

colors = dict(zip(ord_serv['cluster'].unique(),
                  px.colors.qualitative.Dark24[:n_clusters]) )

for c in np.sort(ord_serv['cluster'].unique()):
    fig.add_trace(go.Scatter(x=ord_serv[ord_serv['cluster'] == c]['lng'],
                             y= ord_serv[ord_serv['cluster'] == c]['lat'],
                             marker=dict( size=7 ),
                             showlegend = True,
                             mode='markers',
                             name=f'cluster {c}')
                 )

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


# #### Unique encode for each lat,lng point

# In[ ]:


ord_serv = ord_serv[['lat','lng','Data', 'cluster']]
ord_serv.loc[:,'Data'] = pd.to_datetime(ord_serv.loc[:,'Data'])
ord_serv = ord_serv.sort_values('Data')

ord_serv['pos'] = ord_serv['lat'].astype(str).str.rstrip() +                   ord_serv['lng'].astype(str).str.rstrip() 

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(ord_serv['pos'])
ord_serv['pos'] = le.transform(ord_serv['pos'])


# #### Create datevec with all dates

# In[ ]:


def days_hours_minutes(td):
    return int(td.days), td.seconds//3600, (td.seconds//60)%60

start, stop = ord_serv['Data'].iloc[0], ord_serv['Data'].iloc[-1]

from datetime import date, timedelta
# Criar Vetor de data (15 em 15 minutos )

d,h,m = days_hours_minutes(stop - start)
total_days = d + h/24 + m/24/60 + (1)

date_vec= [start + timedelta(x) for x in 
          np.arange(0, total_days, 1)]


# #### Create DataFrame Dates x Pos

# In[ ]:


my_index = np.sort(ord_serv['pos'].unique())
my_cols = ord_serv.Data.dt.strftime('%Y-%m-%d').unique()

df = pd.DataFrame(columns=list(date_vec), index = list(my_index))
df.loc[:,:] = 0


# In[ ]:


from datetime import datetime
from datetime import timedelta

day_delta = 4
for d in df.columns:

    lim_dates = [d + timedelta(days=-day_delta),
                 d + timedelta(days=day_delta)]

    selected_dates = ord_serv[(ord_serv['Data'] > lim_dates[0]) &
                        (ord_serv['Data'] <= lim_dates[1])]
        
    selected = selected_dates.pos
    df.loc[df.index.isin(selected),d] = 1


# Sum all that belong to the same region

# In[ ]:


regions_dict = dict(zip(ord_serv['pos'], ord_serv['cluster']))
df['cluster'] = df.index.map(regions_dict)

df_cluster = pd.DataFrame(index=df.drop(columns =['cluster']).columns,
                          columns = np.sort(ord_serv['cluster'].unique()))
for cluster in ord_serv['cluster'].unique():
    df_cluster.loc[:,cluster] = df.loc[df['cluster'] == cluster,:]                                .drop(columns = ['cluster']).sum(axis = 0)


# In[ ]:


import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Heatmap(z=df_cluster.T.values,
                         x=df_cluster.T.columns,
                         colorscale = 'gray',
                         reversescale=True,
                         showscale=False)
             )
fig.update_layout()
fig.show()


# #### Regions correlation clustering

# In[ ]:


corr = df_cluster.corr().values

import scipy.cluster.hierarchy as spc

pdist = spc.distance.pdist(corr)
linkage = spc.linkage(pdist, method='complete')
idx = spc.fcluster(linkage, 0.5 * pdist.max(), 'distance')


# In[ ]:


df_cluster_T = df_cluster.T

df_cluster_T['hierarchy'] = idx

hierarchy_dict = dict(zip(df_cluster_T.index,df_cluster_T.hierarchy))

hierarchy_cluster = pd.DataFrame(index=df_cluster_T.drop(columns =['hierarchy']).columns,
                          columns = np.sort(df_cluster_T['hierarchy'].unique()))

for cluster in df_cluster_T['hierarchy'].unique():
    hierarchy_cluster.loc[:,cluster] = (df_cluster_T
                                        .loc[df_cluster_T['hierarchy'] == cluster,:]
                                        .drop(columns = ['hierarchy']).sum(axis = 0)
                                       )


# #### Cluster Analysis

# In[ ]:


for h in df_cluster_T['hierarchy'].unique():
    mean = df_cluster_T[df_cluster_T['hierarchy'] == h ]             .drop(columns=['hierarchy']).T.corr().mean().mean()
    size =  df_cluster_T[df_cluster_T['hierarchy'] == h ].shape[0]
    print(f"{h} -  mean correlation: {mean:.2}  \t | number of grouped regions: {size}")


# In[ ]:


import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Heatmap(z=hierarchy_cluster.T.values,
                         x=hierarchy_cluster.T.columns,
                         colorscale = 'gray',
                         reversescale=True,
                         showscale=False)
             )
fig.update_layout()
fig.show()


# #### Apply cluster to data

# In[ ]:


ord_serv['hcluster']  = ord_serv['pos'].map(regions_dict).map(hierarchy_dict)
orded_clusters = ord_serv.groupby('hcluster').count()                 .sort_values(by = 'lat', ascending = False).reset_index()

order_cluster = dict(zip( orded_clusters.hcluster, orded_clusters.index))


# In[ ]:


ord_serv['hcluster'] = ord_serv['pos'].map(regions_dict)                      .map(hierarchy_dict).map(order_cluster)

fig = go.Figure()

for c in np.sort(ord_serv['hcluster'].unique()):
    fig.add_trace(go.Scatter(x=ord_serv[ord_serv['hcluster'] == c]['lng'],
                             y=ord_serv[ord_serv['hcluster'] == c]['lat'],
                             marker=dict(
                                        size=7,
                                       ),
                        showlegend = True,
                        mode='markers',
                        name = f'hcluster {c}'))

fig.add_trace(go.Scatter(x = est['lng'],
                         y = est['lat'],
                         marker_symbol = 'x',
                         marker=dict(
                                    size=10,
                                    # set color equal to a variable
                                    color='red', 
                                    showscale=False
                                ),
                    showlegend = False,
                    mode='markers'))

fig.show()

