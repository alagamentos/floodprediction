#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
sys.path.insert(1, '../../Pipeline')

import imp
import utils
imp.reload(utils)
from utils import *


# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

ip = pd.read_csv('../../../data/cleandata/Info pluviometricas/Merged Data/merged.csv',
                 sep = ';',
                 dtype = {'Local_0': object, 'Local_1':object,
                          'Local_2':object,  'Local_3':object})
ip.head(2)


# In[ ]:


precipitacao_cols =  [c for c in ip.columns if 'Precipitacao' in c ]
local_cols =  [c for c in ip.columns if 'Local' in c ]
df_p = ip[ ['Data_Hora'] + local_cols + precipitacao_cols]
df_p.loc[:, 'Data_Hora'] = pd.to_datetime(df_p.loc[:,'Data_Hora'], yearfirst=True)


# In[ ]:


# import plotly as py
# from plotly import graph_objects as go

# py.offline.init_notebook_mode()

# fig = go.Figure()

# ano = 2013

# ip_ano = df_p[df_p['Data_Hora'].dt.year == ano]

# for col in precipitacao_cols:
#     fig.add_trace(go.Scatter(
#         x = ip_ano['Data_Hora'],
#         y = ip_ano[col].fillna(0),
#         name = col,
#         connectgaps=False
#                             )
#                  )
    
# fig.show()


# In[ ]:


# def Euclidean_Dist(df, col1, col2):
#     return np.linalg.norm(df[[col1]].values - df[[col2]].values, axis = 1)

# df_p = df_p.fillna(0)
# precipitacao_cols = set(precipitacao_cols)
# dist = {}
# for  col1 in precipitacao_cols:
#     remaining_cols = precipitacao_cols.copy()
#     remaining_cols.remove(col1)
#     cum = 0
#     for i, col2 in enumerate(remaining_cols):
#         cum += Euclidean_Dist(df_p, col1, col2)
#     df_p[col1+'euclidian_d'] = cum / i


# In[ ]:


# import plotly as py
# from plotly import graph_objects as go
# from plotly.subplots import make_subplots

# py.offline.init_notebook_mode()


# fig = make_subplots(2,1, shared_xaxes=True )

# ano = 2013

# ip_ano = df_p[df_p['Data_Hora'].dt.year == ano].fillna(0)

# for col in precipitacao_cols:
#     fig.add_trace(go.Scatter(
#         x = ip_ano['Data_Hora'],
#         y = ip_ano[col],
#         name = col,
#         connectgaps=False
#                             ),
#                   row = 1, col = 1
#                  )
    
# for col in precipitacao_cols:
#     fig.add_trace(go.Scatter(
#         x = ip_ano['Data_Hora'],
#         y = ip_ano[col+'euclidian_d'],
#         name = col,
#         connectgaps=False
#                             ),
#                   row = 2, col = 1
#                  )
# fig.show()


# In[ ]:


# def std_distance(df, col1, remaining_cols):
#     median =  df[remaining_cols].median(axis = 1)
#     std =  df[remaining_cols].std(axis = 1)
#     mask = df[remaining_cols].std(axis = 1) == 0
#     return np.abs((median - df[col1])/ std)

# df_p = df_p.fillna(0)
# precipitacao_cols = set(precipitacao_cols)
# for  col1 in precipitacao_cols:
#     remaining_cols = precipitacao_cols.copy()
#     remaining_cols.remove(col1)
#     df_p[col1 + '_mad'] = std_distance(df_p, col1, remaining_cols)


# In[ ]:


# import plotly as py
# from plotly import graph_objects as go
# from plotly.subplots import make_subplots

# py.offline.init_notebook_mode()


# fig = make_subplots(2,1, shared_xaxes=True )

# ano = 2019

# ip_ano = df_p[df_p['Data_Hora'].dt.year == ano].fillna(0)
# color = ['#c62828', '#283593', '#00685b', '#f9a825', '#009688']

# for i, col in enumerate(precipitacao_cols):
#     fig.add_trace(go.Scatter(
#         x = ip_ano['Data_Hora'],
#         y = ip_ano[col].fillna(0),
#         name = col,
#         legendgroup=col,
#         line = dict(color=color[i]),
#         connectgaps=False),
#                   row = 1, col = 1
#                  )
    
#     fig.add_trace(go.Scatter(
#         x = ip_ano['Data_Hora'],
#         y = ip_ano[col+'_mad'].fillna(0),
#         legendgroup=col,
#         name = col,
#         line = dict(color=color[i]),
#         showlegend = False,
#         connectgaps=False
#                             ),
#                   row = 2, col = 1
#                  )
# fig.show()


# In[ ]:


# import plotly as py
# from plotly import graph_objects as go
# from plotly.subplots import make_subplots

# py.offline.init_notebook_mode()
# fig = make_subplots(5,1, shared_xaxes=True, shared_yaxes=True )

# ano = 2013
# threshold = 50

# ip_ano = df_p[df_p['Data_Hora'].dt.year == ano].fillna(0)
# color = ['#c62828', '#283593', '#00685b', '#f9a825', '#009688']

# for i, col in enumerate(precipitacao_cols):
#     fig.add_trace(go.Scatter(
#         x = ip_ano['Data_Hora'],
#         y = ip_ano[col].fillna(0),
#         showlegend = False,
#         legendgroup=col,
#         line = dict(color='#616161'),
#         connectgaps=False
#                             ),
#                   row = i + 1, col = 1
#                  )
#     fig.add_trace(go.Scatter(
#         x = ip_ano['Data_Hora'],
#         y = ip_ano[col].fillna(0).where(ip_ano[col+'_mad'] > threshold),
#         name = col,
#         legendgroup=col,
#         showlegend = False,
#         line = dict(color='#c62828', width = 4),
#         connectgaps=False
#                             ),
#                   row = i + 1, col = 1
#                  )
    
# fig.update_layout(height=1200, width=800)
# fig.show()


# In[ ]:


start= 1707 - 200
stop = 1707 + 500

n_days = 45
for col in precipitacao_cols:
    peaks = derivative_threshold(ip[col], 30, False, start, stop, lw = 2, figsize = (11, 15))
    zeros = derivative_zero(ip[col].fillna(0), n_days*24*4, False,
                             plot = False, plt_start = start, plt_stop = stop)
    const_not_null = derivative_zero(ip[col].fillna(0), 8, True,
                             plot = False, plt_start = start, plt_stop = stop)
    is_nan = ip[col].isna()
    df_p[col+'_error'] = [zeros[i] or const_not_null[i] or is_nan[i] or peaks[i]
                          for i in range(len(df_p)) ]


# In[ ]:


import plotly as py
from plotly import graph_objects as go
from plotly.subplots import make_subplots

py.offline.init_notebook_mode()
fig = make_subplots(5,1, shared_xaxes=True, shared_yaxes=True )

ano = 2011
threshold = 50

ip_ano = df_p[df_p['Data_Hora'].dt.year == ano].fillna(0)
color = ['#c62828', '#283593', '#00685b', '#f9a825', '#009688']

for i, col in enumerate(precipitacao_cols):
    fig.add_trace(go.Scatter(
        x = ip_ano['Data_Hora'],
        y = ip_ano[col].fillna(0),
        showlegend = False,
        legendgroup=col,
        line = dict(color='#616161'),
        connectgaps=False
                            ),
                  row = i + 1, col = 1
                 )
    fig.add_trace(go.Scatter(
        x = ip_ano['Data_Hora'],
        y = ip_ano[col].fillna(0).where(ip_ano[col+'_error']),
        name = col,
        legendgroup=col,
        showlegend = False,
        line = dict(color='#c62828', width = 4),
        connectgaps=False
                            ),
                  row = i + 1, col = 1
                 )
    
fig.update_layout(height=1200, width=800)
fig.show()


# [Comparison of Spatial Interpolation Schemes for Rainfall ](https://www.mdpi.com/2073-4441/9/5/342/pdf)
# 
# [Tutorial](https://gisgeography.com/inverse-distance-weighting-idw-interpolation/)
# 
# $$Z(S_0) = \sum_{i=1}^{N} \lambda_i Z(S_i) $$
# 
# $$\lambda_i = \frac{d_{i0}^{-p}}{\sum_{i=1}^{N} d_{i0}^{-p'}}, \sum_{i=1}^{N} \lambda_i = 1$$

# In[ ]:


est = pd.read_csv('../../../data/cleandata/Estacoes/lat_lng_estacoes.csv', sep = ';')
est = est.iloc[:-1, :]


# In[ ]:


est = est.set_index('Estacao')


# In[ ]:


def Calculate_Dist(lat1, lon1, lat2, lon2):
    r = 6371
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2)**2 + np.cos(phi1) *        np.cos(phi2) * np.sin(delta_lambda / 2)**2
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


estacoes = list(est.index)

distances = {k: {} for k in estacoes}  

for estacao in estacoes:
    
    rest = [c for c in est.index if c != estacao]
    
    for r in rest:
        distances[estacao][r] = Calculate_Dist(*est.loc[estacao,:].to_list(),                                               *est.loc[r,:].to_list())


# In[ ]:


def interpolate_rain( row , num , distances):
   
    rest = [i for i in range(5) if i != num]
    row = row.fillna(0)
    
    aux_num, aux_den = 0,0
    for r in rest:
        
        p = row[f'Precipitacao_{r}']
        local_a = row[f'Local_{num}']
        local_b = row[f'Local_{r}']
        
        d = distances[local_a][local_b]
        
        aux_num += p/d * (not row[f'Precipitacao_{r}_error'])
        aux_den += 1/d * (not row[f'Precipitacao_{r}_error'])
      
    if aux_den == 0:
        return np.nan
    
    return aux_num/aux_den


# In[ ]:


for i in range(5):
    df_p.loc[df_p[f'Precipitacao_{i}_error'], f'Precipitacao_{i}'] =              df_p[df_p[f'Precipitacao_{i}_error']].apply(interpolate_rain, args = (0, distances), axis = 1 )


# In[ ]:


py.offline.init_notebook_mode()
fig = make_subplots(5,1, shared_xaxes=True, shared_yaxes=True )

ano = 2019
threshold = 50

ip_ano = df_p[df_p['Data_Hora'].dt.year == ano].fillna(0)
color = ['#c62828', '#283593', '#00685b', '#f9a825', '#009688']

for i, col in enumerate(precipitacao_cols):
    fig.add_trace(go.Scatter(
        x = ip_ano['Data_Hora'],
        y = ip_ano[col].fillna(0),
        showlegend = False,
        legendgroup=col,
        line = dict(color='#616161'),
        connectgaps=False
                            ),
                  row = i + 1, col = 1
                 )
    fig.add_trace(go.Scatter(
        x = ip_ano['Data_Hora'],
        y = ip_ano[col].fillna(0).where(ip_ano[col+'_error']),
        name = col,
        legendgroup=col,
        showlegend = False,
        line = dict(color='#c62828', width = 4),
        connectgaps=False
                            ),
                  row = i + 1, col = 1
                 )
    
fig.update_layout(height=1200, width=800)
fig.show()


# In[ ]:


df_p.isna().sum()

