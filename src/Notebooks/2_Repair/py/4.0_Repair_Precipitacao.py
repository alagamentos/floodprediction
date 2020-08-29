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
df_p = ip[ ['Data_Hora'] + precipitacao_cols]
df_p.loc[:, 'Data_Hora'] = pd.to_datetime(df_p.loc[:,'Data_Hora'], yearfirst=True)


# In[ ]:


import plotly as py
from plotly import graph_objects as go

py.offline.init_notebook_mode()

fig = go.Figure()

ano = 2013

ip_ano = df_p[df_p['Data_Hora'].dt.year == ano]

for col in precipitacao_cols:
    fig.add_trace(go.Scatter(
        x = ip_ano['Data_Hora'],
        y = ip_ano[col].fillna(0),
        name = col,
        connectgaps=False
                            )
                 )
    
fig.show()


# In[ ]:


def Euclidean_Dist(df, col1, col2):
    return np.linalg.norm(df[[col1]].values - df[[col2]].values, axis = 1)

df_p = df_p.fillna(0)
precipitacao_cols = set(precipitacao_cols)
dist = {}
for  col1 in precipitacao_cols:
    remaining_cols = precipitacao_cols.copy()
    remaining_cols.remove(col1)
    cum = 0
    for i, col2 in enumerate(remaining_cols):
        cum += Euclidean_Dist(df_p, col1, col2)
    df_p[col1+'euclidian_d'] = cum / i


# In[ ]:


import plotly as py
from plotly import graph_objects as go
from plotly.subplots import make_subplots

py.offline.init_notebook_mode()


fig = make_subplots(2,1, shared_xaxes=True )

ano = 2013

ip_ano = df_p[df_p['Data_Hora'].dt.year == ano].fillna(0)

for col in precipitacao_cols:
    fig.add_trace(go.Scatter(
        x = ip_ano['Data_Hora'],
        y = ip_ano[col],
        name = col,
        connectgaps=False
                            ),
                  row = 1, col = 1
                 )
    
for col in precipitacao_cols:
    fig.add_trace(go.Scatter(
        x = ip_ano['Data_Hora'],
        y = ip_ano[col+'_euclidian_d'],
        name = col,
        connectgaps=False
                            ),
                  row = 2, col = 1
                 )
fig.show()


# In[ ]:


def std_distance(df, col1, remaining_cols):
    median =  df[remaining_cols].median(axis = 1)
    std =  df[remaining_cols].std(axis = 1)
    mask = df[remaining_cols].std(axis = 1) == 0
    return np.abs((median - df[col1])/ std)

df_p = df_p.fillna(0)
precipitacao_cols = set(precipitacao_cols)
for  col1 in precipitacao_cols:
    remaining_cols = precipitacao_cols.copy()
    remaining_cols.remove(col1)
    df_p[col1 + '_mad'] = std_distance(df_p, col1, remaining_cols)


# In[ ]:


import plotly as py
from plotly import graph_objects as go
from plotly.subplots import make_subplots

py.offline.init_notebook_mode()


fig = make_subplots(2,1, shared_xaxes=True )

ano = 2019

ip_ano = df_p[df_p['Data_Hora'].dt.year == ano].fillna(0)
color = ['#c62828', '#283593', '#00685b', '#f9a825', '#009688']

for i, col in enumerate(precipitacao_cols):
    fig.add_trace(go.Scatter(
        x = ip_ano['Data_Hora'],
        y = ip_ano[col].fillna(0),
        name = col,
        legendgroup=col,
        line = dict(color=color[i]),
        connectgaps=False),
                  row = 1, col = 1
                 )
    
    fig.add_trace(go.Scatter(
        x = ip_ano['Data_Hora'],
        y = ip_ano[col+'_mad'].fillna(0),
        legendgroup=col,
        name = col,
        line = dict(color=color[i]),
        showlegend = False,
        connectgaps=False
                            ),
                  row = 2, col = 1
                 )
fig.show()


# In[ ]:


import plotly as py
from plotly import graph_objects as go
from plotly.subplots import make_subplots

py.offline.init_notebook_mode()
fig = make_subplots(5,1, shared_xaxes=True, shared_yaxes=True )

ano = 2013
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
        y = ip_ano[col].fillna(0).where(ip_ano[col+'_mad'] > threshold),
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


start= 0
stop = 500

n_days = 45
for col in precipitacao_cols:
    zeros = derivative_zero(ip[col].fillna(0), n_days*24*4, False,
                             plot = False, plt_start = start, plt_stop = stop)
    const_not_null = derivative_zero(ip[col].fillna(0), 8, True,
                             plot = False, plt_start = start, plt_stop = stop)
    is_nan = ip[col].isna()
    df_p[col+'_error'] = [zeros[i] or const_not_null[i] or is_nan[i] 
                          for i in range(len(df_p)) ]


# In[ ]:


import plotly as py
from plotly import graph_objects as go
from plotly.subplots import make_subplots

py.offline.init_notebook_mode()
fig = make_subplots(5,1, shared_xaxes=True, shared_yaxes=True )

ano = 2012
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

