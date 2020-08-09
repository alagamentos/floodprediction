#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Python imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Pandas Config
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)



ip = pd.read_csv('../../../data/cleandata/Info pluviometricas/Merged Data/merged_Repaired.csv',
                 index_col = [0],
                 dtype = {'Local_0': object, 'Local_1':object,
                          'Local_2':object,  'Local_3':object})

ip.head()


# In[66]:


ip.shape


# In[40]:


import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly as py

py.offline.init_notebook_mode()


# In[65]:



ano = 2011
label = 'RadiacaoSolar_4'

fig = make_subplots(rows = 2, cols = 1, shared_xaxes=True)

fig.add_trace(go.Scatter(x = ip.loc[ip['Ano'] == ano, 'Data_Hora'] ,
                         y = ip.loc[ip['Ano'] == ano, label],
                         name = 'Original',
                         mode="lines",
                         visible = True,
                         line = dict()
                         ),
                         row = 1, col = 1)

# error = ip.loc[ip['Ano'] == ano, [label, 'Data_Hora']].where(ip[label + '_error'])

# fig.add_trace(go.Scatter(x = error['Data_Hora'] ,
#                          y = error[label],
#                          name = 'Error',
#                          mode="lines",
#                          visible = True,
#                          line = dict(color = 'Black')
#                          ),
#                          row = 1, col = 1)

interpol = ip.loc[ip['Ano'] == ano, [label, 'Data_Hora']].where(ip[label + '_interpol'])

fig.add_trace(go.Scatter(x = interpol['Data_Hora'] ,
                         y = interpol[label],
                         name = 'Interpolation',
                         mode="lines",
                         visible = True,
                         line = dict(color='Green')
                         ),
                         row = 1, col = 1 )

repaired = ip.loc[ip['Ano'] == ano, [label + '_pred', 'Data_Hora']].where(ip[label + '_repaired'])

fig.add_trace(go.Scatter(x = repaired['Data_Hora'] ,
                         y = repaired[label + '_pred'],
                         name = 'Regression',
                         mode="lines",
                         visible = True,
                         line = dict(color='Red')
                         ),
                         row = 1, col = 1 )

fig.add_trace(go.Scatter(x = ip.loc[ip['Ano'] == ano, 'Data_Hora'] ,
                         y = ip.loc[ip['Ano'] == ano, 'RadiacaoSolar_0'],
                         name = 'Original',
                         mode="lines",
                         visible = True,
                         line = dict()
                         ),
                         row = 2, col = 1)

fig.update_layout(
    autosize=False,
    width=1600,
    height=800,)


fig.show()


# In[57]:


ip.loc[ip['Ano'] == ano, ['RadiacaoSolar_0']]

