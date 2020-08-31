#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
ow = pd.read_csv('../../../data/cleandata/OpenWeather/history_bulk.csv', sep = ';',
                 parse_dates = ['Data_Hora'])
ip = pd.read_csv('../../../data/cleandata/Info pluviometricas/Merged Data/merged.csv',
                 sep = ';',
                 dtype = {'Local_0': object, 'Local_1':object,
                          'Local_2':object,  'Local_3':object},
                parse_dates = ['Data_Hora'])


# In[ ]:


import plotly as py
from plotly import graph_objects as go

py.offline.init_notebook_mode()


# In[ ]:


fig = go.Figure()

precipitacao_cols = [c for c in ip.columns if 'Precipitacao'in c]

ano = 2011

ip_ano = ip[ip['Data_Hora'].dt.year == ano]
ow_ano = ow[ow['Data_Hora'].dt.year == ano]

for col in precipitacao_cols:
    fig.add_trace(go.Scatter(
        x = ip_ano['Data_Hora'],
        y = ip_ano[col],
        name = col,
        connectgaps=False
                            )
                 )

fig.add_trace(go.Scatter(
    x = ow_ano['Data_Hora'],
    y = ow_ano['Precipitacao'].fillna(0),
    name = 'OpenWeather',
    connectgaps=False
                        )
             )

    
fig.show()

