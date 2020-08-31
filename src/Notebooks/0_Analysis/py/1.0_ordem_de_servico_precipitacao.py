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

ords = pd.read_csv('../../../data/cleandata/Ordens de serviço/Enchentes_LatLong.csv',
                 sep = ';')


# In[ ]:



import plotly as py
from plotly import graph_objects as go

py.offline.init_notebook_mode()


# In[ ]:


ords['Data'] = pd.to_datetime(ords['Data'], yearfirst=True)
ords_gb = ords.groupby('Data').count()['ID1'].reset_index()
ords_gb.columns = ['Data', 'OrdensServico']
ords_gb.head(2)


# In[ ]:


from plotly.subplots import make_subplots
fig = make_subplots(2,1, shared_xaxes=True )

precipitacao_cols = [c for c in ip.columns if 'Precipitacao'in c]

ano = 2011

ip_ano = ip[ip['Data_Hora'].dt.year == ano]
ow_ano = ow[ow['Data_Hora'].dt.year == ano]

ords_gb_ano = ords_gb[ords_gb['Data'].dt.year == ano]

for col in precipitacao_cols:
    fig.add_trace(go.Scatter(
        x = ip_ano['Data_Hora'],
        y = ip_ano[col],
        name = col,
        connectgaps=False
                            ),
                  row = 1, col = 1
                 )
    
fig.add_trace(go.Scatter(
    x = ow_ano['Data_Hora'],
    y = ow_ano['Precipitacao'].fillna(0),
    name = 'OpenWeather',
    connectgaps=False
                        ),
                  row = 1, col = 1
             )
    
fig.add_trace(go.Bar(
    x = ords_gb_ano['Data'],
    y = ords_gb_ano['OrdensServico'],
    name = 'Ordens de Serviço',
                        ),
                  row = 2, col = 1
             )

    
fig.show()


# In[ ]:




