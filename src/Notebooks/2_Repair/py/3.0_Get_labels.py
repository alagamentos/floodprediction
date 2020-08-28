#!/usr/bin/env python
# coding: utf-8

# ### Achar Labels
# 
# Tentar achar a período de enchente verdadeiro baseado nas ordens de serviço e nos valores pluviométricos

# In[ ]:


import pandas as pd
import plotly as py
from plotly import graph_objects as go

py.offline.init_notebook_mode()

ow = pd.read_csv('../../../data/cleandata/openweather/history_bulk.csv', sep = ';',
                 parse_dates = ['Data_Hora'])

ip = pd.read_csv('../../../data/cleandata/Info pluviometricas/Merged Data/merged.csv',
                 sep = ';',
                 dtype = {'Local_0': object, 'Local_1':object,
                          'Local_2':object,  'Local_3':object},
                 parse_dates = ['Data_Hora'])

ords = pd.read_csv('../../../data/cleandata/Ordens de serviço/Enchentes_LatLong.csv',
                 sep = ';')


# #### Agrupar Ordens de Serviço por dia - Count( )
# Total de ordens de serviço por dia

# In[ ]:


ords['Data'] = pd.to_datetime(ords['Data'], yearfirst=True)
ords_gb = ords.fillna(0).groupby('Data').count().max(axis =1).to_frame().reset_index()
ords_gb.columns = ['Data', 'OrdensServico']
ords_gb.head(2)


# 
# Vizualizar os dados de pluviometria junto com os dados de ordens de serviço

# In[ ]:


##

ano = 2011

###

from plotly.subplots import make_subplots
fig = make_subplots(2,1, shared_xaxes=True )

precipitacao_cols = [c for c in ip.columns if 'Precipitacao'in c]

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


# #### Agrupar dados de precipitação por dia 
# Reamostrando os dados de hora em hora. Isso permite somar a precipitação para obtermos o total de precipitação para tal dia

# In[ ]:


precipitacao_cols += ['Data']

ip.insert(0,'Data', ip.loc[:,'Data_Hora'].dt.date)
ip.insert(0,'Time', ip.loc[:,'Data_Hora'].dt.time)

df_p = ip[ip.loc[:,'Data_Hora'].dt.minute == 0][precipitacao_cols].groupby('Data').sum().reset_index()
df_p['Data'] = pd.to_datetime(df_p['Data'], yearfirst=True)
df_m = df_p.merge(ords_gb, how = 'outer', on='Data')
df_m['OrdensServico'] = df_m['OrdensServico'].fillna(0)


# In[ ]:


display(df_m[df_m['Precipitacao_0'].isna()].head(2))
display(df_m[df_m['Precipitacao_0'].isna()].tail(2))


# In[ ]:


ow.insert(0,'Data', ow.loc[:,'Data_Hora'].dt.date)
ow.insert(0,'Time', ow.loc[:,'Data_Hora'].dt.time)

ow_gb = ow.groupby('Data').sum()[['Precipitacao']].reset_index()
ow_gb['Data'] = pd.to_datetime(ow_gb['Data'], yearfirst = True)
ow_gb.columns = ['Data', 'Precipitacao_ow']


# In[ ]:


df_m = ow_gb.merge(df_m, on='Data', how = 'outer')


# In[ ]:


from plotly.subplots import make_subplots
fig = make_subplots(2,1, shared_xaxes=True )

precipitacao_cols = [c for c in ip.columns if 'Precipitacao'in c]

ano = 2011

df_ano = df_m[df_m['Data'].dt.year == ano]
ords_gb_ano = ords_gb[ords_gb['Data'].dt.year == ano]

for col in precipitacao_cols:
    fig.add_trace(go.Bar(
        x = df_ano['Data'],
        y = df_ano[col],
        name = col,
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


import sys
sys.path.insert(1, '../../Pipeline')

import imp
import utils
imp.reload(utils)
from utils import *


# #### Total de dias de ordem de serviço

# In[ ]:


(df_m['OrdensServico'].fillna(0) > 0).sum() 


# In[ ]:


df_m.head()


# In[ ]:


## LocalMax

regions = list_2_regions(df_m['OrdensServico'].fillna(0) > 0)
b_list = regions_2_list(regions, len(df_m))
regions = list_2_regions(b_list)

df_m.loc[:, 'LocalMax'] = 0

df_m['OrdensServico'] = df_m['OrdensServico'].fillna(0)
for r in regions:
    for i in range(r[0], r[1]+1):
        id_max = df_m.loc[i-3: i+3, 'OrdensServico'].idxmax()
        if i == id_max:
            df_m.loc[i-3: i+3, 'LocalMax'] = 0 
            df_m.loc[i, 'LocalMax'] = 1


# ### Label
# Considera os 2 primeiros dias de ordens de serviço consecutivos. Desses 2 dias são comparadas as precipitações totais e o dia com maior precipitação é o Label.

# In[ ]:


## Label

regions = list_2_regions(df_m['OrdensServico'].fillna(0) > 0)
b_list = regions_2_list(regions, len(df_m))
regions = list_2_regions(b_list)
regions = [[r[0] - 1 , r[0] + 1] for r in regions]
b_list = regions_2_list(regions, len(df_m))
df_m.loc[:, 'Label'] = 0
df_m.loc[b_list,'Label'] = 1
   


# In[ ]:


for r in regions:
    if df_m.loc[r[0],precipitacao_cols].sum() >= df_m.loc[r[0] + 1,precipitacao_cols].sum():
        df_m.loc[r[0] + 1, 'Label'] = 0 
    else:
        df_m.loc[r[0], 'Label' ] = 0


# In[ ]:


from plotly.subplots import make_subplots
fig = make_subplots(4,1, shared_xaxes=True )

precipitacao_cols = [c for c in ip.columns if 'Precipitacao'in c]

ano = 2013

df_ano = df_m[df_m['Data'].dt.year == ano]

for col in precipitacao_cols:
    fig.add_trace(go.Bar(
        x = df_ano['Data'],
        y = df_ano[col],
        name = col,
                            ),
                  row = 1, col = 1
                 )
fig.add_trace(go.Bar(
    x = df_ano['Data'],
    y = df_ano['OrdensServico'],
    name = 'Ordens de Serviço',
                        ),
                  row = 2, col = 1
             )

fig.add_trace(go.Bar(
    x = df_ano['Data'],
    y = df_ano['Label'],
    name = 'Label',
                        ),
                  row = 3, col = 1
             )

fig.add_trace(go.Bar(
    x = df_ano['Data'],
    y = df_ano['LocalMax'],
    name = 'LocalMax',
                        ),
                  row = 4, col = 1
             )
fig.show()


# In[ ]:


df_m.groupby('LocalMax').mean()


# In[ ]:


df_m.groupby('Label').mean()

