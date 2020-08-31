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

ow = pd.read_csv('../../../data/cleandata/OpenWeather/history_bulk.csv', sep = ';',
                 parse_dates = ['Data_Hora'])

ip = pd.read_csv('../../../data/cleandata/Info pluviometricas/Merged Data/merged.csv',
                 sep = ';',
                 dtype = {'Local_0': object, 'Local_1':object,
                          'Local_2': object, 'Local_3':object},
                 parse_dates = ['Data_Hora'])

ords = pd.read_csv('../../../data/cleandata/Ordens de serviço/Enchentes_LatLong.csv',
                 sep = ';')


# #### Agrupar Ordens de Serviço por hora - Count( )
# Total de ordens de serviço por hora

# In[ ]:


ords['Data'] = pd.to_datetime(ords['Data']).dt.date
ords['Hora'] = pd.to_datetime(ords['Hora'], format='%H:%M:%S').dt.hour


# In[ ]:


ords_gb = ords.fillna(0).groupby(by=['Data', 'Hora']).count().max(axis=1).to_frame()
ords_gb.columns = ['OrdensServico']
ords_gb.index


# 
# Vizualizar os dados de pluviometria junto com os dados de ordens de serviço

# #### Agrupar dados de precipitação por hora 
# Reamostrando os dados de hora em hora

# In[ ]:


ip.insert(0, 'Data', ip.loc[:,'Data_Hora'].dt.date)
ip.insert(0, 'Hora', ip.loc[:,'Data_Hora'].dt.hour)


# In[ ]:


precipitacao_cols = [c for c in ip.columns if 'Precipitacao'in c]
precipitacao_cols += ['Data', 'Hora']

df_p = ip[ip.loc[:,'Data_Hora'].dt.minute == 0][precipitacao_cols].reset_index(drop=True)
df_p = df_p.fillna(0).groupby(by=['Data', 'Hora']).sum()
df_p.index


# In[ ]:


df_m = df_p.merge(ords_gb, how='outer', left_index=True, right_index=True)
df_m['OrdensServico'] = df_m['OrdensServico'].fillna(0)
df_m


# In[ ]:


df_aux = df_m.reset_index()
df_aux[df_aux['Data'] == '2019-08-30']


# In[ ]:


display(df_m[df_m['Precipitacao_0'].isna()].head(2))
display(df_m[df_m['Precipitacao_0'].isna()].tail(2))


# In[ ]:


ow.insert(0,'Data', ow.loc[:,'Data_Hora'].dt.date)
ow.insert(0,'Hora', ow.loc[:,'Data_Hora'].dt.hour)


# In[ ]:


ow.groupby(by=['Data', 'Hora']).sum()[['Precipitacao']]


# In[ ]:


ow_gb = ow.groupby(by=['Data', 'Hora']).sum()[['Precipitacao']]
ow_gb.columns = ['Precipitacao_ow']
ow_gb.index


# In[ ]:


df_m = ow_gb.merge(df_m, how='outer', left_index=True, right_index=True)
df_m['OrdensServico'] = df_m['OrdensServico'].fillna(0)
df_m


# In[ ]:


df_aux = df_m.reset_index()
df_aux[df_aux['Data'] == '2019-08-30']


# In[ ]:


ow_gb


# In[ ]:


from plotly.subplots import make_subplots
fig = make_subplots(2,1, shared_xaxes=True)

precipitacao_cols = [c for c in ip.columns if 'Precipitacao'in c]

ano = 2011

ip_ano = df_p.reset_index()
ip_ano['Data_Hora'] = pd.to_datetime(ip_ano['Data'].astype(str) + ' ' + ip_ano['Hora'].astype(str) + ':00:00', yearfirst=True)
ip_ano = ip_ano[ip_ano['Data_Hora'].dt.year == ano]

ow_ano = ow_gb.reset_index()
ow_ano['Data_Hora'] = pd.to_datetime(ow_ano['Data'].astype(str) + ' ' + ow_ano['Hora'].astype(str) + ':00:00', yearfirst=True)
ow_ano = ow_ano[ow_ano['Data_Hora'].dt.year == ano]

ords_gb_ano = ords_gb.reset_index()
ords_gb_ano['Data_Hora'] = pd.to_datetime(ords_gb_ano['Data'].astype(str) + ' ' + ords_gb_ano['Hora'].astype(str) + ':00:00', yearfirst=True)
ords_gb_ano = ords_gb_ano[ords_gb_ano['Data_Hora'].dt.year == ano]

for col in precipitacao_cols:
    fig.add_trace(
        go.Scatter(
            x = ip_ano['Data_Hora'],
            y = ip_ano[col],
            name = col,
            connectgaps=False),
        row = 1, col = 1)

fig.add_trace(
    go.Scatter(
        x = ow_ano['Data_Hora'],
        y = ow_ano['Precipitacao_ow'].fillna(0),
        name = 'OpenWeather',
        connectgaps=False),
    row = 1, col = 1)

fig.add_trace(
    go.Bar(
        x = ords_gb_ano['Data_Hora'],
        y = ords_gb_ano['OrdensServico'],
        name = 'Ordens de Serviço',),
    row = 2, col = 1)

fig.show()


# In[ ]:


import sys
sys.path.insert(1, '../../Pipeline')

import imp
import utils
imp.reload(utils)
from utils import *

local_max = pd.read_csv('../../../data/cleandata/Info pluviometricas/Merged Data/merged_labeled.csv',
                 sep = ';')


# In[ ]:


local_max['Data'] = pd.to_datetime(local_max['Data'], yearfirst=True)
local_max.rename(columns={'Precipitacao_ow': 'Precipitacao_5'}, inplace=True)
df_m = df_m.reset_index()
df_m.rename(columns={'Precipitacao_ow': 'Precipitacao_5'}, inplace=True)

display(local_max.head())
display(df_m.head())


# In[ ]:


local_max_cols = [f'LocalMax_{i}' for i in range(6)]

for col in local_max_cols:
    local_max[col] = local_max['LocalMax']


# In[ ]:


rain_threshold = 10

for i in range(6):
    local_max.loc[local_max[f'Precipitacao_{i}'] < rain_threshold, f'LocalMax_{i}'] = 0


# In[ ]:


local_max.head()


# In[ ]:


display(local_max[local_max['Data'] == '2019-08-30'])
display(df_m[df_m['Data'] == '2019-08-30'])


# #### Total de dias de ordem de serviço

# In[ ]:


(local_max['OrdensServico'].fillna(0) > 0).sum()


# In[ ]:


precipitacao_cols = [c for c in df_m.columns if 'Precipitacao' in c]
precipitacao_cols.sort()
cols = ['Data', 'Hora'] + precipitacao_cols

df_mh = df_m.reset_index(drop=True)[cols]
df_mh.head()


# In[ ]:


cols = ['Data', 'LocalMax'] + local_max_cols

df_mh = df_mh.merge(local_max[cols], how='outer', on='Data')
df_mh.head()


# In[ ]:


rain_threshold = 10

for i in range(6):
    df_mh.loc[df_mh[f'Precipitacao_{i}'] < rain_threshold, f'LocalMax_{i}'] = 0


# In[ ]:


df_mh.head()


# In[ ]:


df_mh[df_mh['LocalMax'] == 1)]


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
    fig.add_trace(
        go.Bar(
            x = df_ano['Data'],
            y = df_ano[col],
            name = col,),
    row = 1, col = 1)

fig.add_trace(
    go.Bar(
        x = df_ano['Data'],
        y = df_ano['OrdensServico'],
        name = 'Ordens de Serviço',),
    row = 2, col = 1)

fig.add_trace(
    go.Bar(
        x = df_ano['Data'],
        y = df_ano['Label'],
        name = 'Label',),
    row = 3, col = 1)

fig.add_trace(
    go.Bar(
        x = df_ano['Data'],
        y = df_ano['LocalMax'],
        name = 'LocalMax',),
    row = 4, col = 1)

#fig.show()


# In[ ]:


df_m.groupby('LocalMax').mean()


# In[ ]:


df_m.groupby('Label').mean()


# In[ ]:


#df_m.to_csv('../../../data/cleandata/Info pluviometricas/Merged Data/merged_labeled.csv', sep = ';', index=False)

