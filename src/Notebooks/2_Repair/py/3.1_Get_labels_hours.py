#!/usr/bin/env python
# coding: utf-8

# ### Achar Labels - Hora
# 
# Tentar achar a período de enchente verdadeiro baseado nas ordens de serviço e nos valores pluviométricos

# In[ ]:


import pandas as pd
import plotly as py
from plotly import graph_objects as go
from plotly.subplots import make_subplots
import datetime

py.offline.init_notebook_mode()

ow = pd.read_csv('../../../data/cleandata/OpenWeather/history_bulk.csv', sep = ';'
                )

ip = pd.read_csv('../../../data/cleandata/Info pluviometricas/Merged Data/merged.csv',
                 sep = ';',
                 dtype = {'Local_0': object, 'Local_1':object,
                          'Local_2': object, 'Local_3':object}
                )

ords = pd.read_csv('../../../data/cleandata/Ordens de serviço/labels_day.csv',
                 sep = ';')


# #### Ordens de Serviço

# In[ ]:


ords['Data'] = pd.to_datetime(ords['Data'], yearfirst = True)
for i in range(5):
    ords[f'LocalMax_{i}'] =  ords[f'LocalMax']


# #### Info Pluviométrica

# In[ ]:


ip['Data_Hora'] = pd.to_datetime(ip['Data_Hora'], yearfirst=True)
ip.insert(0, 'Data', ip.loc[:,'Data_Hora'].dt.date)
ip.insert(0, 'Hora', ip.loc[:,'Data_Hora'].dt.hour)

precipitacao_cols = [c for c in ip.columns if 'Precipitacao'in c]
precipitacao_cols += ['Data', 'Hora']

df_p = ip[ip.loc[:,'Data_Hora'].dt.minute == 0][precipitacao_cols].reset_index(drop=True)
df_p['Data'] = pd.to_datetime(df_p['Data'], yearfirst = True)


# #### OpenWeather

# In[ ]:


ow['Data_Hora'] = pd.to_datetime(ow['Data_Hora'], yearfirst=True)
ow.insert(0, 'Data', ow.loc[:,'Data_Hora'].dt.date)
ow.insert(0, 'Hora', ow.loc[:,'Data_Hora'].dt.hour)
ow = ow[~ow['Data_Hora'].duplicated(keep = 'first')]
ow = ow[['Data','Hora','Precipitacao']]
ow['Data'] = pd.to_datetime(ow['Data'], yearfirst = True)


# #### Merge

# In[ ]:


# Merge Infopluviometrica with OpenWeather
df_m = df_p.merge(ow, how = 'outer', on = ['Data','Hora']).sort_values(by = ['Data', 'Hora'])
display(df_m.head(2))
display(df_m.tail(2))


# In[ ]:


# Merge with OrdensServico
df_m = df_m.merge(ords, on = 'Data', how = 'outer')
df_m = df_m.fillna(0)
df_m = df_m.rename(columns = {'Precipitacao':'Precipitacao_5'})
df_m.insert(0,'Data_Hora', 0 )
df_m['Data_Hora'] = pd.to_datetime(df_m['Data'].astype(str) + ' ' +
                                   df_m['Hora'].astype(str) + ':00:00', yearfirst=True)
df_m = df_m.rename(columns = {'LocalMax_ow':'LocalMax_5'})
display(df_m[df_m['LocalMax_4'] == 1].head(2))


# In[ ]:


fig = make_subplots(3,1, shared_xaxes=True)

precipitacao_cols = [c for c in df_m.columns if 'Precipitacao'in c]

ano = 2019
mes = 2

ip_ano = df_m[(df_m['Data'].dt.year == ano) & (df_m['Data'].dt.month == mes)]
ords_ano = ords[(ords['Data'].dt.year == ano) & (ords['Data'].dt.month == mes)]

for col in precipitacao_cols:
    fig.add_trace(
        go.Scatter(
            x = ip_ano['Data_Hora'],
            y = ip_ano[col],
            name = col,
            connectgaps=False),
        row = 1, col = 1)

for i in range(6):
    fig.add_trace(
        go.Bar(
            x = ip_ano['Data_Hora'],
            y = ip_ano[f'LocalMax_{i}'],
            name = f'LocalMax_{i}',),
        row = 2, col = 1)

fig.add_trace(
    go.Bar(
        x = ords_ano['Data'] + datetime.timedelta(hours = 12),
        y = ords_ano['LocalMax'],
        name = 'Local Max',),
    row = 3, col = 1)

fig.show()


# In[ ]:


rain_threshold = 2

for i in range(6):
    df_m.loc[df_m[f'Precipitacao_{i}'].fillna(0) < rain_threshold, f'LocalMax_{i}'] = 0
    
lm_cols = [c for c in df_m.columns if 'LocalMax_' in c]
n_remove = len(df_m.loc[(df_m[lm_cols].max(axis = 1) == 0) &
                      (df_m['LocalMax'] == 1), 'LocalMax'])
print(f'Removing {n_remove} OrdensServico from LocalMax')
df_m.loc[(df_m[lm_cols].max(axis = 1) == 0) &
                      (df_m['LocalMax'] == 1), 'LocalMax'] = 0


# In[ ]:


from plotly.subplots import make_subplots
fig = make_subplots(3,1, shared_xaxes=True)

precipitacao_cols = [c for c in ip.columns if 'Precipitacao'in c]

ano = 2019
mes = 1

ip_ano   = df_m[(df_m['Data'].dt.year == ano) & (df_m['Data'].dt.month == mes)]
ords_ano = ords[(ords['Data'].dt.year == ano) & (ords['Data'].dt.month == mes)]

for i in range(6):
    fig.add_trace(
        go.Scatter(
            x = ip_ano['Data_Hora'],
            y = ip_ano[f'Precipitacao_{i}'],
            name = f'Precipitacao_{i}',
            connectgaps=False),
        row = 1, col = 1)

for i in range(6):
    fig.add_trace(
        go.Bar(
            x = ip_ano['Data_Hora'],
            y = ip_ano[f'LocalMax_{i}'],
            name = f'LocalMax_{i}',),
        row = 2, col = 1)

fig.add_trace(
    go.Bar(
        x = ords_ano['Data'] + datetime.timedelta(hours = 12),
        y = ords_ano['LocalMax'] ,
        name = 'Local Max (Dia)',),
    row = 3, col = 1)

fig.show()


# In[ ]:


interest_cols = [c for c in df_m.columns if 'Local' in c]
df_m = df_m[['Data_Hora']  + interest_cols]
df_m.head()

