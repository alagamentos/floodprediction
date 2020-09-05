#!/usr/bin/env python
# coding: utf-8

# # Inicialização

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

from datetime import datetime
from datetime import timedelta

py.offline.init_notebook_mode()


# # Funções

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

def reverse_ohe(df, features, ignoredFeatures, featuresLength, prefix, suffix = ''):
    all_features = ignoredFeatures + features
    df_pivot = pd.DataFrame(columns = all_features)

    for i in range(featuresLength):
        cols_aux = [f'{feature}{prefix}{i}{suffix}' for feature in features]
        df_aux = df[ignoredFeatures + cols_aux].copy()
        df_aux.columns = all_features
        df_pivot = pd.concat([df_pivot, df_aux])

    return df_pivot.sort_values(by='Data_Hora').copy()

def round_date(date_string):
    left = date_string[:-5]
    minute = date_string[-5:-3]
    minute = str(round(int(minute)/15) * 15)
    minute = '00' if minute == '0' else minute
    if minute == '60':
        minute = '00'
        date_concat = left + minute + ':' + '00'
        date_concat = datetime.strptime(date_concat, '%Y/%m/%d %H:%M:%S')
        date_concat = date_concat + timedelta(hours = 1)
        date_concat = date_concat.strftime('%Y/%m/%d %H:%M:%S')
    else:
        date_concat = left + minute + ':' + '00'

    return date_concat


# # blabla

# In[ ]:


df_merged = pd.read_csv('../../../data/cleandata/Info pluviometricas/Merged Data/merged.csv', sep = ';')
df_repaired = pd.read_csv('../../../data/cleandata/Info pluviometricas/Merged Data/repaired.csv', sep = ';')
df_merged['Data_Hora'] = pd.to_datetime(df_merged['Data_Hora'], yearfirst=True)
df_repaired['Data_Hora'] = pd.to_datetime(df_repaired['Data_Hora'], yearfirst=True)
display(df_merged)
display(df_repaired)


# In[ ]:


df_merged_n = df_merged.drop(columns = [c for c in df_merged.columns.values if 'Sensacao' in c]).dropna().copy()
df_repaired_n = df_repaired.drop(columns = [c for c in df_repaired.columns.values if 'Sensacao' in c]).dropna().copy()

df = df_merged_n.merge(df_repaired_n, on='Data_Hora')
df['Data_Hora'] = pd.to_datetime(df['Data_Hora'], format='%Y/%m/%d %H:%M:%S')
df['Data'] = df['Data_Hora'].dt.strftime('%Y/%m/%d')


# In[ ]:


df_ungrouped = df.copy()
df_ungrouped = df_ungrouped.drop(columns = ['index', 'Data'] + [c for c in df.columns if 'interpol' in c])
cols = [c for c in df_ungrouped.columns if '_pred' not in c and '_repaired' not in c and 'Local_' not in c and 'Data_Hora' not in c and 'Precipitacao' not in c]

for feature in cols: # Tira Local e Precipitacao
    df_ungrouped.loc[df_ungrouped[f'{feature}_repaired'], f'{feature}'] = df_ungrouped.loc[df_ungrouped[f'{feature}_repaired'], f'{feature}_pred']
df_ungrouped = df_ungrouped.drop(columns = [c for c in df_ungrouped.columns if '_pred' in c or '_repaired' in c or 'Null' in c])


# In[ ]:


df_label_h = pd.read_csv('../../../data/cleandata/Ordens de serviço/labels_hour.csv', sep = ';')
df_label_d = pd.read_csv('../../../data/cleandata/Ordens de serviço/labels_day.csv', sep = ';')


# # Por Dia

# In[ ]:


df_ungrouped['Data'] = df_ungrouped['Data_Hora'].dt.strftime('%Y-%m-%d')
df_dia = df_ungrouped.merge(df_label_d[['Data', 'LocalMax_0', 'LocalMax_1', 'LocalMax_2', 'LocalMax_3', 'LocalMax_4']], on='Data', how='outer')
df_dia = df_dia[~df_dia['Local_0'].isna()]
df_dia[['LocalMax_0', 'LocalMax_1', 'LocalMax_2', 'LocalMax_3', 'LocalMax_4']] = df_dia[['LocalMax_0', 'LocalMax_1', 'LocalMax_2', 'LocalMax_3', 'LocalMax_4']].fillna(0)


# In[ ]:


features = [
    'Local',
    'UmidadeRelativa',
    'PressaoAtmosferica',
    'TemperaturaDoAr',
    'TemperaturaInterna',
    'PontoDeOrvalho',
    'RadiacaoSolar',
    'DirecaoDoVento',
    'VelocidadeDoVento',
    'Precipitacao',
    'LocalMax'
]

ignoredFeatures = [
    'Data_Hora'
]

df_dia_grouped = reverse_ohe(df_dia, features, ignoredFeatures, 5, '_')
df_dia_grouped['LocalMax'] = df_dia_grouped['LocalMax'].astype(int)


# In[ ]:


print(f"Precipitacao: {round(df_dia_grouped[df_dia_grouped['Precipitacao'] > 0].shape[0] / df_dia_grouped.shape[0] * 100, 2)}%")
print(f"Ordens: {round(df_dia_grouped[df_dia_grouped['LocalMax'] > 0].shape[0] / df_dia_grouped.shape[0] * 100, 2)}%")
df_dia_grouped[df_dia_grouped['LocalMax'] > 0].shape[0]


# ## Clusterização

# In[ ]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import plotly.express as px


# In[ ]:


df_dia_grouped['Ano'] = df_dia_grouped['Data_Hora'].dt.year
df_dia_grouped['Mes'] = df_dia_grouped['Data_Hora'].dt.month
df_dia_grouped['Dia'] = df_dia_grouped['Data_Hora'].dt.day
df_dia_grouped['Data'] = df_dia_grouped['Data_Hora'].dt.strftime('%Y-%m-%d')


# In[ ]:


df_prec_dia = df_dia_grouped.drop(columns = ['Data_Hora'])
s_prec_p = df_prec_dia.groupby(['Data', 'Local']).sum().reset_index().groupby('Data').mean()['Precipitacao'].reset_index()
s_prec_o = df_prec_dia.groupby(['Data', 'Local']).max().reset_index().groupby('Data').max()['LocalMax'].reset_index()
df_prec_dia = df_prec_dia.groupby(['Data']).mean().reset_index()
df_prec_dia['Precipitacao'] = s_prec_p['Precipitacao']
df_prec_dia['LocalMax'] = s_prec_o['LocalMax']


# In[ ]:


sc = MinMaxScaler(feature_range=(0,1))
df_norm_dia = sc.fit_transform(df_prec_dia[['Precipitacao', 'LocalMax']])
df_norm_dia


# In[ ]:


ks = range(1, 10)
inertias = []
for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)
    
    # Fit model to samples
    model.fit(df_norm_dia)
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
plt.plot(ks, inertias, '-o', color='black')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()


# In[ ]:


cluster = KMeans(n_clusters=4, random_state=42).fit(df_norm_dia)
df_prec_dia['Cluster'] = cluster.labels_


# In[ ]:


# fig = px.bar(df_prec.groupby(['Cluster', 'Local'])[['Mes']].count().reset_index(),
#              x="Cluster", y="Mes", color="Local", barmode="group")

fig = px.bar(df_prec_dia.groupby(['Cluster'])[['Mes']].count().reset_index(),
             x="Cluster", y="Mes", barmode="group")
fig.show()


# In[ ]:


df_prec_dia.groupby('Cluster').min()


# In[ ]:


df_prec_dia.groupby('Cluster').mean()


# In[ ]:


df_prec_dia.groupby('Cluster').sum()


# In[ ]:


print(f"Cluster 0: {round(df_prec_dia.groupby('Cluster').count().iloc[0,0] / df_prec_dia.groupby('Cluster').count()['Mes'].sum() * 100, 2)}% (chuvas fracas/sem chuva)")
print(f"Cluster 1: {round(df_prec_dia.groupby('Cluster').count().iloc[1,0] / df_prec_dia.groupby('Cluster').count()['Mes'].sum() * 100, 2)}% (chuvas perigosas)")
#print(f"Cluster 2: {round(df_prec_dia.groupby('Cluster').count().iloc[2,0] / df_prec_dia.groupby('Cluster').count()['Mes'].sum() * 100, 2)}% (chuvas fortes?)")
#print(f"Cluster 3: {round(df_prec_dia.groupby('Cluster').count().iloc[3,0] / df_prec_dia.groupby('Cluster').count()['Mes'].sum() * 100, 2)}% (???)")
df_prec_dia.groupby('Cluster').count()


# In[ ]:


fig = px.bar(df_prec_dia.groupby(['Cluster'])[['Precipitacao']].mean().reset_index(),
             x="Cluster", y="Precipitacao", barmode="group")
fig.show()


# In[ ]:


df_prec_dia.corr()


# In[ ]:


df_prec_dia[(df_prec_dia['Cluster'] == 0) & (df_prec_dia['LocalMax'] >= 1)][['Precipitacao', 'LocalMax']].boxplot()


# # Por Hora

# In[ ]:


df_ungrouped['Data'] = df_ungrouped['Data_Hora'].dt.strftime('%Y-%m-%d')
df_ungrouped['Hora'] = df_ungrouped['Data_Hora'].dt.hour
df_label_h['Data_Hora'] = pd.to_datetime(df_label_h['Data_Hora'], yearfirst=True)
df_label_h['Data'] = pd.to_datetime(df_label_h['Data_Hora'], yearfirst=True).dt.strftime('%Y-%m-%d')
df_label_h['Hora'] = pd.to_datetime(df_label_h['Data_Hora'], yearfirst=True).dt.hour
df_hora = df_ungrouped.merge(df_label_h[['Data', 'Hora', 'LocalMax_0', 'LocalMax_1', 'LocalMax_2', 'LocalMax_3', 'LocalMax_4']], on=['Data', 'Hora'], how='outer')
df_hora = df_hora[~df_hora['Local_0'].isna()]
df_hora[['LocalMax_0', 'LocalMax_1', 'LocalMax_2', 'LocalMax_3', 'LocalMax_4']] = df_hora[['LocalMax_0', 'LocalMax_1', 'LocalMax_2', 'LocalMax_3', 'LocalMax_4']].fillna(0)


# In[ ]:


features = [
    'Local',
    'UmidadeRelativa',
    'PressaoAtmosferica',
    'TemperaturaDoAr',
    'TemperaturaInterna',
    'PontoDeOrvalho',
    'RadiacaoSolar',
    'DirecaoDoVento',
    'VelocidadeDoVento',
    'Precipitacao',
    'LocalMax'
]

ignoredFeatures = [
    'Data_Hora'
]

df_hora_grouped = reverse_ohe(df_hora, features, ignoredFeatures, 5, '_')
df_hora_grouped['LocalMax'] = df_hora_grouped['LocalMax'].astype(int)


# In[ ]:


print(f"Precipitacao: {round(df_hora_grouped[df_hora_grouped['Precipitacao'] > 0].shape[0] / df_hora_grouped.shape[0] * 100, 2)}%")
print(f"Ordens: {round(df_hora_grouped[df_hora_grouped['LocalMax'] > 0].shape[0] / df_hora_grouped.shape[0] * 100, 2)}%")
df_hora_grouped[df_hora_grouped['LocalMax'] > 0].shape[0]


# ## Clusterização

# In[ ]:


df_hora_grouped['Hora'] = df_hora_grouped['Data_Hora'].dt.hour
df_hora_grouped['Data'] = df_hora_grouped['Data_Hora'].dt.strftime('%Y-%m-%d')


# In[ ]:


df_prec_hora = df_hora_grouped.drop(columns = ['Data_Hora'])
s_prec_p = df_prec_hora.groupby(['Data', 'Hora', 'Local']).sum().reset_index().groupby(['Data', 'Hora']).mean()['Precipitacao'].reset_index()
s_prec_o = df_prec_hora.groupby(['Data', 'Hora', 'Local']).max().reset_index().groupby(['Data', 'Hora']).max()['LocalMax'].reset_index()
df_prec_hora = df_prec_hora.groupby(['Data', 'Hora']).mean().reset_index()
df_prec_hora['Precipitacao'] = s_prec_p['Precipitacao']
df_prec_hora['LocalMax'] = s_prec_o['LocalMax']


# In[ ]:


sc = MinMaxScaler(feature_range=(0,1))
df_norm_hora = sc.fit_transform(df_prec_hora[['Precipitacao', 'LocalMax']])
df_norm_hora


# In[ ]:


ks = range(1, 10)
inertias = []
for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)
    
    # Fit model to samples
    model.fit(df_norm_hora)
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
plt.plot(ks, inertias, '-o', color='black')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()


# In[ ]:


cluster = KMeans(n_clusters=2, random_state=42).fit(df_norm_hora)
df_prec_hora['Cluster'] = cluster.labels_


# In[ ]:


# fig = px.bar(df_prec.groupby(['Cluster', 'Local'])[['Mes']].count().reset_index(),
#              x="Cluster", y="Mes", color="Local", barmode="group")

fig = px.bar(df_prec_hora.groupby(['Cluster'])[['RadiacaoSolar']].count().reset_index(),
             x="Cluster", y="RadiacaoSolar", barmode="group")
fig.show()


# In[ ]:


df_prec_hora.groupby('Cluster').min()


# In[ ]:


df_prec_hora.groupby('Cluster').mean()


# In[ ]:


df_prec_hora.groupby('Cluster').sum()


# In[ ]:


print(f"Cluster 0: {round(df_prec_hora.groupby('Cluster').count().iloc[0,0] / df_prec_hora.groupby('Cluster').count()['RadiacaoSolar'].sum() * 100, 2)}% (chuvas fracas/sem chuva)")
print(f"Cluster 1: {round(df_prec_hora.groupby('Cluster').count().iloc[1,0] / df_prec_hora.groupby('Cluster').count()['RadiacaoSolar'].sum() * 100, 2)}% (chuvas perigosas)")
#print(f"Cluster 2: {round(df_prec_hora.groupby('Cluster').count().iloc[2,0] / df_prec_hora.groupby('Cluster').count()['RadiacaoSolar'].sum() * 100, 2)}% (chuvas fortes?)")
#print(f"Cluster 3: {round(df_prec_hora.groupby('Cluster').count().iloc[3,0] / df_prec_hora.groupby('Cluster').count()['RadiacaoSolar'].sum() * 100, 2)}% (???)")
df_prec_hora.groupby('Cluster').count()


# In[ ]:


fig = px.bar(df_prec_hora.groupby(['Cluster'])[['Precipitacao']].mean().reset_index(),
             x="Cluster", y="Precipitacao", barmode="group")
fig.show()


# In[ ]:


df_prec_hora.corr()


# In[ ]:


df_prec_dia[['Data', 'LocalMax']].rename(columns = {'LocalMax': 'LocalMax_Dia'})
df_prec_hora[['Data', 'Hora', 'LocalMax']].rename(columns = {'LocalMax': 'LocalMax_Hora'})


# In[ ]:


df_hora_grouped.merge(
    df_prec_dia[['Data', 'LocalMax']].rename(columns = {'LocalMax': 'LocalMax_Dia'}),
    on = 'Data'
).merge(
    df_prec_hora[['Data', 'Hora', 'LocalMax']].rename(columns = {'LocalMax': 'LocalMax_Hora'}),
    on = ['Data', 'Hora']
).drop(
    columns = ['LocalMax', 'Hora', 'Data']
).to_csv('../../../data/cleandata/Info pluviometricas/Merged Data/clustered_label.csv', sep = ';', index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




