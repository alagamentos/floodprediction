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


# In[ ]:


df = pd.read_csv('../../../data/cleandata/Ordens de serviço/Enchentes_LatLong.csv',
                 sep = ';')

est = pd.read_csv('../../../data/cleandata/Estacoes/lat_lng_estacoes.csv', sep = ';')


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


# # Pre-processamento

# In[ ]:


ord_serv = get_distances(est, df)
ord_serv.loc[ord_serv['Distance'] > 4.5, 'Est. Prox'] = 'Null'


# In[ ]:


ord_serv = ord_serv[['lat','lng','Data', 'Hora', 'Est. Prox']]
#ord_serv.loc[:,'Data'] = pd.to_datetime(ord_serv.loc[:,'Data'])
ord_serv = ord_serv.sort_values(['Data', 'Hora'])

ord_serv['pos'] = ord_serv['lat'].astype(str).str.rstrip() +                   ord_serv['lng'].astype(str).str.rstrip() 

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(ord_serv['pos'])
ord_serv['pos'] = le.transform(ord_serv['pos'])


# In[ ]:


ord_serv['Data_Hora'] = ord_serv['Data'] + ' ' + ord_serv['Hora']


# In[ ]:


df_merged = pd.read_csv('../../../data/cleandata/Info pluviometricas/Merged Data/merged.csv', sep = ';')
#df_merged = pd.read_csv('../../../data/cleandata/Info pluviometricas/Merged Data/old_merged.csv', sep = ';')
df_repaired = pd.read_csv('../../../data/cleandata/Info pluviometricas/Merged Data/repaired.csv', sep = ';')
df_merged['Data_Hora'] = pd.to_datetime(df_merged['Data_Hora'])
df_repaired['Data_Hora'] = pd.to_datetime(df_repaired['Data_Hora'])
display(df_merged)
display(df_repaired)


# In[ ]:


df_ord = ord_serv[['Est. Prox', 'Data_Hora']]
df_ord['Data_Hora'] = pd.to_datetime(df_ord['Data_Hora'], yearfirst=True).dt.strftime('%Y/%m/%d %H:%M:%S').apply(round_date)
df_ord['Data_Hora'] = pd.to_datetime(df_ord['Data_Hora'], yearfirst=True)
df_ord['Data'] = df_ord['Data_Hora'].dt.strftime('%Y/%m/%d')
df_ord = df_ord.drop(columns = 'Data_Hora')
df_ord = df_ord[df_ord['Est. Prox'] != 'OpenWeather']
df_ord


# In[ ]:


# my_map = dict(zip(ord_serv['pos'], ord_serv['Est. Prox']))
# df['Estacao'] = df.index.map(my_map)
# df = df[~(df['Estacao'] == 'Null')]

# df_est = pd.DataFrame(columns=list(my_cols))

df_est = pd.DataFrame(columns=['Data'] + list(df_ord['Est. Prox'].unique()))

for index, row in df_ord.iterrows():
    if (df_est['Data'] == row['Data']).any():
        df_est.loc[df_est['Data'] == row['Data'], row['Est. Prox']] = df_est.loc[df_est['Data'] == row['Data'], row['Est. Prox']] + 1
    else:
        df_est.loc[df_est.shape[0]] = [row['Data'], 0, 0, 0, 0, 0, 0]
        df_est.loc[df_est['Data'] == row['Data'], row['Est. Prox']] = 1

df_est


# In[ ]:


df_merged_n = df_merged.drop(columns = [c for c in df_merged.columns.values if 'Sensacao' in c]).dropna().copy()
df_repaired_n = df_repaired.drop(columns = [c for c in df_repaired.columns.values if 'Sensacao' in c]).dropna().copy()

df = df_merged_n.merge(df_repaired_n, on='Data_Hora')
df['Data_Hora'] = pd.to_datetime(df['Data_Hora'], format='%Y/%m/%d %H:%M:%S')
df['Data'] = df['Data_Hora'].dt.strftime('%Y/%m/%d')
df = df.merge(df_est, on='Data', how = 'outer')
df['Data_Hora'] = pd.to_datetime(df['Data_Hora'], yearfirst=True)
df = df.sort_values(by = 'Data_Hora')
df


# In[ ]:


df = df[~df['index'].isna()]
df[[c for c in df_est.columns if 'Data' not in c]] = df[[c for c in df_est.columns if 'Data' not in c]].fillna(0)


# # Clusterização

# In[ ]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import plotly.express as px


# In[ ]:


df_ungrouped = df.copy()
df_ungrouped = df_ungrouped.drop(columns = ['index', 'Data'] + [c for c in df.columns if 'interpol' in c])
df_ungrouped['Vitoria'] = df_ungrouped['Vitoria'] + df_ungrouped['Null']
cols = [c for c in df_ungrouped.columns if '_pred' not in c and '_repaired' not in c and c not in ['Vitoria', 'Erasmo',
       'Paraiso', 'Null', 'RM', 'Camilopolis'] and 'Local_' not in c and 'Data_Hora' not in c and 'Precipitacao' not in c]

for feature in cols: # Tira Local e Precipitacao
    df_ungrouped.loc[df_ungrouped[f'{feature}_repaired'], f'{feature}'] = df_ungrouped.loc[df_ungrouped[f'{feature}_repaired'], f'{feature}_pred']

df_ungrouped = df_ungrouped.drop(columns = [c for c in df_ungrouped.columns if '_pred' in c or '_repaired' in c or 'Null' in c])

est_to_ord = {
    'Camilopolis': 'Ordens_0',
    'Erasmo': 'Ordens_1',
    'Paraiso': 'Ordens_2',
    'RM': 'Ordens_3',
    'Vitoria': 'Ordens_4'
}
df_ungrouped = df_ungrouped.rename(columns = est_to_ord)
df_ungrouped[[c for c in df_ungrouped.columns if 'Ordens' in c]] = df_ungrouped[[c for c in df_ungrouped.columns if 'Ordens' in c]].astype(int)


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
    'Ordens'
]

ignoredFeatures = [
    'Data_Hora'
]

df_grouped = reverse_ohe(df_ungrouped, features, ignoredFeatures, 5, '_')
df_grouped['Ordens'] = df_grouped['Ordens'].astype(int)


# In[ ]:


print(f"Precipitacao: {round(df_grouped[df_grouped['Precipitacao'] > 0].shape[0] / df_grouped.shape[0] * 100, 2)}%")
print(f"Ordens: {round(df_grouped[df_grouped['Ordens'] > 0].shape[0] / df_grouped.shape[0] * 100, 2)}%")
df_grouped[df_grouped['Ordens'] > 0].shape[0]


# In[ ]:


df_prec = df_grouped.copy()
df_prec['Ano'] = df_prec['Data_Hora'].dt.year
df_prec['Mes'] = df_prec['Data_Hora'].dt.month
df_prec['Dia'] = df_prec['Data_Hora'].dt.day
df_prec['Data'] = df_prec['Data_Hora'].dt.strftime('%Y-%m-%d')


# In[ ]:


# # Agrupar por dia e local
# df_prec = df_prec.drop(columns = ['Data_Hora'])
# s_prec_p = df_prec.groupby(['Local', 'Ano', 'Mes', 'Dia']).sum().reset_index()['Precipitacao']
# s_prec_o = df_prec.groupby(['Local', 'Ano', 'Mes', 'Dia']).max().reset_index()['Ordens']
# df_prec = df_prec.groupby(['Local', 'Ano', 'Mes', 'Dia']).mean().reset_index()
# df_prec['Precipitacao'] = s_prec_p
# df_prec['Ordens'] = s_prec_o
# df_prec['Ordens'] = df_prec['Ordens'].astype(int)


# In[ ]:


# Agrupar por dia
df_prec = df_prec.drop(columns = ['Data_Hora'])
s_prec_p = df_prec.groupby(['Data', 'Local']).sum().reset_index().groupby('Data').mean()['Precipitacao'].reset_index()
s_prec_o = df_prec.groupby(['Data', 'Local']).max().reset_index().groupby('Data').max()['Ordens'].reset_index()
df_prec = df_prec.groupby(['Data']).mean().reset_index()
df_prec['Precipitacao'] = s_prec_p['Precipitacao']
df_prec['Ordens'] = s_prec_o['Ordens']
#df_prec['Ordens'] = df_prec['Ordens'].astype(int)


# In[ ]:


df_prec


# In[ ]:


sc = MinMaxScaler(feature_range=(0,1))
df_norm = sc.fit_transform(df_prec[['Precipitacao', 'Ordens']])
df_norm


# In[ ]:


ks = range(1, 10)
inertias = []
for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)
    
    # Fit model to samples
    model.fit(df_norm)
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
plt.plot(ks, inertias, '-o', color='black')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()


# In[ ]:


cluster = KMeans(n_clusters=4, random_state=42).fit(df_norm)
df_prec['Cluster'] = cluster.labels_


# In[ ]:


# fig = px.bar(df_prec.groupby(['Cluster', 'Local'])[['Mes']].count().reset_index(),
#              x="Cluster", y="Mes", color="Local", barmode="group")

fig = px.bar(df_prec.groupby(['Cluster'])[['Mes']].count().reset_index(),
             x="Cluster", y="Mes", barmode="group")
fig.show()


# In[ ]:


df_prec.groupby('Cluster').min()


# In[ ]:


df_prec.groupby('Cluster').mean()


# In[ ]:


df_prec.groupby('Cluster').sum()


# In[ ]:


print(f"Cluster 0: {round(df_prec.groupby('Cluster').count().iloc[0,0] / df_prec.groupby('Cluster').count()['Mes'].sum() * 100, 2)}% (chuvas fracas/sem chuva)")
print(f"Cluster 1: {round(df_prec.groupby('Cluster').count().iloc[1,0] / df_prec.groupby('Cluster').count()['Mes'].sum() * 100, 2)}% (chuvas perigosas)")
print(f"Cluster 2: {round(df_prec.groupby('Cluster').count().iloc[2,0] / df_prec.groupby('Cluster').count()['Mes'].sum() * 100, 2)}% (chuvas fortes?)")
print(f"Cluster 3: {round(df_prec.groupby('Cluster').count().iloc[3,0] / df_prec.groupby('Cluster').count()['Mes'].sum() * 100, 2)}% (???)")
df_prec.groupby('Cluster').count()


# In[ ]:


fig = px.bar(df_prec.groupby(['Cluster'])[['Precipitacao']].mean().reset_index(),
             x="Cluster", y="Precipitacao", barmode="group")
fig.show()


# In[ ]:


df_prec.corr()


# In[ ]:


df_prec[(df_prec['Cluster'] == 0) & (df_prec['Ordens'] >= 1)][['Precipitacao', 'Ordens']].boxplot()


# In[ ]:


# df_m = pd.read_csv('../../../data/cleandata/Info pluviometricas/Merged Data/merged.csv', sep = ';')
# df_m['Data_Hora'].max()


# # Preparatório para AutoML

# In[ ]:


# df_prec['Data'] = pd.to_datetime(df_prec['Ano'].map(str) + "-" + df_prec['Mes'].map(str) + "-" + df_prec['Dia'].map(str))
# df_grouped['Data'] = pd.to_datetime(df_grouped['Data_Hora'].dt.strftime('%Y-%m-%d'))

# df_cluster = df_grouped.merge(df_prec[['Data', 'Local', 'Cluster']], on=['Data', 'Local'])
# df_cluster = df_cluster.drop(columns = 'Data')
# df_cluster.head(20)


# In[ ]:


# df_cluster['Data'] = df_cluster['Data_Hora'].dt.strftime('%Y-%m-%d')
# s_prec_p = df_cluster.groupby(['Data']).sum().reset_index()['Precipitacao']
# s_prec_o = df_cluster.groupby(['Data']).sum().reset_index()['Ordens']
# df_cluster_group = df_cluster.drop(columns='Local').groupby(['Data']).mean().reset_index()
# df_cluster_group['Precipitacao'] = s_prec_p
# df_cluster_group['Ordens'] = s_prec_o
# df_cluster_group


# In[ ]:


df_cluster = df_prec.drop(columns = ['Data', 'Ano', 'Dia'])
#df_cluster['Ordens'] = df_cluster['Ordens'].shift(-1, fill_value = 0)
df_cluster.loc[df_cluster['Ordens'] >= 1, 'Ordens'] = 1
df_cluster


# In[ ]:


# df_cluster.to_csv('../../../data/cleandata/Info pluviometricas/Merged Data/clustered_data.csv', sep = ';', index=False)


# In[ ]:


import pandas_gbq
from google.oauth2 import service_account

PROJECT_ID = 'temporal-285820'
TABLE_clustered = 'info_pluviometrica.clustered_date'

CREDENTIALS = service_account.Credentials.from_service_account_file('../../../key/temporal-285820-cde76c259484.json')
pandas_gbq.context.credentials = CREDENTIALS


# In[ ]:


# pandas_gbq.to_gbq(df_cluster, TABLE_clustered, project_id=PROJECT_ID, credentials=CREDENTIALS, if_exists='replace')
# print('clustered done!')


# In[ ]:


# df_grouped['Data'] = df_grouped['Data_Hora'].dt.strftime('%Y-%m-%d')
# #df_clustered_total = df_grouped.merge(df_prec[['Data', 'Cluster']], on='Data').drop(columns = 'Data')
# df_clustered_total = df_grouped.merge(df_prec[['Data', 'OrdensServico', 'Cluster', 'LocalMax', 'Label']], on='Data').drop(columns = 'Data')
# #df_clustered_total['Ordens'] = df_clustered_total['Ordens'].shift(-1, fill_value = 0)
# df_clustered_total.loc[df_clustered_total['Ordens'] >= 1, 'Ordens'] = 1
# df_clustered_total


# In[ ]:


# df_clustered_total.to_csv('../../../data/cleandata/Info pluviometricas/Merged Data/clustered.csv', sep = ';', index=False)


# # Clusterizar por label nova

# In[ ]:


ords = pd.read_csv('../../../data/cleandata/Ordens de serviço/Ordens_Label.csv', sep = ';')


# In[ ]:


df_label = df_prec.merge(ords, on='Data')


# In[ ]:


sc = MinMaxScaler(feature_range=(0,1))
df_norm = sc.fit_transform(df_label[['Precipitacao', 'Label']])
df_norm


# In[ ]:


ks = range(1, 10)
inertias = []
for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)
    
    # Fit model to samples
    model.fit(df_norm)
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
plt.plot(ks, inertias, '-o', color='black')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()


# In[ ]:


cluster = KMeans(n_clusters=2, random_state=42).fit(df_norm)
df_label['Cluster'] = cluster.labels_


# In[ ]:


fig = px.bar(df_label.groupby(['Cluster'])[['Mes']].count().reset_index(),
             x="Cluster", y="Mes", barmode="group")
fig.show()


# In[ ]:


print(f"Cluster 0: {round(df_prec.groupby('Cluster').count().iloc[0,0] / df_prec.groupby('Cluster').count()['Mes'].sum() * 100, 2)}% (chuvas fracas/sem chuva)")
print(f"Cluster 1: {round(df_prec.groupby('Cluster').count().iloc[1,0] / df_prec.groupby('Cluster').count()['Mes'].sum() * 100, 2)}% (chuvas perigosas)")
print(f"Cluster 2: {round(df_prec.groupby('Cluster').count().iloc[2,0] / df_prec.groupby('Cluster').count()['Mes'].sum() * 100, 2)}% (chuvas fortes?)")
print(f"Cluster 3: {round(df_prec.groupby('Cluster').count().iloc[3,0] / df_prec.groupby('Cluster').count()['Mes'].sum() * 100, 2)}% (???)")
df_label.groupby('Cluster').count()


# In[ ]:


df_label.groupby('Cluster').min()


# In[ ]:


df_label.groupby('Cluster').mean()


# In[ ]:


df_label.groupby('Cluster').sum()


# In[ ]:


df_label[['Cluster', 'Label']].boxplot()


# In[ ]:


df_prec = df_label.copy()
df_cluster = df_prec.drop(columns = ['Data', 'Ano', 'Dia'])
df_cluster.loc[df_cluster['Ordens'] >= 1, 'Ordens'] = 1
df_cluster


# In[ ]:


df_grouped['Data'] = df_grouped['Data_Hora'].dt.strftime('%Y-%m-%d')
#df_clustered_total = df_grouped.merge(df_prec[['Data', 'Cluster']], on='Data').drop(columns = 'Data')
df_clustered_total = df_grouped.merge(df_prec[['Data', 'OrdensServico', 'Cluster', 'LocalMax', 'Label']], on='Data').drop(columns = 'Data')
#df_clustered_total['Ordens'] = df_clustered_total['Ordens'].shift(-1, fill_value = 0)
df_clustered_total.loc[df_clustered_total['Ordens'] >= 1, 'Ordens'] = 1
df_clustered_total


# In[ ]:


df_cluster.to_csv('../../../data/cleandata/Info pluviometricas/Merged Data/clustered_data.csv', sep = ';', index=False)
df_clustered_total.to_csv('../../../data/cleandata/Info pluviometricas/Merged Data/clustered.csv', sep = ';', index=False)

