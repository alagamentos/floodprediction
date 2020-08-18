#!/usr/bin/env python
# coding: utf-8

# # Inicialização

# In[1]:


path = 'https://raw.githubusercontent.com/tbrugz/geodata-br/master/geojson/geojs-35-mun.json'

from urllib.request import urlopen
import json
with urlopen(path) as response:
    counties = json.load(response)
    
SA = [ i for i in counties['features'] if i['properties']['name'] == 'Santo André' ][0]


# In[2]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from plotly import graph_objects as go
import plotly as py

from datetime import datetime
from datetime import timedelta

py.offline.init_notebook_mode()


# In[3]:


df = pd.read_csv('../../../data/cleandata/Ordens de serviço/Enchentes_LatLong.csv',
                 sep = ';')

est = pd.read_csv('../../../data/cleandata/Estacoes/lat_lng_estacoes.csv', sep = ';')


# # Funções

# In[4]:


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
        date_concat = datetime.strptime(date_concat, '%d/%m/%Y %H:%M:%S')
        date_concat = date_concat + timedelta(hours = 1)
        date_concat = date_concat.strftime('%d/%m/%Y %H:%M:%S')
    else:
        date_concat = left + minute + ':' + '00'

    return date_concat


# # Pre-processamento

# In[5]:


ord_serv = get_distances(est, df)
ord_serv.loc[ord_serv['Distance'] > 4.5, 'Est. Prox'] = 'Null'


# In[6]:


ord_serv = ord_serv[['lat','lng','Data', 'Hora', 'Est. Prox']]
#ord_serv.loc[:,'Data'] = pd.to_datetime(ord_serv.loc[:,'Data'])
ord_serv = ord_serv.sort_values(['Data', 'Hora'])

ord_serv['pos'] = ord_serv['lat'].astype(str).str.rstrip() +                   ord_serv['lng'].astype(str).str.rstrip() 

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(ord_serv['pos'])
ord_serv['pos'] = le.transform(ord_serv['pos'])


# In[7]:


ord_serv['Data_Hora'] = ord_serv['Data'] + ' ' + ord_serv['Hora']


# In[8]:


#df_merged = pd.read_csv('../../../data/cleandata/Info pluviometricas/Merged Data/merged.csv', sep = ';')
df_merged = pd.read_csv('../../../data/cleandata/Info pluviometricas/Merged Data/old_merged.csv', sep = ';')
df_repaired = pd.read_csv('../../../data/cleandata/Info pluviometricas/Merged Data/repaired.csv', sep = ';')
df_merged['Data_Hora'] = pd.to_datetime(df_merged['Data_Hora'])
df_repaired['Data_Hora'] = pd.to_datetime(df_repaired['Data_Hora'])
display(df_merged)
display(df_repaired)


# In[9]:


df_ord = ord_serv[['Est. Prox', 'Data_Hora']]
df_ord['Data_Hora'] = pd.to_datetime(df_ord['Data_Hora']).dt.strftime('%d/%m/%Y %H:%M:%S').apply(round_date)
df_ord['Data_Hora'] = pd.to_datetime(df_ord['Data_Hora'])
df_ord['Data'] = df_ord['Data_Hora'].dt.strftime('%d/%m/%Y')
df_ord = df_ord.drop(columns = 'Data_Hora')
df_ord


# In[10]:


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


# In[11]:


df_merged_n = df_merged.drop(columns = [c for c in df_merged.columns.values if 'Sensacao' in c]).dropna().copy()
df_repaired_n = df_repaired.drop(columns = [c for c in df_repaired.columns.values if 'Sensacao' in c]).dropna().copy()

df = df_merged_n.merge(df_repaired_n, on='Data_Hora')
df['Data_Hora'] = pd.to_datetime(df['Data_Hora'], format='%d/%m/%Y %H:%M:%S')
df['Data'] = df['Data_Hora'].dt.strftime('%d/%m/%Y')
df = df.merge(df_est, on='Data', how = 'outer')
df['Data_Hora'] = pd.to_datetime(df['Data_Hora'])
df = df.sort_values(by = 'Data_Hora')
df


# In[12]:


df = df[~df['index'].isna()]
df[[c for c in df_est.columns if 'Data' not in c]] = df[[c for c in df_est.columns if 'Data' not in c]].fillna(0)


# # Clusterização

# In[13]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import plotly.express as px


# In[14]:


df_ungrouped = df.copy()
df_ungrouped = df_ungrouped.drop(columns = ['index', 'Data', 'Hora'] + [c for c in df.columns if 'interpol' in c])
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


# In[15]:


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


# In[16]:


print(f"Precipitacao: {round(df_grouped[df_grouped['Precipitacao'] > 0].shape[0] / df_grouped.shape[0] * 100, 2)}%")
print(f"Ordens: {round(df_grouped[df_grouped['Ordens'] > 0].shape[0] / df_grouped.shape[0] * 100, 2)}%")
df_grouped[df_grouped['Ordens'] > 0].shape[0]


# In[31]:


df_prec = df_grouped.copy()
df_prec['Ano'] = df_prec['Data_Hora'].dt.year
df_prec['Mes'] = df_prec['Data_Hora'].dt.month
df_prec['Dia'] = df_prec['Data_Hora'].dt.day


# In[32]:


# Agrupar por dia
df_prec = df_prec.drop(columns = ['Data_Hora'])
# s_prec_p = df_prec.groupby(['Local', 'Ano', 'Mes', 'Dia']).sum().reset_index()['Precipitacao']
# s_prec_o = df_prec.groupby(['Local', 'Ano', 'Mes', 'Dia']).max().reset_index()['Ordens']
# df_prec = df_prec.groupby(['Local', 'Ano', 'Mes', 'Dia']).mean().reset_index()
# df_prec['Precipitacao'] = s_prec_p
# df_prec['Ordens'] = s_prec_o
#df_prec['Ordens'] = df_prec['Ordens'].astype(int)


# In[33]:


sc = MinMaxScaler(feature_range=(0,1))
df_norm = sc.fit_transform(df_prec[['Precipitacao', 'Ordens']])
df_norm


# In[34]:


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


# In[35]:


cluster = KMeans(n_clusters=4, random_state=42).fit(df_norm)
df_prec['Cluster'] = cluster.labels_


# In[36]:


fig = px.bar(df_prec.groupby(['Cluster', 'Local'])[['Mes']].count().reset_index(),
             x="Cluster", y="Mes", color="Local", barmode="group")
fig.show()


# In[37]:


df_prec.groupby('Cluster').min()


# In[38]:


df_prec.groupby('Cluster').mean()


# In[39]:


df_prec.groupby('Cluster').sum()


# In[40]:


print(f"Cluster 0: {round(df_prec.groupby('Cluster').count().iloc[0,0] / df_prec.groupby('Cluster').count()['Local'].sum() * 100, 2)}% (chuvas fracas/sem chuva)")
print(f"Cluster 1: {round(df_prec.groupby('Cluster').count().iloc[1,0] / df_prec.groupby('Cluster').count()['Local'].sum() * 100, 2)}% (chuvas perigosas)")
print(f"Cluster 2: {round(df_prec.groupby('Cluster').count().iloc[2,0] / df_prec.groupby('Cluster').count()['Local'].sum() * 100, 2)}% (???)")
print(f"Cluster 3: {round(df_prec.groupby('Cluster').count().iloc[3,0] / df_prec.groupby('Cluster').count()['Local'].sum() * 100, 2)}% (chuvas fortes?)")
df_prec.groupby('Cluster').count()


# In[41]:


fig = px.bar(df_prec.groupby(['Cluster'])[['Precipitacao']].mean().reset_index(),
             x="Cluster", y="Precipitacao", barmode="group")
fig.show()


# In[42]:


print('a')
print('b')
print('c')
print('d')
print('e')
print('f')


# In[43]:


df_prec.corr()


# In[44]:


df_m = pd.read_csv('../../../data/cleandata/Info pluviometricas/Merged Data/merged.csv', sep = ';')
df_m['Data_Hora'].max()

