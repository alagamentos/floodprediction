#!/usr/bin/env python
# coding: utf-8

# In[83]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_samples, silhouette_score
import pandas as pd
import numpy as np
import plotly.express as px

# Pandas Config
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


# In[84]:


def reverse_ohe(df, features, ignoredFeatures, featuresLength, prefix, suffix = ''):
    all_features = ignoredFeatures + features
    df_pivot = pd.DataFrame(columns = all_features)

    for i in range(featuresLength):
        cols_aux = [f'{feature}{prefix}{i}{suffix}' for feature in features]
        df_aux = df[ignoredFeatures + cols_aux].copy()
        df_aux.columns = all_features
        df_pivot = pd.concat([df_pivot, df_aux])

    return df_pivot.sort_values(by='Data_Hora').copy()


# In[85]:


merged = pd.read_csv('../../../data/cleandata/Info pluviometricas/Merged Data/merged.csv',
                sep=';',
                dtype = {'Local_0': object, 'Local_1':object,
                          'Local_2':object,  'Local_3':object})

merged.head()


# In[86]:


merged[['Precipitacao_0', 'Precipitacao_1', 'Precipitacao_2', 'Precipitacao_3', 'Precipitacao_4']].corr()


# In[87]:


regions = pd.read_csv('../../../data/cleandata/Info pluviometricas/Merged Data/error_regions.csv', sep=';')

regions.head()


# In[88]:


df = merged.merge(regions, on = 'Data_Hora')

df.head()


# In[89]:


orders = pd.read_csv('../../../data/cleandata/Ordens de serviço/Enchentes_LatLong.csv', sep=';')

orders.head()


# In[90]:


orders[orders['Data'] == '07/01/2011']


# In[91]:


df[df['Data'] == '07/01/11'][['Hora', 'Precipitacao_0']]


# In[92]:


orders[orders['Data'] == '15/04/2018']


# In[93]:


df[df['Data'] == '15/04/18'][['Hora', 'Precipitacao_0']]


# In[94]:


features = [
    'Local',
    'UmidadeRelativa',
    'PressaoAtmosferica',
    'TemperaturaDoAr',
    'TemperaturaInterna',
    'PontoDeOrvalho',
    'SensacaoTermica',
    'RadiacaoSolar',
    'DirecaoDoVento',
    'VelocidadeDoVento',
    'Precipitacao',
]

ignoredFeatures = [
    'Data_Hora'
]


# In[95]:


repaired = pd.read_csv('../../../data/cleandata/Info pluviometricas/Merged Data/repaired.csv', sep=';')

repaired.head()


# In[96]:


df_ungrouped = merged.copy()

for i in range(5):
    for feature in features[1:-1]: # Tira Local e Precipitacao
        df_ungrouped[f'{feature}_{i}'] = repaired[f'{feature}_{i}_pred']

df_ungrouped.head()


# In[97]:


df_grouped = reverse_ohe(df_ungrouped, features, ignoredFeatures, 5, '_')
df_grouped.head()


# In[98]:


df_grouped['Data_Hora'] = pd.to_datetime(df_grouped['Data_Hora'])


# In[99]:


df_cluster = df_grouped[['Data_Hora', 'Local', 'Precipitacao', 'UmidadeRelativa', 'RadiacaoSolar']].copy()


# In[100]:


df_cluster.isna().sum()


# In[101]:


df_cluster.dropna(inplace=True)


# In[102]:


df_cluster.isna().sum()


# In[103]:


df_cluster['Ano'] = df_cluster['Data_Hora'].dt.year
df_cluster['Mes'] = df_cluster['Data_Hora'].dt.month
df_cluster['Dia'] = df_cluster['Data_Hora'].dt.day


# In[104]:


#df_cluster['Local'] = df_cluster['Local'].rank(method='dense', ascending=False).astype(int)


# In[105]:


df_cluster = df_cluster.groupby(['Local', 'Ano', 'Mes', 'Dia']).sum().reset_index()


# In[106]:


sc = MinMaxScaler(feature_range=(0,1))


# In[107]:


df_norm = sc.fit_transform(df_cluster[['Precipitacao', 'UmidadeRelativa', 'RadiacaoSolar']])
df_norm


# In[108]:


cluster = KMeans(n_clusters=4, random_state=42).fit(df_norm)


# In[121]:


df_cluster['Cluster'] = cluster.labels_
df_cluster


# In[115]:


df_cluster.groupby(['Cluster', 'Local'])[['Cluster']].count()


# In[116]:


df_cluster.groupby(['Cluster', 'Mes'])[['Cluster']].count()


# In[113]:


fig = px.bar(df_cluster.groupby(['Cluster', 'Local'])[['Mes']].count().reset_index(),
             x="Cluster", y="Mes", color="Local", barmode="group")
fig.show()


# In[114]:


fig = px.bar(df_cluster.groupby(['Cluster', 'Local'])[['Mes']].count().reset_index(),
             x="Local", y="Mes", color="Cluster", barmode="group")
fig.show()


# In[128]:


cols = [c for c in merged.columns if 'UmidadeRelativa' in c]
merged[cols].corr(method='spearman')


# In[129]:


cols = [c for c in df_ungrouped.columns if 'UmidadeRelativa' in c]
df_ungrouped[cols].corr(method='spearman')


# In[131]:


regions[[c for c in regions.columns if 'Data_Hora' not in c]].sum() / regions.shape[0] * 100


# In[ ]:


#df_grouped.corr()


# In[119]:


#df_cluster.sample(n=100000).plot.scatter('Mes', 'Precipitacao')

