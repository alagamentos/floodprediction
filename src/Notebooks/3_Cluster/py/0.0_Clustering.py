#!/usr/bin/env python
# coding: utf-8

# In[37]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_samples, silhouette_score
import pandas as pd
import numpy as np

# Pandas Config
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


# In[38]:


def reverse_ohe(df, features, ignoredFeatures, featuresLength, prefix, suffix = ''):
    all_features = ignoredFeatures + features
    df_pivot = pd.DataFrame(columns = all_features)

    for i in range(featuresLength):
        cols_aux = [f'{feature}{prefix}{i}{suffix}' for feature in features]
        df_aux = df[ignoredFeatures + cols_aux].copy(deep=True)
        df_aux.columns = all_features
        df_pivot = pd.concat([df_pivot, df_aux])

    return df_pivot.sort_values(by='Data_Hora').copy(deep=True)


# In[39]:


merged = pd.read_csv('../../../data/cleandata/Info pluviometricas/Merged Data/merged.csv',
                sep=';',
                dtype = {'Local_0': object, 'Local_1':object,
                          'Local_2':object,  'Local_3':object})

merged.head()


# In[40]:


regions = pd.read_csv('../../../data/cleandata/Info pluviometricas/Merged Data/error_regions.csv', sep=';')

regions.head()


# In[41]:


df = merged.merge(regions, on = 'Data_Hora')

df.head()


# In[42]:


orders = pd.read_csv('../../../data/cleandata/Ordens de serviço/Enchentes_LatLong.csv', sep=';')

orders.head()


# In[43]:


orders[orders['Data'] == '07/01/2011']


# In[44]:


df[df['Data'] == '07/01/11'][['Hora', 'Precipitacao_0']]


# In[45]:


orders[orders['Data'] == '15/04/2018']


# In[46]:


df[df['Data'] == '15/04/18'][['Hora', 'Precipitacao_0']]


# In[47]:


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


# In[ ]:


repaired = pd.read_csv('../../../data/cleandata/Info pluviometricas/Merged Data/repaired.csv', sep=';')

repaired.head()


# In[48]:


merged_grouped = reverse_ohe(merged, features, ignoredFeatures, 5, '_')
merged_grouped


# In[49]:


merged_grouped['Data_Hora'] = pd.to_datetime(merged_grouped['Data_Hora'])


# In[50]:


df_aux = merged_grouped[['Data_Hora', 'Local', 'Precipitacao']].copy()


# In[51]:


df_aux.isna().sum()


# In[52]:


df_aux.dropna(inplace=True)


# In[53]:


df_aux.isna().sum()


# In[54]:


df_aux['Ano'] = df_aux['Data_Hora'].dt.year


# In[55]:


df_aux['Mes'] = df_aux['Data_Hora'].dt.month


# In[56]:


df_aux['Dia'] = df_aux['Data_Hora'].dt.day


# In[57]:


df_aux_bkp = df_aux.copy()


# In[58]:


df_aux['Local'] = df_aux['Local'].rank(method='dense', ascending=False).astype(int)


# In[59]:


sc = MinMaxScaler(feature_range=(0,1))


# In[60]:


df_aux = sc.fit_transform(df_aux[['Mes', 'Local', 'Precipitacao']])


# In[61]:


df_aux


# In[91]:


cluster = KMeans(n_clusters=3, random_state=42).fit(df_aux)


# In[92]:


df_cluster = df_aux_bkp.copy()


# In[93]:


df_cluster['Cluster'] = cluster.labels_


# In[108]:


df_cluster.groupby(['Cluster', 'Local', 'Ano', 'Mes', 'Dia']).sum().reset_index()


# In[111]:


df_cluster.groupby(['Cluster', 'Local', 'Ano', 'Mes', 'Dia']).sum().reset_index().plot.scatter('Cluster', 'Local')


# In[96]:


#df_cluster.sample(n=100000).plot.scatter('Mes', 'Precipitacao')

