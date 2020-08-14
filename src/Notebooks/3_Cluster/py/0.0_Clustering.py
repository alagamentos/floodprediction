#!/usr/bin/env python
# coding: utf-8

# In[68]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

# Pandas Config
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


# In[69]:


def reverse_ohe(df, features, ignoredFeatures, featuresLength, prefix, suffix = ''):
    all_features = ignoredFeatures + features
    df_pivot = pd.DataFrame(columns = all_features)

    for i in range(featuresLength):
        cols_aux = [f'{feature}{prefix}{i}{suffix}' for feature in features]
        df_aux = df[ignoredFeatures + cols_aux].copy()
        df_aux.columns = all_features
        df_pivot = pd.concat([df_pivot, df_aux])

    return df_pivot.sort_values(by='Data_Hora').copy()


# In[70]:


merged = pd.read_csv('../../../data/cleandata/Info pluviometricas/Merged Data/merged.csv',
                sep=';',
                dtype = {'Local_0': object, 'Local_1':object,
                          'Local_2':object,  'Local_3':object})

merged.head()


# In[71]:


merged[['Precipitacao_0', 'Precipitacao_1', 'Precipitacao_2', 'Precipitacao_3', 'Precipitacao_4']].corr()


# In[72]:


regions = pd.read_csv('../../../data/cleandata/Info pluviometricas/Merged Data/error_regions.csv', sep=';')

regions.head()


# In[73]:


df = merged.merge(regions, on = 'Data_Hora')

df.head()


# In[74]:


orders = pd.read_csv('../../../data/cleandata/Ordens de serviço/Enchentes_LatLong.csv', sep=';')

orders.head()


# In[75]:


orders[orders['Data'] == '07/01/2011']


# In[76]:


df[df['Data'] == '07/01/11'][['Hora', 'Precipitacao_0']]


# In[77]:


orders[orders['Data'] == '15/04/2018']


# In[78]:


df[df['Data'] == '15/04/18'][['Hora', 'Precipitacao_0']]


# In[79]:


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


# In[80]:


repaired = pd.read_csv('../../../data/cleandata/Info pluviometricas/Merged Data/repaired.csv', sep=';')

repaired.head()


# In[81]:


df_ungrouped = merged.copy()

for i in range(5):
    for feature in features[1:-1]: # Tira Local e Precipitacao
        df_ungrouped[f'{feature}_{i}'] = repaired[f'{feature}_{i}_pred']

df_ungrouped.head()


# In[82]:


df_grouped = reverse_ohe(df_ungrouped, features, ignoredFeatures, 5, '_')
df_grouped.head()


# In[83]:


df_grouped['Data_Hora'] = pd.to_datetime(df_grouped['Data_Hora'])


# In[84]:


df_cluster = df_grouped[['Data_Hora', 'Local', 'Precipitacao', 'UmidadeRelativa', 'RadiacaoSolar']].copy()


# In[85]:


df_cluster.isna().sum()


# In[86]:


df_cluster.dropna(inplace=True)


# In[87]:


df_cluster.isna().sum()


# In[88]:


df_cluster['Ano'] = df_cluster['Data_Hora'].dt.year
df_cluster['Mes'] = df_cluster['Data_Hora'].dt.month
df_cluster['Dia'] = df_cluster['Data_Hora'].dt.day


# In[89]:


#df_cluster['Local'] = df_cluster['Local'].rank(method='dense', ascending=False).astype(int)


# In[90]:


df_cluster = df_cluster.groupby(['Local', 'Ano', 'Mes', 'Dia']).sum().reset_index()


# In[91]:


sc = MinMaxScaler(feature_range=(0,1))
df_norm = sc.fit_transform(df_cluster[['Precipitacao', 'UmidadeRelativa', 'RadiacaoSolar']])
df_norm


# In[93]:


cluster = KMeans(n_clusters=4, random_state=42).fit(df_norm)


# In[94]:


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


# In[27]:


df_cluster['Cluster'] = cluster.labels_
df_cluster


# In[28]:


df_cluster.groupby(['Cluster', 'Local'])[['Cluster']].count()


# In[29]:


df_cluster.groupby(['Cluster', 'Mes'])[['Cluster']].count()


# In[30]:


fig = px.bar(df_cluster.groupby(['Cluster', 'Local'])[['Mes']].count().reset_index(),
             x="Cluster", y="Mes", color="Local", barmode="group")
fig.show()


# In[31]:


fig = px.bar(df_cluster.groupby(['Cluster', 'Local'])[['Mes']].count().reset_index(),
             x="Local", y="Mes", color="Cluster", barmode="group")
fig.show()


# In[32]:


cols = [c for c in merged.columns if 'UmidadeRelativa' in c]
merged[cols].corr(method='spearman')


# In[33]:


cols = [c for c in df_ungrouped.columns if 'UmidadeRelativa' in c]
df_ungrouped[cols].corr(method='spearman')


# In[34]:


regions[[c for c in regions.columns if 'Data_Hora' not in c]].sum() / regions.shape[0] * 100


# # PCA + KMeans

# In[37]:


df_pca = df_grouped.sample(n=100000).copy()
df_pca = df_pca[[c for c in df_pca.columns if c not in ['SensacaoTermica']]].copy().dropna()
df_pca['Mes'] = df_pca['Data_Hora'].dt.month
df_pca['Local'] = df_pca['Local'].rank(method='dense', ascending=False).astype(int)
df_pca = df_pca.drop(columns = ['Data_Hora'])
df_pca


# In[38]:


X_std = StandardScaler().fit_transform(df_pca)


# In[39]:


pca = PCA(n_components=11)
principalComponents = pca.fit_transform(X_std)


# In[40]:


features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_ratio_, color='black')
plt.xlabel('PCA features')
plt.ylabel('variance %')
plt.xticks(features)


# In[184]:


# Save components to a DataFrame
PCA_components = pd.DataFrame(principalComponents)
plt.scatter(PCA_components[0], PCA_components[1], alpha=.1, color='black')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')


# In[42]:


ks = range(1, 10)
inertias = []
for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)
    
    # Fit model to samples
    model.fit(PCA_components.iloc[:,:3])
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
plt.plot(ks, inertias, '-o', color='black')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()


# # PCA + KMeans (sem aplicar reverse_ohe)

# In[62]:


df_pca = df_ungrouped.sample(n=100000).copy()
df_pca = df_pca[[c for c in df_pca.columns if c not in ['index', 'Data', 'Hora', 'SensacaoTermica'] and 'Local' not in c]].copy()#.dropna()
# df_pca['Mes'] = df_pca['Data_Hora'].dt.month
# df_pca['Local'] = df_pca['Local'].rank(method='dense', ascending=False).astype(int)
# df_pca = df_pca.drop(columns = ['Data_Hora'])
for column in df_pca[[c for c in df_pca.columns if 'Data_Hora' not in c]].columns.values:
    df_pca.loc[df_pca[column].isna(), column] = df_pca[column].mean()

df_pca = df_pca.set_index('Data_Hora')
df_pca


# In[63]:


X_std = StandardScaler().fit_transform(df_pca)
pca = PCA(n_components=20)
principalComponents = pca.fit_transform(X_std)


# In[64]:


features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_ratio_, color='black')
plt.xlabel('PCA features')
plt.ylabel('variance %')
plt.xticks(features)


# In[183]:


# Save components to a DataFrame
PCA_components = pd.DataFrame(principalComponents)
plt.scatter(PCA_components[0], PCA_components[1], alpha=.1, color='black')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')


# In[67]:


ks = range(1, 10)
inertias = []
for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)
    
    # Fit model to samples
    model.fit(PCA_components.iloc[:,:7])
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
plt.plot(ks, inertias, '-o', color='black')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()


# # Testando clusterização com apenas a precipitação
# Objetivo: Forçar a distinção de "tipos de chuva" para detectar chuvas mais fortes
# 
# Testando clusterizações com a precipitação e outras features, independente da feature selecionada, a precipitação não tem influência suficiente e a clusterização acaba sendo realizada em função das outras features, completamente ignorando a tendência da precipitação.

# In[164]:


#df_prec = df_grouped[df_grouped['Precipitacao'] > 0]
df_prec = df_grouped.copy()
df_prec['Ano'] = df_prec['Data_Hora'].dt.year
df_prec['Mes'] = df_prec['Data_Hora'].dt.month
df_prec['Dia'] = df_prec['Data_Hora'].dt.day
df_prec = df_prec.drop(columns = ['Data_Hora'])
s_prec = df_prec.groupby(['Local', 'Ano', 'Mes', 'Dia']).sum().reset_index()['Precipitacao']
df_prec = df_prec.groupby(['Local', 'Ano', 'Mes', 'Dia']).mean().reset_index()
df_prec['Precipitacao'] = s_prec
df_prec = df_prec[[c for c in df_prec.columns if c not in ['SensacaoTermica']]].copy().dropna()
df_prec['Local'] = df_prec['Local'].rank(method='dense', ascending=False).astype(int)
df_prec


# In[165]:


sc = MinMaxScaler(feature_range=(0,1))
df_norm = sc.fit_transform(df_prec[['Precipitacao']])
df_norm


# In[166]:


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


# In[177]:


cluster = KMeans(n_clusters=4, random_state=42).fit(df_norm)
df_prec['Cluster'] = cluster.labels_


# In[178]:


fig = px.bar(df_prec.groupby(['Cluster', 'Local'])[['Mes']].count().reset_index(),
             x="Cluster", y="Mes", color="Local", barmode="group")
fig.show()


# In[182]:


df_prec.groupby('Cluster').min()


# In[180]:


fig = px.bar(df_prec.groupby(['Cluster'])[['Precipitacao']].mean().reset_index(),
             x="Cluster", y="Precipitacao", barmode="group")
fig.show()


# In[181]:


df_prec.groupby('Cluster').count()


# In[ ]:




