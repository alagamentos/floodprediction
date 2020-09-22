#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import plotly as py
from plotly import graph_objects as go
from plotly import express as px
import seaborn as sns

import os
import xgboost

from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, accuracy_score, f1_score, confusion_matrix, recall_score, precision_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.utils import resample
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


df_full = pd.read_csv('../../../data/cleandata/Info pluviometricas/Merged Data/full_data.csv', sep = ';')
df_full.head(10)


# In[ ]:


#df_prep = pd.read_csv('../../../data/cleandata/Info pluviometricas/Merged Data/prepped_data.csv', sep = ';')
#df_prep.head()


# # Data adjusts

# In[ ]:


df_full['Data_Hora'] = pd.to_datetime(df_full['Data_Hora'], yearfirst=True)


# In[ ]:


estacoes = {
    'Camilopolis': 1,
    'Erasmo': 2,
    'Paraiso': 3,
    'RM': 4,
    'Vitoria': 5
}


# In[ ]:


df = df_full.replace(estacoes)
df['Mes'] = df['Data_Hora'].dt.month
df.head(10)


# # Data analysis

# In[ ]:


columns = [
    'UmidadeRelativa',
    'PressaoAtmosferica',
    'TemperaturaDoAr',
    'TemperaturaInterna',
    'PontoDeOrvalho',
    'RadiacaoSolar',
    'DirecaoDoVento',
    'VelocidadeDoVento',
    'Precipitacao'
]


# In[ ]:


df[columns].isna().sum()


# In[ ]:


df[columns].describe()


# In[ ]:


df[columns].corr()


# In[ ]:


for column in columns:
    fig = px.box(df, x='Mes', y=column)
    fig.write_image(f'../../../images/Boxplot_{column}.png')


# In[ ]:


#df.iloc[np.unravel_index(x_scaled.argmax(), np.array(x_scaled).shape)] # Acha o dado de quanto ocorreu o maior valor


# # Global Configs

# In[ ]:


def remove_outliers(df):
    df.loc[df['DirecaoDoVento'] < 0, 'DirecaoDoVento'] = 0
    df.loc[df['DirecaoDoVento'] > 359, 'DirecaoDoVento'] = 359

    df.loc[df['PontoDeOrvalho'] > df['PontoDeOrvalho'].quantile(0.9999), 'PontoDeOrvalho'] = df['PontoDeOrvalho'].quantile(0.999903)

    df.loc[df['PressaoAtmosferica'] < df['PressaoAtmosferica'].quantile(0.1), 'PressaoAtmosferica'] = df['PressaoAtmosferica'].quantile(0.1)
    df.loc[df['PressaoAtmosferica'] > df['PressaoAtmosferica'].quantile(0.999999), 'PressaoAtmosferica'] = df['PressaoAtmosferica'].quantile(0.999999)

    df.loc[df['RadiacaoSolar'] < 0, 'RadiacaoSolar'] = 0
    df.loc[df['RadiacaoSolar'] > df['RadiacaoSolar'].quantile(0.88), 'RadiacaoSolar'] = df['RadiacaoSolar'].quantile(0.88)

    df.loc[df['UmidadeRelativa'] < df['UmidadeRelativa'].quantile(0.15), 'UmidadeRelativa'] = df['UmidadeRelativa'].quantile(0.15)
    df.loc[df['UmidadeRelativa'] > 100, 'UmidadeRelativa'] = 100

    df.loc[df['VelocidadeDoVento'] < 0, 'VelocidadeDoVento'] = 0
    df.loc[df['VelocidadeDoVento'] > df['VelocidadeDoVento'].quantile(0.99999), 'VelocidadeDoVento'] = df['VelocidadeDoVento'].quantile(0.999999)
    
    #df.loc[df['TemperaturaInterna'] > df['TemperaturaInterna'].quantile(0.999999), 'TemperaturaInterna'] = df['TemperaturaInterna'].quantile(0.999999)


# # PCA

# ## Configs

# In[ ]:


N_explained_variance = 5
drop_columns = ['Data_Hora', 'TemperaturaInterna', 'Mes'] + [x for x in df.columns if 'Local' in x]


# ## Todas estações

# In[ ]:


print(f'\n***************Todas Estações***************\n'.upper())

df_scaler = df.drop(columns=drop_columns)

std_scaler = StandardScaler()
x_scaled = std_scaler.fit_transform(df_scaler)

pca = PCA()
pca_data = pca.fit_transform(x_scaled)

per_var = np.round(pca.explained_variance_ratio_*100, decimals=1)
labels = [f'PC{x}' for x in range(1, len(per_var)+1)]

plt.bar(x=range(1, len(per_var)+1), height=per_var, tick_label=labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.show()

print(f'\nExplained Variance Ratio dos {N_explained_variance} primeiros PCAs: {sum(pca.explained_variance_ratio_[:N_explained_variance])}\n')

print(f'PCAs - Todas estações:')

for i in range(len(pca.components_)):
    features = list(zip(df_scaler, pca.components_[i]))
    features.sort(key=lambda x: abs(x[1]))
    features.reverse()

    print(f'\nPCA {i+1}:')
    display(features)


# ## Por estação

# In[ ]:


for estacao in estacoes:
    print(f'\n***************Estação {estacao}***************\n'.upper())

    df_scaler = df[df['Local'] == estacoes[estacao]]
    df_scaler = df_scaler.drop(columns=drop_columns)

    std_scaler = StandardScaler()
    x_scaled = std_scaler.fit_transform(df_scaler)

    pca = PCA()
    pca_data = pca.fit_transform(x_scaled)

    per_var = np.round(pca.explained_variance_ratio_*100, decimals=1)
    labels = [f'PC{x}' for x in range(1, len(per_var)+1)]

    plt.bar(x=range(1, len(per_var)+1), height=per_var, tick_label=labels)
    plt.ylabel('Percentage of Explained Variance')
    plt.xlabel('Principal Component')
    plt.title('Scree Plot')
    plt.show()

    print(f'\nExplained Variance Ratio dos {N_explained_variance} primeiros PCAs: {sum(pca.explained_variance_ratio_[:N_explained_variance])}\n')

    print(f'Feature Importante por PCA')

    for i in range(len(pca.components_)):
        features = list(zip(df_scaler, pca.components_[i]))
        features.sort(key=lambda x: abs(x[1]))
        features.reverse()

        print(f'\nPCA {i+1}:')
        display(features)

    print('\n\n')


# ## Todas estações - Sem Outliers

# In[ ]:


print(f'\n***************Todas Estações***************\n'.upper())

df_scaler = df.drop(columns=drop_columns)

remove_outliers(df_scaler)

std_scaler = StandardScaler()
x_scaled = std_scaler.fit_transform(df_scaler)

pca = PCA()
pca_data = pca.fit_transform(x_scaled)

per_var = np.round(pca.explained_variance_ratio_*100, decimals=1)
labels = [f'PC{x}' for x in range(1, len(per_var)+1)]

plt.bar(x=range(1, len(per_var)+1), height=per_var, tick_label=labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.show()

print(f'\nExplained Variance Ratio dos {N_explained_variance} primeiros PCAs: {sum(pca.explained_variance_ratio_[:N_explained_variance])}\n')

print(f'PCAs - Todas estações:')

for i in range(len(pca.components_)):
    features = list(zip(df_scaler, pca.components_[i]))
    features.sort(key=lambda x: abs(x[1]))
    features.reverse()

    print(f'\nPCA {i+1}:')
    display(features)


# ## Por Estação - Sem Outliers

# In[ ]:


for estacao in estacoes:
    print(f'\n***************Estação {estacao}***************\n'.upper())

    df_scaler = df[df['Local'] == estacoes[estacao]]
    df_scaler = df_scaler.drop(columns=drop_columns)
    
    remove_outliers(df_scaler)

    std_scaler = StandardScaler()
    x_scaled = std_scaler.fit_transform(df_scaler)

    pca = PCA()
    pca_data = pca.fit_transform(x_scaled)

    per_var = np.round(pca.explained_variance_ratio_*100, decimals=1)
    labels = [f'PC{x}' for x in range(1, len(per_var)+1)]

    plt.bar(x=range(1, len(per_var)+1), height=per_var, tick_label=labels)
    plt.ylabel('Percentage of Explained Variance')
    plt.xlabel('Principal Component')
    plt.title('Scree Plot')
    plt.show()

    print(f'\nExplained Variance Ratio dos {N_explained_variance} primeiros PCAs: {sum(pca.explained_variance_ratio_[:N_explained_variance])}\n')

    print(f'Feature Importante por PCA')

    for i in range(len(pca.components_)):
        features = list(zip(df_scaler, pca.components_[i]))
        features.sort(key=lambda x: abs(x[1]))
        features.reverse()

        print(f'\nPCA {i+1}:')
        display(features)

    print('\n\n')


# # LDA

# ## Configs

# In[ ]:


label_column = 'LocalMax_h'
drop_columns = ['Data_Hora', 'TemperaturaInterna', 'Mes'] + [x for x in df.columns if 'Local' in x]


# ## Todas estações

# In[ ]:


print(f'\n***************Todas Estações***************\n'.upper())

df_aux = df.copy()

# separate minority and majority classes
not_ordem = df_aux[df_aux[label_column]==0].copy()
ordem = df_aux[df_aux[label_column]==1].copy()

# upsample minority
ordem_upsampled = resample(ordem,
                        replace=True, # sample with replacement
                        n_samples=len(not_ordem), # match number in majority class
                        random_state=42) # reproducible results

# combine majority and upsampled minority
df_upsampled = pd.concat([not_ordem, ordem_upsampled])

X = df_upsampled.drop(columns=drop_columns).values
y = df_upsampled[label_column].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

lda = LDA(n_components=1)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)

classifier = RandomForestClassifier(max_depth=5, random_state=42)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
confusion_matrix(y_test, y_pred)


# ## Todas estações - Sem Outliers

# In[ ]:


print(f'\n***************Todas Estações***************\n'.upper())

df_aux = df.copy()

remove_outliers(df_aux)

# separate minority and majority classes
not_ordem = df_aux[df_aux[label_column] == 0].copy()
ordem = df_aux[df_aux[label_column] == 1].copy()

# upsample minority
ordem_upsampled = resample(ordem,
                        replace=True, # sample with replacement
                        n_samples=len(not_ordem), # match number in majority class
                        random_state=42) # reproducible results

# combine majority and upsampled minority
df_upsampled = pd.concat([not_ordem, ordem_upsampled])

X = df_upsampled.drop(columns=drop_columns).values
y = df_upsampled[label_column].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

lda = LDA(n_components=1)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)

classifier = RandomForestClassifier(max_depth=2, random_state=42)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
confusion_matrix(y_test, y_pred)


# # Pair plot

# In[ ]:


columns = [
    'UmidadeRelativa',
    'PressaoAtmosferica',
    'TemperaturaDoAr',
    'PontoDeOrvalho',
    'RadiacaoSolar',
    'VelocidadeDoVento',
    'Precipitacao',
    'LocalMax_d'
]


# In[ ]:


df_aux = df[df['Local'] == 1].copy()
remove_outliers(df_aux)
sns.pairplot(df_aux[columns], hue='LocalMax_d')


# In[ ]:


df_aux = pd.DataFrame()
df_aux['LocalMax_d_All'] = df[0:292806:5]['LocalMax_d_All']

for i in range(1, 6):
    df_aux[f'Precipitacao_{i}'] = df.loc[df['Local'] == i, 'Precipitacao'].reset_index(drop=True)

sns.pairplot(df_aux, hue='LocalMax_d_All')


# In[ ]:


df_aux


# # Temperatura do Ar x Ponto de Orvalho

# In[ ]:


df_aux = df[df['Local'] == 4].copy()
df_aux = df_aux[0:70080]
#remove_outliers(df_aux)

df_aux['Diff_Temp_Orvalho'] = df_aux['TemperaturaDoAr'] - df_aux['PontoDeOrvalho']

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x = df_aux['Data_Hora'],
        y = df_aux['Precipitacao'],
        name = 'Precipitacao'
    )
)

fig.add_trace(
    go.Scatter(
        x = df_aux['Data_Hora'],
        y = df_aux['TemperaturaDoAr'],
        name = 'TemperaturaDoAr'
    )
)

fig.add_trace(
    go.Scatter(
        x = df_aux['Data_Hora'],
        y = df_aux['PontoDeOrvalho'],
        name = 'PontoDeOrvalho'
    )
)

fig.add_trace(
    go.Scatter(
        x = df_aux['Data_Hora'],
        y = df_aux['Diff_Temp_Orvalho'],
        name = 'Diff_Temp_Orvalho'
    )
)

fig.add_trace(
    go.Scatter(
        x = df_aux['Data_Hora'],
        y = df_aux['UmidadeRelativa'] - df_aux['UmidadeRelativa'].min(),
        name = 'UmidadeRelativa'
    )
)

fig.add_trace(
    go.Scatter(
        x = df_aux['Data_Hora'],
        y = df_aux['RadiacaoSolar'] / 50,
        name = 'RadiacaoSolar'
    )
)

fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(step="all")
        ])
    )
)
fig.update_xaxes(rangeslider_visible=True)


if not os.path.exists("../../../assets"):
    os.mkdir("../../../assets")

fig.write_html('../../../assets/temp_orvalho.html')

