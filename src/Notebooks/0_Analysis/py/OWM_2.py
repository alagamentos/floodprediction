#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from plotly import graph_objects as go
import plotly as py

from datetime import datetime
from datetime import timedelta

import xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, accuracy_score, f1_score, confusion_matrix, recall_score, precision_score

from sklearn.utils import resample


# In[ ]:


df_owm = pd.read_csv('../../../data/cleandata/Info pluviometricas/Merged Data/OpenWeatherMapSantoAndre.csv')
#df_est = pd.read_csv('../../../ordens.csv', sep=';')


# In[ ]:


df_owm.columns


# In[ ]:


df_owm['wind_speed']


# In[ ]:


df_owm['Data_Hora'] = pd.to_datetime(df_owm['dt_iso'].str[:-10])
df_owm['Data_Hora'] = df_owm.apply(lambda x: x['Data_Hora'] + pd.Timedelta(hours = x['timezone'] / 3600), axis = 1)
df_owm = df_owm[(datetime.strptime('2019-08-30', '%Y-%m-%d') >= df_owm['Data_Hora']) & (df_owm['Data_Hora'] >= datetime.strptime('2010-01-01', '%Y-%m-%d'))]
df_owm = df_owm.drop(columns = ['sea_level', 'grnd_level', 'rain_3h', 'snow_1h', 'snow_3h'])
df_owm = df_owm.fillna(0)
df_owm = df_owm.drop_duplicates(subset='Data_Hora')


# In[ ]:


df_est['Data'] = pd.to_datetime(df_est['Data'], yearfirst=True)
df_est.head()


# In[ ]:


df_loc = pd.read_csv('../../../data/cleandata/Ordens de serviço/labels_day.csv', sep=';')
df_loc['Data'] = pd.to_datetime(df_loc['Data'], yearfirst=True)
df_loc = df_loc[['Data', 'LocalMax']]
df_loc.columns = ['Data', 'Label']
df_loc.head()


# In[ ]:


print(df_est['Data'].min())
print(df_est['Data'].max())


# In[ ]:


# df_est = pd.read_csv('../../../data/cleandata/Ordens de serviço/Ordens_Label.csv', sep = ';')
# df_est['Data'] = pd.to_datetime(df_est['Data'], yearfirst=True)
# df_est = df_est[['Data', 'LocalMax']]
# df_est.columns = ['Data', 'Label']
# df_owm['Data'] = pd.to_datetime(df_owm['Data_Hora'].dt.strftime('%Y-%m-%d'), yearfirst=True)
# df = df_owm.merge(df_est, on='Data', how='left')
# df = df.fillna(0)


# In[ ]:


df_owm['Data'] = pd.to_datetime(df_owm['Data_Hora'].dt.strftime('%Y-%m-%d'), yearfirst=True)
df = df_owm.merge(df_loc, on='Data', how='left')
df = df.fillna(0)
#df['Label'] = df.apply(lambda x: 1 if x['Vitoria'] + x['Erasmo'] + x['Paraiso'] + x['Null'] + x['RM'] + x['Camilopolis'] > 0 else 0, axis = 1)


# In[ ]:


df_g = df.groupby('Data').sum().reset_index()[['Data', 'rain_1h']]
df_g.columns = ['Data', 'rain_sum']
df = df.merge(df_g, on='Data')


# In[ ]:


df['Mes'] = df['Data_Hora'].dt.month
df['Hora'] = df['Data_Hora'].dt.hour
df['Dia'] = df['Data_Hora'].dt.day
# df = df.drop(columns = ['dt', 'dt_iso', 'timezone', 'city_name', 'lat', 'lon', 'weather_icon', 'weather_id', 'weather_main',
#                         'Vitoria', 'Erasmo', 'Paraiso', 'RM', 'Null', 'Camilopolis'])
df = df.drop(columns = ['dt', 'dt_iso', 'timezone', 'city_name', 'lat', 'lon', 'weather_icon', 'weather_id', 'weather_main'])
# df = df.drop(columns = ['dt', 'dt_iso', 'timezone', 'city_name', 'lat', 'lon', 'weather_icon', 'weather_id', 'weather_main',
#                         'Data_Hora', 'Data'])
df['weather_description'] = df['weather_description'].rank(method='dense', ascending=False).astype(int)


# In[ ]:


#df = df.groupby('Data').max()


# In[ ]:


df_slice = df.copy()
#df_slice = df_clustered_total[(df_clustered_total['Cluster'].isin([0]))]
#df_slice.loc[df_slice['Cluster'] == 0, 'Ordens'] = 0
df_slice.loc[(df_slice['Label'] == 1) & (df_slice['rain_sum'] <= 10), 'Label'] = 0


# In[ ]:


df_slice.groupby('Label').count()


# In[ ]:


xgb = xgboost.XGBClassifier()

cols_rem = ['Label', 'Data', 'Data_Hora'] #+ ['temp_min', 'temp_max', 'clouds_all', 'weather_description']
cols_rem = cols_rem + ['temp', 'feels_like', 'temp_min', 'temp_max', 'pressure', 'humidity', 'wind_speed', 'wind_deg', 'clouds_all', 'weather_description']

x = df_slice[[c for c in df_slice.columns if c not in cols_rem]]
print(x.columns)
#x = df_slice.drop(columns = 'Label')
#x = x.drop(columns = 'Cluster')
y = df_slice['Label']

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3, random_state = 378, stratify=y)

# concatenate our training data back together
X = pd.concat([x_treino, y_treino], axis=1)

# separate minority and majority classes
not_ordem = X[X['Label']==0].copy()
ordem = X[X['Label']==1].copy()

# upsample minority
ordem_upsampled = resample(ordem,
                        replace=True, # sample with replacement
                        n_samples=len(not_ordem), # match number in majority class
                        random_state=378) # reproducible results

# combine majority and upsampled minority
upsampled = pd.concat([not_ordem, ordem_upsampled])

x_treino = upsampled.drop(columns = 'Label')
y_treino = upsampled['Label']

display(y_treino.value_counts())

#xgb.fit(x_treino, y_treino, eval_set = [(x_treino, y_treino), (x_teste, y_teste)], eval_metric=f1_score)
param = {'max_depth':50, 'eta':1, 'objective':'binary:logistic', 'min_child_weight': 1, 'lambda': 1, 'alpha': 0, 'gamma': 0}

df_train = xgboost.DMatrix(data=x_treino, label=y_treino)
df_test = xgboost.DMatrix(data=x_teste, label=y_teste)

bst = xgboost.train(param, df_train, 2, feval=f1_score)
y_teste_pred = bst.predict(xgboost.DMatrix(data=x_teste, label=y_teste))
y_teste_pred = [1 if i>0.5 else 0 for i in y_teste_pred]
y_treino_pred = bst.predict(xgboost.DMatrix(data=x_treino, label=y_treino))
y_treino_pred = [1 if i>0.5 else 0 for i in y_treino_pred]

print(f"Treino: {accuracy_score(y_treino, y_treino_pred)}")
print(f"Teste: {accuracy_score(y_teste, y_teste_pred)}")
print(f"Precisão: {precision_score(y_teste, y_teste_pred)}")
print(f"Recall: {recall_score(y_teste, y_teste_pred)}")
print(f"F1: {f1_score(y_teste, y_teste_pred)}")
display(confusion_matrix(y_teste, y_teste_pred, normalize='true'))
display(confusion_matrix(y_teste, y_teste_pred,))


# In[ ]:


x_treino


# In[ ]:


x_a = x_teste.copy()
x_a['Label'] = y_teste
x_a['Label_Pred'] = y_teste_pred
x_a[x_a['Label_Pred'] == 1]


# In[ ]:


def getPrecMomento(x):
    return df_label_pred.loc[
        (df_label_pred['Data'] == x['Data']) &
        (df_label_pred['Data_Hora'] <= x['Data_Hora']), 'rain_1h'
    ].sum()

#df_cluster_hora['PrecMomento'] = df_cluster_hora.apply(getPrecMomento, axis=1)
#df_cluster_hora.head(100).apply(getPrecMomento, axis=1)


df_pred = bst.predict(xgboost.DMatrix(data=df_slice[['rain_1h', 'rain_sum', 'Mes', 'Hora', 'Dia']]))
df_pred = [1 if i>0.5 else 0 for i in df_pred]
df_slice['Label_Pred'] = df_pred
#df_slice.loc[df_slice['Label_Pred'] == 1, ['Data_Hora', 'rain_1h', 'rain_sum', 'Mes', 'Label', 'Label_Pred']]

df_label_pred = df_slice.loc[df_slice['Label_Pred'] == 1, ['Data_Hora', 'Data', 'Dia', 'Hora', 'rain_1h', 'rain_sum', 'Mes', 'Label', 'Label_Pred']].copy()
df_label_pred['rain_momento'] = df_label_pred.apply(getPrecMomento, axis=1)
df_label_pred.head(24)

df_pred = bst.predict(xgboost.DMatrix(data=df_label_pred[['rain_1h', 'rain_momento', 'Mes', 'Hora', 'Dia']].rename(columns={'rain_momento':'rain_sum'})))
df_pred = [1 if i>0.5 else 0 for i in df_pred]

df_label_pred['Label_Momento'] = df_pred


# In[ ]:


df_label_pred[['Data_Hora', 'rain_1h', 'rain_momento', 'rain_sum', 'Label', 'Label_Pred', 'Label_Momento']]


# In[ ]:


df_label_pred[['Data_Hora', 'rain_1h', 'rain_momento', 'rain_sum', 'Label_Pred', 'Label_Momento']].to_csv("labels_prediction.csv", index=False, sep=";", decimal=",")

