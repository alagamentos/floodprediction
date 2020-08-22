#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from plotly import graph_objects as go
import plotly as py

from datetime import datetime
from datetime import timedelta

import xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, accuracy_score, f1_score, confusion_matrix


# In[ ]:


df_cluster = pd.read_csv('../../../data/cleandata/Info pluviometricas/Merged Data/clustered_data.csv', sep = ';')
df_cluster


# In[ ]:


xgb = xgboost.XGBClassifier()

x = df_cluster[[c for c in df_cluster.columns if 'Ordens' not in c]]
y = df_cluster['Ordens']

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3, random_state = 378)

#xgb.fit(x_treino, y_treino, eval_set = [(x_treino, y_treino), (x_teste, y_teste)], eval_metric=f1_score)
param = {'max_depth':2, 'eta':1, 'objective':'binary:logistic'}

df_train = xgboost.DMatrix(data=x_treino, label=y_treino)

bst = xgboost.train(param, df_train, 2, feval=f1_score)
y_pred = bst.predict(xgboost.DMatrix(data=x_teste, label=y_teste))


# In[ ]:


y_pred = [1 if i>0.5 else 0 for i in y_pred]


# In[ ]:


print(accuracy_score(y_teste, y_pred))
print(f1_score(y_teste, y_pred))


# In[ ]:


df_clustered_total = pd.read_csv('../../../data/cleandata/Info pluviometricas/Merged Data/clustered.csv', sep = ';')
df_clustered_total


# In[ ]:


df_clustered_total['Local'] = df_clustered_total['Local'].rank(method='dense', ascending=False).astype(int)
df_clustered_total = df_clustered_total.drop(columns = 'Data_Hora')


# In[ ]:


df_slice = df_clustered_total[(df_clustered_total['Ordens'] == 1) | (df_clustered_total['Cluster'].isin([1,2]))]


# In[ ]:


df_slice.groupby('Ordens').count()


# In[ ]:


xgb = xgboost.XGBClassifier()

x = df_slice[[c for c in df_slice.columns if 'Ordens' not in c]]
y = df_slice['Ordens']

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3, random_state = 378)

#xgb.fit(x_treino, y_treino, eval_set = [(x_treino, y_treino), (x_teste, y_teste)], eval_metric=f1_score)
param = {'max_depth':2, 'eta':1, 'objective':'binary:logistic'}

df_train = xgboost.DMatrix(data=x_treino, label=y_treino)

bst = xgboost.train(param, df_train, 2, feval=f1_score)
y_pred = bst.predict(xgboost.DMatrix(data=x_teste, label=y_teste))
y_pred = [1 if i>0.5 else 0 for i in y_pred]


# In[ ]:


print(accuracy_score(y_teste, y_pred))
print(f1_score(y_teste, y_pred))
confusion_matrix(y_teste, y_pred, normalize='true')


# In[ ]:


y_teste.shape[0] / y.shape[0]


# In[ ]:


df_slice


# In[ ]:


import pandas_gbq
from google.oauth2 import service_account

PROJECT_ID = 'temporal-285820'
TABLE_clustered = 'info_pluviometrica.clustered_slice'

CREDENTIALS = service_account.Credentials.from_service_account_file('../../../key/temporal-285820-cde76c259484.json')
pandas_gbq.context.credentials = CREDENTIALS


# In[ ]:


pandas_gbq.to_gbq(df_slice, TABLE_clustered, project_id=PROJECT_ID, credentials=CREDENTIALS, if_exists='replace')
print('clustered done!')

