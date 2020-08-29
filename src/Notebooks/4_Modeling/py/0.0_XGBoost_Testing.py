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
from sklearn.metrics import plot_confusion_matrix, accuracy_score, f1_score, confusion_matrix, recall_score, precision_score

from sklearn.utils import resample


# # 1 - Classificar o dataset clusterizado por dia

# In[ ]:


df_cluster = pd.read_csv('../../../data/cleandata/Info pluviometricas/Merged Data/clustered_data.csv', sep = ';')
df_cluster


# In[ ]:


df_cluster['Ordens'] = df_cluster['OrdensServico']
df_cluster.loc[df_cluster['Ordens'] >= 1, 'Ordens'] = 1
df_cluster = df_cluster.drop(columns=['Label', 'OrdensServico', 'LocalMax'])


# In[ ]:


xgb = xgboost.XGBClassifier()

cols_rem = ['Ordens', 'Cluster']

x = df_cluster[[c for c in df_cluster.columns if c not in cols_rem]]
y = df_cluster['Ordens']

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3, random_state = 378)

# concatenate our training data back together
X = pd.concat([x_treino, y_treino], axis=1)

# separate minority and majority classes
not_ordem = X[X['Ordens']==0].copy()
ordem = X[X['Ordens']==1].copy()

# upsample minority
ordem_upsampled = resample(ordem,
                        replace=True, # sample with replacement
                        n_samples=len(not_ordem), # match number in majority class
                        random_state=378) # reproducible results

# combine majority and upsampled minority
upsampled = pd.concat([not_ordem, ordem_upsampled])

x_treino = upsampled[[c for c in df_cluster.columns if c not in cols_rem]]
y_treino = upsampled['Ordens']

display(y_treino.value_counts())

#xgb.fit(x_treino, y_treino, eval_set = [(x_treino, y_treino), (x_teste, y_teste)], eval_metric=f1_score)
param = {'max_depth':10, 'eta':2, 'objective':'binary:logistic', 'min_child_weight': 1, 'lambda': 1, 'alpha': 0, 'gamma': 0}

df_train = xgboost.DMatrix(data=x_treino, label=y_treino)

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


df_cluster.groupby(['Ordens']).count()


# # 2 - Classificar o dataset clusterizado por 15 mins

# In[ ]:


df_clustered_total = pd.read_csv('../../../data/cleandata/Info pluviometricas/Merged Data/clustered.csv', sep = ';')
df_clustered_total


# In[ ]:


df_clustered_total['Ordens'] = df_clustered_total['LocalMax']
df_clustered_total.loc[df_clustered_total['Ordens'] >= 1, 'Ordens'] = 1
df_clustered_total = df_clustered_total.drop(columns=['Label', 'LocalMax', 'OrdensServico'])
df_clustered_total['Data'] = pd.to_datetime(df_clustered_total['Data_Hora'], yearfirst=True).dt.strftime('%Y-%m-%d')
df_clustered_total = df_clustered_total.sort_values(by=['Data_Hora', 'Local'])


# In[ ]:


df_clustered_total.groupby('Ordens').count()


# In[ ]:


# for i in range(24*4):
#     df_clustered_total['Prec_Shift_' + str(i)] = df_clustered_total['Precipitacao'].shift(-5 * (i+1), fill_value=0)

# df_clustered_total['PrecSum'] = df_clustered_total[[c for c in df_clustered_total.columns if 'Prec_Shift' in c]].sum(axis=1)
# df_clustered_total = df_clustered_total.drop(columns = [c for c in df_clustered_total.columns if 'Prec_Shift' in c])
# df_clustered_total.head(10)


# In[ ]:


df_prec_sum = df_clustered_total.groupby(['Data', 'Local']).sum().reset_index()[['Data', 'Local', 'Precipitacao']]
df_prec_sum.columns = ['Data', 'Local', 'PrecSum']
df_clustered_total = df_clustered_total.merge(df_prec_sum, on=['Data', 'Local'])
df_clustered_total.head(10)


# In[ ]:


df_clustered_total['Hora'] = pd.to_datetime(df_clustered_total['Data_Hora'], yearfirst=True).dt.hour

df_ohe = df_clustered_total.groupby(['Data', 'Local', 'Hora']).sum().reset_index()[['Data', 'Local', 'Hora', 'Precipitacao']]
s_ohe = df_ohe['Hora']
df_ohe = pd.get_dummies(df_ohe, columns = ['Hora'])
df_ohe['Hora'] = s_ohe

for i in range(24):
    df_ohe.loc[df_ohe['Hora_' + str(i)] == 1, 'Hora_' + str(i)] = df_ohe.loc[df_ohe['Hora_' + str(i)] == 1, 'Precipitacao']

df_clustered_total = df_clustered_total.merge(df_ohe[['Data', 'Local'] + [c for c in df_ohe.columns if 'Hora' in c]], on=['Data', 'Local', 'Hora'])


# In[ ]:


df_clustered_total['Local'] = df_clustered_total['Local'].rank(method='dense', ascending=False).astype(int)
df_clustered_total_a = df_clustered_total.copy()
df_clustered_total = df_clustered_total.drop(columns = ['Data_Hora', 'Data', 'Hora'])


# In[ ]:


# df_clustered_total['Local'] = df_clustered_total['Local'].rank(method='dense', ascending=False).astype(int)
# df_clustered_total['Minuto'] = pd.to_datetime(df_clustered_total['Data_Hora'], yearfirst=True).dt.minute
# df_clustered_total = df_clustered_total[df_clustered_total['Minuto'] == 0]
# df_clustered_total = df_clustered_total.drop(columns = ['Data_Hora', 'Minuto'])


# In[ ]:


#df_slice = df_clustered_total[(df_clustered_total['Ordens'] == 1) | (df_clustered_total['Cluster'].isin([1,2]))]
#df_slice = df_clustered_total[(df_clustered_total['Ordens'] == 1) | (df_clustered_total['Precipitacao'] > 10)]
df_slice = df_clustered_total.copy()
#df_slice = df_clustered_total[(df_clustered_total['Cluster'].isin([0]))]
#df_slice.loc[df_slice['Cluster'] == 0, 'Ordens'] = 0
df_slice.loc[(df_slice['Ordens'] == 1) & (df_slice['PrecSum'] <= 10), 'Ordens'] = 0
#df_slice.loc[(df_slice['Ordens'] == 1) & ~((df_clustered_total[[c for c in df_clustered_total.columns if 'Hora_' in c]] <= 20).sum(axis = 1) < 24), 'Ordens'] = 0


# In[ ]:


df_slice.groupby('Ordens').count()


# In[ ]:


# Testar prever cluster

xgb = xgboost.XGBClassifier()

cols_rem = ['Ordens', 'Cluster'] + [c for c in df_slice.columns if 'Hora_' in c]

x = df_slice[[c for c in df_slice.columns if c not in cols_rem]]
#x = x.drop(columns = 'Cluster')
y = df_slice['Ordens']

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3, random_state = 378, stratify=y)

# concatenate our training data back together
X = pd.concat([x_treino, y_treino], axis=1)

# separate minority and majority classes
not_ordem = X[X['Ordens']==0].copy()
ordem = X[X['Ordens']==1].copy()

# upsample minority
ordem_upsampled = resample(ordem,
                        replace=True, # sample with replacement
                        n_samples=len(not_ordem), # match number in majority class
                        random_state=378) # reproducible results

# combine majority and upsampled minority
upsampled = pd.concat([not_ordem, ordem_upsampled])

x_treino = upsampled[[c for c in df_slice.columns if c not in cols_rem]]
y_treino = upsampled['Ordens']

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


confusion_matrix(y_treino, y_treino_pred, )

# Treino: 0.9995160230081979
# Teste: 0.9987405226034058
# F1: 0.48172446110590444
# array([[0.99882856, 0.00117144],
#        [0.13175676, 0.86824324]])
# array([[438261,    514],
#        [    39,    257]], dtype=int64)


# In[ ]:


cols_rem = ['Cluster'] + [c for c in df_slice.columns if 'Hora_' in c]
df_other = df_clustered_total.loc[(df_clustered_total['Ordens'] == 1) & (df_clustered_total['PrecSum'] <= 10), [c for c in df_slice.columns if c not in cols_rem]]
x_other = df_other.drop(columns = 'Ordens')
y_other = df_other['Ordens']
y_other_pred = bst.predict(xgboost.DMatrix(data=x_other, label=y_other))
y_other_pred
# y_other_pred = [1 if i>0.5 else 0 for i in y_other_pred]
# print(f"Acurácia: {accuracy_score(y_other, y_other_pred)}")
# print(f"Precisão: {precision_score(y_other, y_other_pred)}")
# print(f"Recall: {recall_score(y_other, y_other_pred)}")
# print(f"F1: {f1_score(y_other, y_other_pred)}")
# display(confusion_matrix(y_other, y_other_pred, normalize='true'))
# display(confusion_matrix(y_other, y_other_pred,))


# In[ ]:


df_clustered_total_a['Ordens_New'] = df_clustered_total_a['Ordens']
df_clustered_total_a.loc[(df_clustered_total_a['Ordens_New'] == 1) & (df_clustered_total_a['PrecSum'] <= 10), 'Ordens_New'] = 0
df_clustered_total_a['Data_Hora'] = pd.to_datetime(df_clustered_total_a['Data_Hora'], yearfirst=True)


# In[ ]:


from plotly.subplots import make_subplots
fig = make_subplots(3,1, shared_xaxes=True )

precipitacao_cols = [c for c in df_clustered_total_a.columns if 'Precipitacao' in c]

ano = 2014
df_ano = df_clustered_total_a[df_clustered_total_a['Data_Hora'].dt.year == ano].groupby(['Data', 'Local']).sum().reset_index()
df_ano.loc[df_ano['Ordens'] >= 1, 'Ordens'] = 1
df_ano.loc[df_ano['Ordens_New'] >= 1, 'Ordens_New'] = 1

display(df_ano['Ordens'].value_counts())
display(df_ano['Ordens_New'].value_counts())

for col in precipitacao_cols:
    fig.add_trace(go.Bar(
        x = df_ano['Data'],
        y = df_ano[col],
        name = col,
                            ),
                  row = 1, col = 1
                 )
fig.add_trace(go.Bar(
    x = df_ano['Data'],
    y = df_ano['Ordens'],
    name = 'Ordens de Serviço',
                        ),
                  row = 2, col = 1
             )

fig.add_trace(go.Bar(
    x = df_ano['Data'],
    y = df_ano['Ordens_New'],
    name = 'Label',
                        ),
                  row = 3, col = 1
             )
fig.show()


# In[ ]:


display(df_clustered_total_a['Ordens'].value_counts())
display(df_clustered_total_a['Ordens_New'].value_counts())


# In[ ]:


df_a = x_teste.copy()
df_a['Label'] = y_teste
df_a['Label_Pred'] = y_teste_pred
df_a


# In[ ]:


df_a[(df_a['Label'] == 1) & (df_a['Label_Pred'] == 0)]


# In[ ]:


df_slice.groupby(['Ordens']).count()


# In[ ]:


y_teste.shape[0] / y.shape[0]


# In[ ]:


df_slice


# In[ ]:


# import pandas_gbq
# from google.oauth2 import service_account

# PROJECT_ID = 'temporal-285820'
# TABLE_clustered = 'info_pluviometrica.clustered_slice'

# CREDENTIALS = service_account.Credentials.from_service_account_file('../../../key/temporal-285820-cde76c259484.json')
# pandas_gbq.context.credentials = CREDENTIALS


# In[ ]:


# pandas_gbq.to_gbq(df_slice, TABLE_clustered, project_id=PROJECT_ID, credentials=CREDENTIALS, if_exists='replace')
# print('clustered done!')


# In[ ]:


df_clustered_total[(df_clustered_total['Cluster'].isin([1,2]))].groupby('Ordens').count()


# # 3 - Tentar prever cluster

# In[ ]:


#df_slice = df_clustered_total[(df_clustered_total['Ordens'] == 1) | (df_clustered_total['Cluster'].isin([1,2]))]
df_slice = df_clustered_total[(df_clustered_total['Ordens'] == 1) | (df_clustered_total['PrecSum'] > 10)]
#df_slice = df_clustered_total.copy()
#df_slice.loc[df_slice['Cluster'] == 0, 'Ordens'] = 0
df_slice.loc[(df_slice['Ordens'] == 1) & (df_slice['PrecSum'] <= 10), 'Ordens'] = 0


# In[ ]:


df_slice['Cluster'].value_counts()


# In[ ]:


# Testar prever cluster

xgb = xgboost.XGBClassifier()

cols_rem = ['Ordens', 'Cluster']

x = df_slice[[c for c in df_slice.columns if c not in cols_rem]]
#x = x.drop(columns = 'Cluster')
y = df_slice['Cluster']

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3, random_state = 378, stratify=y)

# concatenate our training data back together
X = pd.concat([x_treino, y_treino], axis=1)

# separate minority and majority classes
zero = X[X['Cluster']==0].copy()
one = X[X['Cluster']==1].copy()
# two = X[X['Cluster']==2].copy()
# three = X[X['Cluster']==3].copy()

# upsample minority
# three_upsampled = resample(three,
#                         replace=True, # sample with replacement
#                         n_samples=len(zero), # match number in majority class
#                         random_state=378) # reproducible results

# # upsample minority
# two_upsampled = resample(two,
#                         replace=True, # sample with replacement
#                         n_samples=len(zero), # match number in majority class
#                         random_state=378) # reproducible results

# upsample minority
one_upsampled = resample(one,
                        replace=True, # sample with replacement
                        n_samples=len(zero), # match number in majority class
                        random_state=378) # reproducible results

# combine majority and upsampled minority
#upsampled = pd.concat([zero, one_upsampled, two_upsampled, three_upsampled])
upsampled = pd.concat([zero, one_upsampled])

x_treino = upsampled[[c for c in df_slice.columns if c not in cols_rem]]
y_treino = upsampled['Cluster']

display(y_treino.value_counts())

#xgb.fit(x_treino, y_treino, eval_set = [(x_treino, y_treino), (x_teste, y_teste)], eval_metric=f1_score)
param = {'max_depth':50, 'eta':1, 'objective':'multi:softmax', 'num_class': 4, 'min_child_weight': 1, 'lambda': 1, 'alpha': 0, 'gamma': 0}

df_train = xgboost.DMatrix(data=x_treino, label=y_treino)
df_test = xgboost.DMatrix(data=x_teste, label=y_teste)

bst = xgboost.train(param, df_train, 2, feval=f1_score)
y_teste_pred = bst.predict(xgboost.DMatrix(data=x_teste, label=y_teste))
#y_teste_pred = [1 if i>0.5 else 0 for i in y_teste_pred]
y_treino_pred = bst.predict(xgboost.DMatrix(data=x_treino, label=y_treino))
#y_treino_pred = [1 if i>0.5 else 0 for i in y_treino_pred]

# print(f"Treino: {accuracy_score(y_treino, y_treino_pred)}")
# print(f"Teste: {accuracy_score(y_teste, y_teste_pred)}")
# print(f"Precisão: {precision_score(y_teste, y_teste_pred, average='macro')}")
# print(f"Recall: {recall_score(y_teste, y_teste_pred, average='macro')}")
# print(f"F1: {f1_score(y_teste, y_teste_pred, average='macro')}")
# display(confusion_matrix(y_teste, y_teste_pred, normalize='true'))
# display(confusion_matrix(y_teste, y_teste_pred,))

print(f"Treino: {accuracy_score(y_treino, y_treino_pred)}")
print(f"Teste: {accuracy_score(y_teste, y_teste_pred)}")
print(f"Precisão: {precision_score(y_teste, y_teste_pred)}")
print(f"Recall: {recall_score(y_teste, y_teste_pred)}")
print(f"F1: {f1_score(y_teste, y_teste_pred)}")
display(confusion_matrix(y_teste, y_teste_pred, normalize='true'))
display(confusion_matrix(y_teste, y_teste_pred,))


# In[ ]:


df_m = pd.read_csv('../../../data/cleandata/Info pluviometricas/Merged Data/merged.csv', sep = ';')
df_m['Data_Hora'] = pd.to_datetime(df_m['Data_Hora'], yearfirst=True)
df_m['Data'] = df_m['Data_Hora'].dt.strftime('%Y-%m-%d')


# In[ ]:


df_m.groupby('Data').sum()[[c for c in df_m.columns if 'Precipitacao' in c]].reset_index()


# In[ ]:


df_clustered_total.groupby('Cluster').sum()

