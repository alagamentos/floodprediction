#!/usr/bin/env python
# coding: utf-8

# # 0 - Inicialização

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

# ## Carregar dados

# In[ ]:


df_cluster = pd.read_csv('../../../data/cleandata/Info pluviometricas/Merged Data/clustered_label.csv', sep = ';')
df_cluster['Data_Hora'] = pd.to_datetime(df_cluster['Data_Hora'], yearfirst=True)
df_cluster['Data'] = df_cluster['Data_Hora'].dt.strftime('%Y-%m-%d')
df_cluster


# ## Agrupar por dia

# In[ ]:


# df_cluster_dia = df_cluster.drop(columns = ['Data_Hora', 'LocalMax_Hora']).rename(columns = {'LocalMax_Dia': 'LocalMax'})
# s_prec_p = df_cluster_dia.groupby(['Data', 'Local']).sum().reset_index().groupby('Data').mean()['Precipitacao'].reset_index()
# s_prec_o = df_cluster_dia.groupby(['Data', 'Local']).max().reset_index().groupby('Data').max()['LocalMax'].reset_index()
# df_cluster_dia = df_cluster_dia.groupby(['Data']).mean().reset_index()
# df_cluster_dia['Precipitacao'] = s_prec_p['Precipitacao']
# df_cluster_dia['LocalMax'] = s_prec_o['LocalMax']
# df_cluster_dia

df_cluster_dia = df_cluster.drop(columns = ['Data_Hora', 'LocalMax_Hora']).rename(columns = {'LocalMax_Dia': 'LocalMax'})
df_prec_p = df_cluster_dia.groupby(['Data', 'Local']).sum()['Precipitacao'].reset_index()
df_prec_o = df_cluster_dia.groupby(['Data', 'Local']).max()['LocalMax'].reset_index()
df_cluster_dia = df_cluster_dia.groupby(['Data', 'Local']).mean().reset_index().drop(columns = ['Precipitacao', 'LocalMax'])
df_cluster_dia = df_cluster_dia.merge(df_prec_p, on = ['Data', 'Local'])
df_cluster_dia = df_cluster_dia.merge(df_prec_o, on = ['Data', 'Local'])
df_cluster_dia['Local'] = df_cluster_dia['Local'].replace({'Camilopolis': 1, 'Erasmo': 2, 'Paraiso': 3, 'RM': 4, 'Vitoria': 5})
df_cluster_dia['Mes'] = df_cluster_dia['Data'].str[5:7].astype(int)
df_cluster_dia


# In[ ]:


xgb = xgboost.XGBClassifier()

cols_rem = ['LocalMax', 'Data']

x = df_cluster_dia[[c for c in df_cluster_dia.columns if c not in cols_rem]]
y = df_cluster_dia['LocalMax']

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3, random_state = 378)

# concatenate our training data back together
X = pd.concat([x_treino, y_treino], axis=1)

# separate minority and majority classes
not_ordem = X[X['LocalMax']==0].copy()
ordem = X[X['LocalMax']==1].copy()

# upsample minority
ordem_upsampled = resample(ordem,
                        replace=True, # sample with replacement
                        n_samples=len(not_ordem), # match number in majority class
                        random_state=378) # reproducible results

# combine majority and upsampled minority
upsampled = pd.concat([not_ordem, ordem_upsampled])

x_treino = upsampled[[c for c in df_cluster_dia.columns if c not in cols_rem]]
y_treino = upsampled['LocalMax']

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


# # 2 - Classificar o dataset clusterizado por 15 mins

# In[ ]:


df_label_d = pd.read_csv('../../../data/cleandata/Ordens de serviço/labels_day.csv', sep = ';')
df_cluster_hora = df_cluster.merge(df_label_d[['Data', 'LocalMax']], on = 'Data', how = 'left').fillna(0)
df_cluster_hora = df_cluster_hora.drop(columns=['LocalMax_Hora', 'LocalMax_Dia'])


# In[ ]:


#df_cluster_hora = df_cluster.copy()
#df_cluster_hora = df_cluster_hora.drop(columns=['LocalMax_Hora']).rename(columns = {'LocalMax_Dia': 'LocalMax'})
df_cluster_hora = df_cluster_hora.sort_values(by=['Data_Hora', 'Local'])


# In[ ]:


df_cluster_hora.groupby('LocalMax').count()


# Prec24h -> soma da precipitação nas últimas 24 horas

# In[ ]:


# for i in range(24*4):
#     df_cluster_hora['Prec_Shift_' + str(i)] = df_cluster_hora['Precipitacao'].shift(-5 * (i+1), fill_value=0)

# df_cluster_hora['Prec24h'] = df_cluster_hora[[c for c in df_cluster_hora.columns if 'Prec_Shift' in c]].sum(axis=1)
# df_cluster_hora = df_cluster_hora.drop(columns = [c for c in df_cluster_hora.columns if 'Prec_Shift' in c])
# df_cluster_hora.head(10)


# PrecMomento -> soma da precipitação até aquele momento do dia

# In[ ]:


init_time = datetime.now()
def getPrecMomento(x):
    return df_cluster_hora.loc[
        (df_cluster_hora['Data'] == x['Data']) &
        (df_cluster_hora['Data_Hora'] <= x['Data_Hora']) &
        (df_cluster_hora['Local'] == x['Local']), 'Precipitacao'
    ].sum()

#df_cluster_hora['PrecMomento'] = df_cluster_hora.apply(getPrecMomento, axis=1)
df_cluster_hora.head(100).apply(getPrecMomento, axis=1)
print(datetime.now() - init_time)


# In[ ]:


df_cluster_hora.shape


# PrecSum -> soma da precipitação no dia

# In[ ]:


df_prec_sum = df_cluster_hora.groupby(['Data', 'Local']).sum().reset_index()[['Data', 'Local', 'Precipitacao']]
df_prec_sum.columns = ['Data', 'Local', 'PrecSum']
df_cluster_hora = df_cluster_hora.merge(df_prec_sum, on=['Data', 'Local'])
df_cluster_hora.head(10)


# In[ ]:


# df_clustered_total['Hora'] = pd.to_datetime(df_clustered_total['Data_Hora'], yearfirst=True).dt.hour

# df_ohe = df_clustered_total.groupby(['Data', 'Local', 'Hora']).sum().reset_index()[['Data', 'Local', 'Hora', 'Precipitacao']]
# s_ohe = df_ohe['Hora']
# df_ohe = pd.get_dummies(df_ohe, columns = ['Hora'])
# df_ohe['Hora'] = s_ohe

# for i in range(24):
#     df_ohe.loc[df_ohe['Hora_' + str(i)] == 1, 'Hora_' + str(i)] = df_ohe.loc[df_ohe['Hora_' + str(i)] == 1, 'Precipitacao']

# df_clustered_total = df_clustered_total.merge(df_ohe[['Data', 'Local'] + [c for c in df_ohe.columns if 'Hora' in c]], on=['Data', 'Local', 'Hora'])


# In[ ]:


df_cluster_hora_a = df_cluster_hora.copy()


# In[ ]:


df_cluster_hora['Hora'] = df_cluster_hora['Data_Hora'].dt.hour
df_cluster_hora['Mes'] = df_cluster_hora['Data_Hora'].dt.month
df_cluster_hora['Local'] = df_cluster_hora['Local'].replace({'Camilopolis': 1, 'Erasmo': 2, 'Paraiso': 3, 'RM': 4, 'Vitoria': 5})


# Filtrando hora no minuto 0
df_hora = df_cluster_hora.groupby(['Data', 'Local', 'Hora']).sum().reset_index()[['Data', 'Local', 'Hora', 'Precipitacao']]
# df_clustered_total = df_clustered_total.groupby(['Data', 'Local', 'Hora']).mean().reset_index()
# s_prec = df_clustered_total.groupby(['Data', 'Local', 'Hora']).sum()[['Precipitacao']]
# df_clustered_total['Precipitacao'] = s_prec

df_cluster_hora['Minuto'] = df_cluster_hora['Data_Hora'].dt.minute
df_cluster_hora = df_cluster_hora[df_cluster_hora['Minuto'] == 0]
#df_cluster_hora = df_cluster_hora.drop(columns = ['Data_Hora', 'Minuto'])
# df_clustered_total = df_clustered_total.drop(columns = ['Data_Hora', 'Minuto', 'Precipitacao'])
# df_clustered_total = df_clustered_total.merge(df_hora, on=['Data', 'Local', 'Hora'])


# In[ ]:


#df_slice = df_clustered_total[(df_clustered_total['Ordens'] == 1) | (df_clustered_total['Cluster'].isin([1,2]))]
#df_slice = df_clustered_total[(df_clustered_total['Ordens'] == 1) | (df_clustered_total['PrecSum'] > 10)]
df_slice = df_cluster_hora.copy()
#df_slice = df_clustered_total[(df_clustered_total['Cluster'].isin([0]))]
#df_slice.loc[df_slice['Cluster'] == 0, 'Ordens'] = 0

df_slice.loc[(df_slice['LocalMax'] == 1) & (df_slice['PrecSum'] <= 10), 'LocalMax'] = 0

#df_slice.loc[(df_slice['Ordens'] == 1) & ~((df_clustered_total[[c for c in df_clustered_total.columns if 'Hora_' in c]] <= 20).sum(axis = 1) < 24), 'Ordens'] = 0
#df_slice = df_slice[df_slice['Local'] == 4]


# In[ ]:


df_slice.groupby('LocalMax').count()


# In[ ]:


df_slice.shape


# In[ ]:


for l in range(6):
    if l != 0:
        df_train = df_slice[df_slice['Local'] == l]
    else:
        df_train = df_slice.copy()
        
    print(f'----- LOCAL {l} -----')

    # Testar prever cluster

    xgb = xgboost.XGBClassifier()

    cols_rem = ['LocalMax', 'Cluster', 'Data', 'Hora', 'Data_Hora', 'Ordens', 'Minuto'] + [c for c in df_train.columns if 'Hora_' in c]
    #cols_rem = cols_rem + ['DirecaoDoVento', 'VelocidadeDoVento']
    #cols_rem = cols_rem + ['TemperaturaInterna', 'PontoDeOrvalho']

    x = df_train[[c for c in df_train.columns if c not in cols_rem]]
    if l == 0:
        print(x.columns)
    #x = x.drop(columns = 'Cluster')
    y = df_train['LocalMax']

    x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3, random_state = 378, stratify=y)

    # concatenate our training data back together
    X = pd.concat([x_treino, y_treino], axis=1)

    # separate minority and majority classes
    not_ordem = X[X['LocalMax']==0].copy()
    ordem = X[X['LocalMax']==1].copy()

    # upsample minority
    ordem_upsampled = resample(ordem,
                            replace=True, # sample with replacement
                            n_samples=len(not_ordem), # match number in majority class
                            random_state=378) # reproducible results

    # combine majority and upsampled minority
    upsampled = pd.concat([not_ordem, ordem_upsampled])

    x_treino = upsampled[[c for c in df_slice.columns if c not in cols_rem]]
    y_treino = upsampled['LocalMax']

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


test_cases = [
    [],
    ['DirecaoDoVento', 'VelocidadeDoVento'],
    ['DirecaoDoVento', 'VelocidadeDoVento', 'TemperaturaInterna'],
    ['DirecaoDoVento', 'VelocidadeDoVento', 'TemperaturaInterna', 'PontoDeOrvalho'],
    ['DirecaoDoVento', 'VelocidadeDoVento', 'TemperaturaInterna', 'PontoDeOrvalho', 'UmidadeRelativa'],
    ['DirecaoDoVento', 'VelocidadeDoVento', 'TemperaturaInterna', 'PontoDeOrvalho', 'UmidadeRelativa', 'PressaoAtmosferica'],
    ['DirecaoDoVento', 'VelocidadeDoVento', 'TemperaturaInterna', 'PontoDeOrvalho', 'UmidadeRelativa', 'PressaoAtmosferica', 'RadiacaoSolar'],
    ['DirecaoDoVento', 'VelocidadeDoVento', 'TemperaturaInterna', 'PontoDeOrvalho', 'UmidadeRelativa', 'PressaoAtmosferica', 'RadiacaoSolar', 'TemperaturaDoAr'],
    ['DirecaoDoVento', 'VelocidadeDoVento', 'TemperaturaInterna', 'PontoDeOrvalho', 'UmidadeRelativa', 'PressaoAtmosferica', 'RadiacaoSolar', 'TemperaturaDoAr', 'Local'],
    ['DirecaoDoVento', 'VelocidadeDoVento', 'TemperaturaInterna', 'PontoDeOrvalho', 'UmidadeRelativa', 'PressaoAtmosferica', 'RadiacaoSolar', 'TemperaturaDoAr', 'PrecSum'],
    ['DirecaoDoVento', 'VelocidadeDoVento', 'TemperaturaInterna', 'PontoDeOrvalho', 'UmidadeRelativa', 'PressaoAtmosferica', 'RadiacaoSolar', 'TemperaturaDoAr', 'Precipitacao']
]

df_training_result = pd.DataFrame(columns = ['Removed_Cols', 'Remaining_Cols', 'Local', 'Train_Acc', 'Test_Acc', 'Precision', 'Recall', 'F1', 'Ver_Pos'])

for case in test_cases:
    for l in range(6):
        if l != 0:
            df_train = df_slice[df_slice['Local'] == l]
        else:
            df_train = df_slice.copy()

        print(f'----- LOCAL {l} -----')

        # Testar prever cluster

        xgb = xgboost.XGBClassifier()

        cols_rem = ['LocalMax', 'Cluster', 'Data', 'Hora', 'Data_Hora', 'Ordens', 'Minuto'] + [c for c in df_train.columns if 'Hora_' in c]
        cols_rem = cols_rem + case
        #cols_rem = cols_rem + ['TemperaturaInterna', 'PontoDeOrvalho']

        x = df_train[[c for c in df_train.columns if c not in cols_rem]]
        if l == 0:
            print(x.columns)
        #x = x.drop(columns = 'Cluster')
        y = df_train['LocalMax']

        x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3, random_state = 378, stratify=y)

        # concatenate our training data back together
        X = pd.concat([x_treino, y_treino], axis=1)

        # separate minority and majority classes
        not_ordem = X[X['LocalMax']==0].copy()
        ordem = X[X['LocalMax']==1].copy()

        # upsample minority
        ordem_upsampled = resample(ordem,
                                replace=True, # sample with replacement
                                n_samples=len(not_ordem), # match number in majority class
                                random_state=378) # reproducible results

        # combine majority and upsampled minority
        upsampled = pd.concat([not_ordem, ordem_upsampled])

        x_treino = upsampled[[c for c in df_slice.columns if c not in cols_rem]]
        y_treino = upsampled['LocalMax']

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
        
        df_training_result = df_training_result.append({
            'Removed_Cols': case,
            'Remaining_Cols': list(x.columns),
            'Local': l,
            'Train_Acc': accuracy_score(y_treino, y_treino_pred),
            'Test_Acc': accuracy_score(y_teste, y_teste_pred),
            'Precision': precision_score(y_teste, y_teste_pred),
            'Recall': recall_score(y_teste, y_teste_pred),
            'F1': f1_score(y_teste, y_teste_pred),
            'Ver_Pos': confusion_matrix(y_teste, y_teste_pred, normalize='true')[1,1]
        }, ignore_index=True)


# In[ ]:


# pd.DataFrame(
#     columns = ['Case', 'Local', 'Train_Acc', 'Test_Acc', 'Precision', 'Recall', 'F1', 'Ver_Pos']
# ).append({
#     'Case': case,
#     'Local': l,
#     'Train_Acc': accuracy_score(y_treino, y_treino_pred),
#     'Test_Acc': accuracy_score(y_teste, y_teste_pred),
#     'Precision': precision_score(y_teste, y_teste_pred),
#     'Recall': recall_score(y_teste, y_teste_pred),
#     'F1': f1_score(y_teste, y_teste_pred),
#     'Ver_Pos': confusion_matrix(y_teste, y_teste_pred, normalize='true')[1,1]
# }, ignore_index=True)
#{'Camilopolis': 1, 'Erasmo': 2, 'Paraiso': 3, 'RM': 4, 'Vitoria': 5}
display(df_training_result.tail(30))
#df_training_result.to_csv('training_test.csv', sep=';')


# In[ ]:


df_slice.drop(columns=['Data', 'Hora'])#.to_csv('dados_treinamento_15min.csv')
# [['Local', 'UmidadeRelativa', 'PressaoAtmosferica', 'TemperaturaDoAr', 'TemperaturaInterna',
#           'PontoDeOrvalho', 'RadiacaoSolar', 'DirecaoDoVento', 'VelocidadeDoVento', 'Precipitacao'
#           'PrecSum', 'LocalMax']]

