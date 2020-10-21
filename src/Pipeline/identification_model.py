#!/usr/bin/env python
# coding: utf-8

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

import logging
import os

logging.basicConfig(level=logging.INFO,
                    format='## Prep Data - %(asctime)s - %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

def upsampleData(X, label):
    # Separar verdadeiro e falso
    false_label = X[X[label]==0].copy()
    true_label = X[X[label]==1].copy()

    # Realizar upsample para os valores verdadeiros
    label_upsampled = resample(true_label,
                            replace=True, # sample with replacement
                            n_samples=len(false_label), # match number in majority class
                            random_state=378) # reproducible results
    upsampled = pd.concat([false_label, label_upsampled])

    # Separar x e y
    x = upsampled[[c for c in X.columns if label not in c]]
    y = upsampled[label]

    return x, y

def trainXGB(df, cols_rem, label, verbose=True):
    # Separar x e y e remover colunas desnecessárias
    x = df[[c for c in df.columns if c not in cols_rem]]
    y = df[label]

    # Separar dados de treinamento e teste
    x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3, random_state = 378, stratify=y)

    # Upsample
    X = pd.concat([x_treino, y_treino], axis=1)
    x_treino, y_treino = upsampleData(X, label)

    # Parâmetros do XGBClassifier
    param = {'max_depth':50, 'eta':1, 'objective':'binary:logistic', 'min_child_weight': 1, 'lambda': 1, 'alpha': 0, 'gamma': 0}

    # Gerar DMatrix com dados de treinamento e teste
    df_train = xgboost.DMatrix(data=x_treino, label=y_treino)

    # Treinar modelo e predizer em cima dos dados de treinamento e teste
    bst = xgboost.train(param, df_train, 2, feval=f1_score)
    y_teste_pred = bst.predict(xgboost.DMatrix(data=x_teste, label=y_teste))
    y_teste_pred = [1 if i>0.5 else 0 for i in y_teste_pred]
    y_treino_pred = bst.predict(xgboost.DMatrix(data=x_treino, label=y_treino))
    y_treino_pred = [1 if i>0.5 else 0 for i in y_treino_pred]

    # Mostrar resultados se verbose é verdadeiro
    if verbose:
        print(f"Treino: {accuracy_score(y_treino, y_treino_pred)}")
        print(f"Teste: {accuracy_score(y_teste, y_teste_pred)}")
        print(f"Precisão: {precision_score(y_teste, y_teste_pred)}")
        print(f"Recall: {recall_score(y_teste, y_teste_pred)}")
        print(f"F1: {f1_score(y_teste, y_teste_pred)}")

    # Salvar resultados em um dict
    results = {
        'Features': list(x.columns),
        'Train_Acc': accuracy_score(y_treino, y_treino_pred),
        'Test_Acc': accuracy_score(y_teste, y_teste_pred),
        'Precision': precision_score(y_teste, y_teste_pred),
        'Recall': recall_score(y_teste, y_teste_pred),
        'F1': f1_score(y_teste, y_teste_pred),
        'Ver_Pos': confusion_matrix(y_teste, y_teste_pred, normalize='true')[1,1]
    }

    return bst, results, y_treino_pred, y_teste_pred

if __name__ == "__main__":
  dir = os.path.dirname(os.path.realpath(__file__))

  logging.info(f'Carregando dados preparados')

  df = pd.read_csv(os.path.join(dir, '../../data/cleandata/Info pluviometricas/Merged Data/prepped_data.csv'), sep=';')
  df['Data_Hora'] = pd.to_datetime(df['Data_Hora'], yearfirst=True)
  df = df.sort_values(['Data_Hora', 'Local'])

  logging.info(f'Treinando modelo de identificação 0H')

  label = 'Label'
  cols_rem = ['LocalMax', 'Label', 'Label_Old', 'Cluster', 'Data', 'Hora', 'Data_Hora', 'Ordens', 'Minuto'] + [c for c in df.columns if 'Hora_' in c]
  model, training_res, y_train_pred, y_test_pred = trainXGB(df, cols_rem, label, verbose=False)

  logging.info(f'Obtendo label nova através de predições')
  df_m = df[df['Label'] == 1].copy()
  df_m['Data'] = df_m['Data_Hora'].dt.strftime("%Y-%m-%d")

  def getPrecMomento(row):
      prec_momento = df_m.loc[(df_m['Data_Hora'] <= row['Data_Hora']) & (df_m['Local'] == row['Local']) & (df_m['Data'] == row['Data']), 'Precipitacao'].sum()
      return prec_momento

  df_m['PrecMomento'] = df_m.apply(getPrecMomento, axis=1)
  df_m = df_m.rename(columns = {'PrecSum': 'PrecSumOld', 'PrecMomento': 'PrecSum'})
  label_pred = model.predict(xgboost.DMatrix(data=df_m[training_res['Features']]))
  df_m['Label_Pred'] = [1 if i>0.5 else 0 for i in label_pred]

  logging.info(f'Preparando dataset para label nova')
  df_g = df_m.groupby(['Data', 'Local']).max()
  df_g = df_m.groupby(['Data', 'Local', 'Label_Pred']).min().reset_index()

  df_g = df_g.loc[df_g['Label_Pred'] == 1, ['Data', 'Local', 'Data_Hora']].rename(columns={'Data_Hora':'Min_Hora'})
  df_g['Min_Hora'] = df_g['Min_Hora'].dt.hour

  df_new = df.copy()
  df_new['Data'] = df_new['Data_Hora'].dt.strftime('%Y-%m-%d')
  df_new = df_new.merge(df_g, on=['Local', 'Data'], how='left').fillna(24)

  df_new['Label_New'] = 0
  df_new.loc[(df_new['Label'] == 1) & (df_new['Data_Hora'].dt.hour >= df_new['Min_Hora']), 'Label_New'] = 1
  df_new = df_new.rename(columns = {'Label': 'Label_Old', 'Label_New': 'Label'})

  logging.info(f'Salvando label nova')
  df_new[['Data_Hora', 'Local', 'Label']].to_csv(os.path.join(dir, '../../data/cleandata/Ordens de serviço/labels_predict.csv'), sep=';')
  logging.info(f'Dataset salvo como labels_predict.csv')

  logging.info(f'Salvando modelo')
  model.save_model(os.path.join(dir, '../../data/model/Identificacao_0H.json'))
  logging.info(f'Modelo salvo como Identificacao_0H.json')
