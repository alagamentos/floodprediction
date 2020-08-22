import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

# import matplotlib.pyplot as plt
# from plotly import graph_objects as go
# import plotly as py

from datetime import datetime
from datetime import timedelta

import logging
from pathlib import Path
from os import mkdir
from os.path import join as pjoin
import os
from utils import *


logging.basicConfig(level=logging.INFO,
                    format='## Clusterize - %(asctime)s - %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


def Calculate_Dist(lat1, lon1, lat2, lon2):
  r = 6371
  phi1 = np.radians(lat1)
  phi2 = np.radians(lat2)
  delta_phi = np.radians(lat2 - lat1)
  delta_lambda = np.radians(lon2 - lon1)
  a = np.sin(delta_phi / 2)**2 + np.cos(phi1) *\
      np.cos(phi2) *   np.sin(delta_lambda / 2)**2
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
    date_concat = datetime.strptime(date_concat, '%Y-%m-%d %H:%M:%S')
    date_concat = date_concat + timedelta(hours = 1)
    date_concat = date_concat.strftime('%Y-%m-%d %H:%M:%S')
  else:
    date_concat = left + minute + ':' + '00'

  return date_concat



if __name__ == "__main__":
  logging.info('## Initializing')
  path = pjoin(os.getcwd(), 'data/cleandata/')
  # Ler ordens de serviço
  df = pd.read_csv(pjoin(path, 'Ordens de serviço/Enchentes_LatLong.csv'), sep = ';')
  est = pd.read_csv(pjoin(path, 'Estacoes/lat_lng_estacoes.csv'), sep = ';')

  # Operar em ordens de serviço
  ord_serv = get_distances(est, df)
  ord_serv.loc[ord_serv['Distance'] > 4.5, 'Est. Prox'] = 'Null'
  ord_serv = ord_serv[['lat','lng','Data', 'Hora', 'Est. Prox']]
  ord_serv = ord_serv.sort_values(['Data', 'Hora'])

  ord_serv['pos'] = ord_serv['lat'].astype(str).str.rstrip() + \
                    ord_serv['lng'].astype(str).str.rstrip()

  le = preprocessing.LabelEncoder()
  le.fit(ord_serv['pos'])
  ord_serv['pos'] = le.transform(ord_serv['pos'])
  ord_serv['Data_Hora'] = ord_serv['Data'] + ' ' + ord_serv['Hora']

  # Ler informações pluviométricas
  df_merged = pd.read_csv(pjoin(path, 'Info pluviometricas/Merged Data/merged.csv'), sep = ';')
  df_repaired = pd.read_csv(pjoin(path, 'Info pluviometricas/Merged Data/repaired.csv'), sep = ';')
  df_merged['Data_Hora'] = pd.to_datetime(df_merged['Data_Hora'], yearfirst=True)
  df_repaired['Data_Hora'] = pd.to_datetime(df_repaired['Data_Hora'], yearfirst=True)

  # Preparar ordens
  df_ord = ord_serv[['Est. Prox', 'Data_Hora']].copy()
  df_ord['Data_Hora'] = pd.to_datetime(df_ord['Data_Hora'], yearfirst=True).dt.strftime('%Y-%m-%d %H:%M:%S').apply(round_date)
  df_ord['Data_Hora'] = pd.to_datetime(df_ord['Data_Hora'], yearfirst=True)
  df_ord['Data'] = df_ord['Data_Hora'].dt.strftime('%Y-%m-%d')
  df_ord = df_ord.drop(columns = 'Data_Hora')
  df_ord = df_ord[df_ord['Est. Prox'] != 'OpenWeather']

  # Relacionar quantidade de ordens de serviço para cada dia/estação
  logging.info('## Prepare data for clustering')
  df_est = pd.DataFrame(columns=['Data'] + list(df_ord['Est. Prox'].unique()))

  for index, row in df_ord.iterrows():
    if (df_est['Data'] == row['Data']).any():
      df_est.loc[df_est['Data'] == row['Data'], row['Est. Prox']] = df_est.loc[df_est['Data'] == row['Data'], row['Est. Prox']] + 1
    else:
      df_est.loc[df_est.shape[0]] = [row['Data'], 0, 0, 0, 0, 0, 0]
      df_est.loc[df_est['Data'] == row['Data'], row['Est. Prox']] = 1

  # Juntar dados de informação pluviométrica e das ordens de serviço
  df_merged_n = df_merged.drop(columns = [c for c in df_merged.columns.values if 'Sensacao' in c]).dropna().copy()
  df_repaired_n = df_repaired.drop(columns = [c for c in df_repaired.columns.values if 'Sensacao' in c]).dropna().copy()

  df = df_merged_n.merge(df_repaired_n, on='Data_Hora')
  df['Data_Hora'] = pd.to_datetime(df['Data_Hora'], yearfirst=True)
  df['Data'] = df['Data_Hora'].dt.strftime('%Y-%m-%d')
  df = df.merge(df_est, on='Data', how = 'outer')
  df['Data_Hora'] = pd.to_datetime(df['Data_Hora'], yearfirst=True)
  df = df.sort_values(by = 'Data_Hora')

  # Temp: remover casos onde não há dados de medições
  df = df[~df['index'].isna()]

  # Substituir valores que contém erro
  df_ungrouped = df.copy()
  df_ungrouped = df_ungrouped.drop(columns = ['index', 'Data'] + [c for c in df.columns if 'interpol' in c])
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

  # Agrupar colunas por estação
  logging.info('## Pivot and group data')
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

  # Agrupar dados por local e data
  df_prec = df_grouped.copy()
  df_prec['Ano'] = df_prec['Data_Hora'].dt.year
  df_prec['Mes'] = df_prec['Data_Hora'].dt.month
  df_prec['Dia'] = df_prec['Data_Hora'].dt.day
  df_prec = df_prec.drop(columns = ['Data_Hora'])
  s_prec_p = df_prec.groupby(['Local', 'Ano', 'Mes', 'Dia']).sum().reset_index()['Precipitacao']
  s_prec_o = df_prec.groupby(['Local', 'Ano', 'Mes', 'Dia']).max().reset_index()['Ordens']
  df_prec = df_prec.groupby(['Local', 'Ano', 'Mes', 'Dia']).mean().reset_index()
  df_prec['Precipitacao'] = s_prec_p
  df_prec['Ordens'] = s_prec_o
  df_prec['Ordens'] = df_prec['Ordens'].astype(int)

  # Modelo K-Means
  logging.info('## Train K-Means')
  sc = MinMaxScaler(feature_range=(0,1))
  df_norm = sc.fit_transform(df_prec[['Precipitacao', 'Ordens']])
  cluster = KMeans(n_clusters=4, random_state=42).fit(df_norm)
  df_prec['Cluster'] = cluster.labels_

  # Preparatório para AutoML
  logging.info('## Organize data')
  df_prec['Data'] = pd.to_datetime(df_prec['Ano'].map(str) + "-" + df_prec['Mes'].map(str) + "-" + df_prec['Dia'].map(str))
  df_grouped['Data'] = pd.to_datetime(df_grouped['Data_Hora'].dt.strftime('%Y-%m-%d'))

  df_cluster = df_grouped.merge(df_prec[['Data', 'Local', 'Cluster']], on=['Data', 'Local'])
  df_cluster = df_cluster.drop(columns = 'Data')

  logging.info('## Saving data to cluster.csv')
  df_cluster.to_csv(pjoin(path, 'Info pluviometricas/Merged Data/clustered.csv'), sep = ';', index=False)
  logging.info('## Done!')
