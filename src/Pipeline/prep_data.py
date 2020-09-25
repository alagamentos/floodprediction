import pandas as pd
import numpy as np

from datetime import datetime
from datetime import timedelta

import logging
import os

logging.basicConfig(level=logging.INFO,
                    format='## Prep Data - %(asctime)s - %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

def reverse_ohe(df, features, ignoredFeatures, featuresLength, prefix, suffix = ''):
    all_features = ignoredFeatures + features
    df_pivot = pd.DataFrame(columns = all_features)

    for i in range(featuresLength):
        cols_aux = [f'{feature}{prefix}{i}{suffix}' for feature in features]
        df_aux = df[ignoredFeatures + cols_aux].copy()
        df_aux.columns = all_features
        df_pivot = pd.concat([df_pivot, df_aux])

    return df_pivot.sort_values(by='Data_Hora').copy()

def repair_data(df, cols):
    df_aux = df.copy()

    for feature in cols:
        df_aux.loc[df_aux[f'{feature}_repaired'], f'{feature}'] = df_aux.loc[df_aux[f'{feature}_repaired'], f'{feature}_pred']
    df_aux = df_aux.drop(columns = [c for c in df_aux.columns if '_pred' in c or '_repaired' in c])

    return df_aux.copy()

if __name__ == "__main__":
  dir = os.path.dirname(os.path.realpath(__file__))

  # Carregar e corrigir medições
  logging.info(f'Carregando medições')
  df_merged = pd.read_csv(os.path.join(dir, '../../data/cleandata/Info pluviometricas/Merged Data/merged.csv'), sep = ';')
  df_repaired = pd.read_csv(os.path.join(dir, '../../data/cleandata/Info pluviometricas/Merged Data/repaired.csv'), sep = ';')
  df_merged['Data_Hora'] = pd.to_datetime(df_merged['Data_Hora'], yearfirst=True)
  df_repaired['Data_Hora'] = pd.to_datetime(df_repaired['Data_Hora'], yearfirst=True)

  df = df_repaired[df_merged.columns.drop('index')].copy()
  df = df.drop(columns = [c for c in df.columns if 'Sensacao' in c or 'Interna' in c])
  df['Data'] = df['Data_Hora'].dt.strftime('%Y-%m-%d')

  logging.info(f'Carregando labels')
  # Carregar labels e juntar ao dataframe
  df_label_h = pd.read_csv(os.path.join(dir, '../../data/cleandata/Ordens de serviço/labels_hour.csv'), sep = ';')
  df_label_h.columns = ['Data_Hora', 'LocalMax_h_All', 'LocalMax_h_0', 'LocalMax_h_1', 'LocalMax_h_2', 'LocalMax_h_3',
                      'LocalMax_h_4', 'LocalMax_h_ow', 'Local_h_0', 'Local_h_1', 'Local_h_2', 'Local_h_3',
                      'Local_h_4', 'Local_h_Null']
  df_label_h['Data_Hora'] = pd.to_datetime(df_label_h['Data_Hora'], yearfirst=True)
  df_label_h['Data'] = df_label_h['Data_Hora'].dt.strftime('%Y-%m-%d')
  df_label_h['Hora'] = df_label_h['Data_Hora'].dt.hour
  df_label_h = df_label_h.drop(columns='Data_Hora')
  df_label_d = pd.read_csv(os.path.join(dir, '../../data/cleandata/Ordens de serviço/labels_day.csv'), sep = ';')
  df_label_d.columns = ['Data', 'LocalMax_d_All', 'LocalMax_d_0', 'LocalMax_d_1', 'LocalMax_d_2', 'LocalMax_d_3',
                      'LocalMax_d_4', 'LocalMax_d_ow', 'Local_d_0', 'Local_d_1', 'Local_d_2', 'Local_d_3',
                      'Local_d_4', 'Local_d_Null']

  df_labels = df.merge(df_label_d, on='Data', how='left')
  df_labels['Hora'] = df_labels['Data_Hora'].dt.hour
  df_labels = df_labels.merge(df_label_h, on=['Data', 'Hora'], how='left')
  df_labels = df_labels.fillna(0)
  df_labels = df_labels.drop(columns = ['Data', 'Hora'])


  # "Reverse OHE"
  features = [
      'Local',
      'UmidadeRelativa',
      'PressaoAtmosferica',
      'TemperaturaDoAr',
      'PontoDeOrvalho',
      'RadiacaoSolar',
      'DirecaoDoVento',
      'VelocidadeDoVento',
      'Precipitacao',
      'LocalMax_d',
      'LocalMax_h',
      'Local_d',
      'Local_h',
  ]

  ignoredFeatures = [
      'Data_Hora',
      'LocalMax_d_All',
      'LocalMax_d_ow',
      'Local_d_Null',
      'LocalMax_h_All',
      'LocalMax_h_ow',
      'Local_h_Null'
  ]

  logging.info(f'Aplicando reverse OHE')
  # Dataframe "completo" pronto para salvar
  df_labels_grouped = reverse_ohe(df_labels, features, ignoredFeatures, 5, '_')
  # Selecionar colunas de interesse

  logging.info(f'Preparando dataset resumido')
  df_labels_simple = df_labels_grouped[['Data_Hora', 'Local', 'Precipitacao', 'LocalMax_d_All']].copy()
  df_labels_simple.columns = ['Data_Hora', 'Local', 'Precipitacao', 'Label']
  df_labels_simple['Mes'] = df_labels_simple['Data_Hora'].dt.month
  df_labels_simple['Dia'] = df_labels_simple['Data_Hora'].dt.day


  # Substituir Local
  df_labels_simple = df_labels_simple.replace({'Camilopolis': 1, 'Erasmo': 2, 'Paraiso': 3, 'RM': 4, 'Vitoria': 5})


  # Agrupar por hora
  # TODO: tratamento inicial ~> selecionar apenas minuto 0
  df_labels_simple = df_labels_simple[df_labels_simple['Data_Hora'].dt.minute == 0].copy()


  # Adicionar soma de precipitação do dia
  df_labels_simple['Data'] = df_labels_simple['Data_Hora'].dt.strftime('%Y-%m-%d')
  df_prec_sum = df_labels_simple.groupby(['Data', 'Local']).sum().reset_index()[['Data', 'Local', 'Precipitacao']]
  df_prec_sum.columns = ['Data', 'Local', 'PrecSum']
  df_labels_simple = df_labels_simple.merge(df_prec_sum, on=['Data', 'Local'])
  df_labels_simple = df_labels_simple.drop(columns = 'Data')


  # Filtrar soma de precipitação do dia <= 10
  df_labels_simple.loc[(df_labels_simple['Label'] == 1) & (df_labels_simple['PrecSum'] <= 10), 'Label'] = 0


  # Reordenar colunas
  df_labels_simple = df_labels_simple[['Data_Hora', 'Mes', 'Dia', 'Local', 'Precipitacao', 'PrecSum', 'Label']].copy()

  # Salvar datasets
  logging.info(f'Salvando datasets')
  df_labels_grouped.to_csv(os.path.join(dir, '../../data/cleandata/Info pluviometricas/Merged Data/full_data.csv'), index=False, sep=';')
  df_labels_simple.to_csv(os.path.join(dir, '../../data/cleandata/Info pluviometricas/Merged Data/prepped_data.csv'), index=False, sep=';')

  logging.info(f'Pronto')
