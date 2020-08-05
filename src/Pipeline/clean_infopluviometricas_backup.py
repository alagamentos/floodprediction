import logging
from pathlib import Path
from os import mkdir
from os.path import join as pjoin
import os
import xlrd
import sys
import numpy as np
import pandas as pd
SAVE = True  # Merged files
SAVE_SINGLES = True  # Save each individual station
INCLUDE_MEAN = False  # Include mean of all 4 stations on merged dataframe


logging.basicConfig(level=logging.INFO,
                    format='## Clean - %(levelname)s: %(message)s')
data_path = os.path.dirname(__file__)


def create_dir(_path):
  if not os.path.exists(_path):
    mkdir(_path)
    logging.info(f' creating directory: {_path}')


def clean(df, file, save=True):
  '''
  clean - Clean each file and save as a single csv - This results in multiple files; one for each station
  '''

  df.columns = (list(df.iloc[2].values))  # Get column names
  df = df.loc[:, df.columns.notnull()]   # Remove nan columns
  df = df[~((df['Data / Hora'] == 'Data / Hora') &
            (df['Pressão Atmosférica'] == 'Pressão Atmosférica'))]  # Remove all headers

  df = df[df.iloc[:, 0].str.contains(':', na=False) &
          df.iloc[:, 0].str.contains('/', na=False)]  # Get data rows only

  df.insert(0, 'Data', '')
  df.insert(1, 'Hora', '')
  df[['Data', 'Hora']] = df['Data / Hora'].str.split(expand=True)
  #df.drop('Data / Hora', axis = 1, inplace = True) # Split into 2 columns

  drop_cols = [4, 6, 7, 10, 12, 14, 15, 17, 20, 22]
  #  drop_cols = [3, 5, 6, 9, 11, 12, 14, 16, 19, 2]
  df = df.drop(df.columns[drop_cols], axis=1)

  col_names = ['Data', 'Hora', 'Data / Hora',
              'UmidadeRelativa', 'PressaoAtmosferica',
              'Temperatura do Ar', 'TemperaturaInterna',
              'PontoDeOrvalho', 'SensacaoTermica',
              'RadiacaoSolar', 'DirecaoDoVento',
              'VelocidadeDoVento', 'Precipitacao']
  df.columns = col_names
  df['Local'] = os.path.basename(file).split()[0].split('_')[0]

  if save:
    save_to_file(df, file)

  return df


def save_to_file(df, file):
  save_path = file.replace('rawdata', 'cleandata')
  save_path = save_path.replace('.xls', '.csv')
  print('*************')
  print(save_path)
  #logging.info('saving to ', save_path, '\n')
  df.to_csv(save_path, sep=';', index=False)


def concatenate(df_list, name, concat_path, save=True):
  '''
  concatenate - For each station turn mutiples files into one
  '''

  if df_list:
    df = pd.concat(df_list, axis=0)

    if save:
      save_concat(df, name, concat_path)

    return df


def save_concat(df, name, path):
  file = pjoin(path, name) + '.csv'
  df.to_csv(file, sep=';', index=False)


def include_mean(df):
  col_names = ['UmidadeRelativa_', 'PressaoAtmosferica_', 'SensacaoTermica_',
              'RadiacaoSolar_', 'DirecaoDoVento_', 'VelocidadeDoVento_', 'Precipitacao_',
              'PontoDeOrvalho_', 'Temperatura do Ar_', 'TemperaturaInterna_']

  for col_name in col_names:
    selected_cols = [col for col in df.columns if col_name in col]
    new_name = col_name + 'mean'
    df[new_name] = df[selected_cols].mean(axis=1, skipna=True)

  return df


if __name__ == '__main__':
  root = Path(__file__).resolve().parents[2]  # resolve() -> absolute path
  path = pjoin(root, 'data/rawdata/Info pluviometricas')
  files = []
  directories = []

  for r, d, f in os.walk(path):
    directories.extend(d)

    for file in f:
      if '.xls' in file:
        files.append(pjoin(r, file))

  # Create dir
  _path = pjoin(root, "data/cleandata")
  create_dir(_path)

  _path = pjoin(root, "data/cleandata/Info pluviometricas")
  create_dir(_path)

  _path = pjoin(root, "data/cleandata/Info pluviometricas/Concatenated Data/")
  create_dir(_path)

  for directory in directories:
    _path = pjoin(root, "data/cleandata/Info pluviometricas/", directory)
    create_dir(_path)

  # Load csvs
  dic = {directory: [] for directory in directories}

  #Load cleaned data into dictonary
  logging.info(f' loading  {len(files)}/90  files')

  i = 0

  for file in files:
    for d in directories:
      if d in file:
        logging.info(f' {i}/{len(files)}')
        filename = os.path.basename(file)
        wb = xlrd.open_workbook(file, logfile=open(
            os.devnull, 'w'))  # Supress xlrd warnings
        df = pd.read_excel(wb)
        dic[d].append(clean(df, file, SAVE_SINGLES))
        i += 1

  # Concatanate and save
  logging.info(' merging files')
  concatenated = {}
  for d in directories:
    _path = pjoin(
        root, "data/cleandata/Info pluviometricas/Concatenated Data", directory)
    concatenated[d] = concatenate(dic[d], d, _path, SAVE)

  # Merge
  keys = list(concatenated.keys())
  estacao0 = concatenated[keys[0]].copy(
      deep=True).drop(columns=['Data', 'Hora'])
  estacao1 = concatenated[keys[1]].copy(
      deep=True).drop(columns=['Data', 'Hora'])
  estacao2 = concatenated[keys[2]].copy(
      deep=True).drop(columns=['Data', 'Hora'])
  estacao3 = concatenated[keys[3]].copy(
      deep=True).drop(columns=['Data', 'Hora'])
  estacao4 = concatenated[keys[4]].copy(
      deep=True).drop(columns=['Data', 'Hora'])

  merge1 = estacao0.merge(estacao1, on='Data / Hora',
                        how='outer', suffixes=('_0', '_1'))
  merge2 = estacao2.merge(estacao3, on='Data / Hora',
                        how='outer', suffixes=('_2', '_3'))
  merge3 = merge1.merge(merge2, on='Data / Hora', how='outer')

  # Manualy ad suffixes to estacao4
  new_cols = []
  for col in estacao4.columns:
    if col != 'Data / Hora':
      col = col + '_4'

    new_cols.append(col)

  estacao4.columns = new_cols

  merged = merge3.merge(estacao4, on='Data / Hora', how='outer')

  merged.insert(0, 'Data', '')
  merged.insert(1, 'Hora', '')
  merged[['Data', 'Hora']] = merged['Data / Hora'].str.split(expand=True)

  if INCLUDE_MEAN:
    merged = include_mean(merged)

  if SAVE:
    logging.info(' saving files')
    save_path = pjoin(root, 'data/cleandata/Info pluviometricas/Merged Data/')
    create_dir(save_path)

    merged.to_csv(pjoin(save_path, 'merged.csv'),
                decimal='.', sep=';', index=False)

  logging.info(' done!')
