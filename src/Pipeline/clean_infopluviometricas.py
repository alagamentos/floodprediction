import logging
from pathlib import Path
from os import mkdir
from os.path import join as pjoin
import os
import xlrd
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

SAVE = True  # Merged files
SAVE_SINGLES = False  # Save each individual xls file into csv
SAVE_CONCATENATED = True  # Save each station as a csv file
INCLUDE_MEAN = False  # Include mean of all 4 stations on merged dataframe


logging.basicConfig(level=logging.INFO, format='## Clean InfoPluviométrica - %(levelname)s: %(message)s')


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

  df = df.rename(columns = {'Data / Hora':'Data_Hora'})

  df = df[~((df['Data_Hora'] == 'Data_Hora') &
            (df['Pressão Atmosférica'] == 'Pressão Atmosférica'))]  # Remove all headers

  df = df[df.iloc[:, 0].str.contains(':', na=False) &
          df.iloc[:, 0].str.contains('/', na=False)]  # Get data rows only

  df.insert(0, 'Data', '')
  df.insert(1, 'Hora', '')
  df[['Data', 'Hora']] = df['Data_Hora'].str.split(expand=True)
  #df.drop('Data_Hora', axis = 1, inplace = True) # Split into 2 columns

  drop_cols = [4, 6, 7, 10, 12, 14, 15, 17, 20, 22]
  #  drop_cols = [3, 5, 6, 9, 11, 12, 14, 16, 19, 2]
  df = df.drop(df.columns[drop_cols], axis=1)

  col_names = ['Data', 'Hora', 'Data_Hora',
              'UmidadeRelativa', 'PressaoAtmosferica',
              'TemperaturaDoAr', 'TemperaturaInterna',
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
  #logging.info('saving to ', save_path, '\n')
  df.to_csv(save_path, sep=';', index=False)


def concatenate(df_list, name, concat_path, save=True):
  '''
  concatenate - For each station turn mutiples files into one
  '''

  if df_list:
    df = pd.concat(df_list, axis=0)

    if save:
      df_save = df.copy()
      df_save['Data_Hora'] = pd.to_datetime(df_save['Data_Hora'], dayfirst = True)
      save_concat(df_save.sort_values(by = 'Data_Hora', ascending = True), name, concat_path)

    return df


def save_concat(df, name, path):
  file = pjoin(path, name) + '.csv'
  print('$$$$ saving:', file)
  df.to_csv(file, sep=';', index=False)


def include_mean(df):
  col_names = ['UmidadeRelativa_', 'PressaoAtmosferica_', 'SensacaoTermica_',
               'RadiacaoSolar_', 'DirecaoDoVento_', 'VelocidadeDoVento_', 'Precipitacao_',
               'PontoDeOrvalho_', 'TemperaturaDoAr_', 'TemperaturaInterna_']

  for col_name in col_names:
    selected_cols = [col for col in df.columns if col_name in col]
    new_name = col_name + 'mean'
    df[new_name] = df[selected_cols].mean(axis=1, skipna=True)

  return df

def round_date(date_string):
    left = date_string[:-5]
    minute = date_string[-5:-3]
    minute = str(round(int(minute)/15) * 15)
    minute = '00' if minute == '0' else minute
    if minute == '60':
        minute = '00'
        date_concat = left + minute + ':' + '00'
        date_concat = datetime.strptime(date_concat, '%d/%m/%y %H:%M:%S')
        date_concat = date_concat + timedelta(hours = 1)
    else:
        date_concat = left + minute + ':' + '00'
        date_concat = datetime.strptime(date_concat, '%d/%m/%y %H:%M:%S')
    date_concat = date_concat.strftime('%Y/%m/%d %H:%M:%S')

    return date_concat

def days_hours_minutes(td):
  return int(td.days), td.seconds//3600, (td.seconds//60)%60

if __name__ == '__main__':
  root = os.getcwd()
  path = pjoin(root, 'data/rawdata/Info pluviometricas')
  files = []
  directories = []

  for r, d, f in os.walk(path):
    directories.extend(d)

    for file in f:
      if '.xls' in file:
        files.append(pjoin(r, file))

  directories.sort()

  # Create dir
  _path = pjoin(root, "data/cleandata")
  create_dir(_path)

  _path = pjoin(root, "data/cleandata/Info pluviometricas")
  create_dir(_path)

  if SAVE_SINGLES:
    for directory in directories:
      _path = pjoin(root, "data/cleandata/Info pluviometricas", directory)
      create_dir(_path)

  if SAVE_CONCATENATED:
    _path = pjoin(
        root, "data/cleandata/Info pluviometricas/Concatenated Data/")
    create_dir(_path)

    for directory in directories:
      _path = pjoin(
          root, "data/cleandata/Info pluviometricas/Concatenated Data", directory)
      create_dir(_path)

  # Load csvs
  dic = {directory: [] for directory in directories}

  #Load cleaned data into dictonary
  logging.info(f' loading  {len(files)}/90  files')

  i = 0

  for file in files:
    for d in directories:
      if d in file:
        logging.info(f' {i + 1}/{len(files)}')
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
        root, "data/cleandata/Info pluviometricas/Concatenated Data", d)
    concatenated[d] = concatenate(dic[d], d, _path, SAVE_CONCATENATED)

  # Merge
  keys = list(concatenated.keys())
  estacao0 = concatenated[keys[0]].copy(deep=True).drop(
      columns=['Data', 'Hora']).drop_duplicates(subset=['Data_Hora'], keep='last')
  estacao1 = concatenated[keys[1]].copy(deep=True).drop(
      columns=['Data', 'Hora']).drop_duplicates(subset=['Data_Hora'], keep='last')
  estacao2 = concatenated[keys[2]].copy(deep=True).drop(
      columns=['Data', 'Hora']).drop_duplicates(subset=['Data_Hora'], keep='last')
  estacao3 = concatenated[keys[3]].copy(deep=True).drop(
      columns=['Data', 'Hora']).drop_duplicates(subset=['Data_Hora'], keep='last')
  estacao4 = concatenated[keys[4]].copy(deep=True).drop(
      columns=['Data', 'Hora']).drop_duplicates(subset=['Data_Hora'], keep='last')

  estacao0['Data_Hora'] = estacao0['Data_Hora'].apply(round_date)
  estacao1['Data_Hora'] = estacao1['Data_Hora'].apply(round_date)
  estacao2['Data_Hora'] = estacao2['Data_Hora'].apply(round_date)
  estacao3['Data_Hora'] = estacao3['Data_Hora'].apply(round_date)
  estacao4['Data_Hora'] = estacao4['Data_Hora'].apply(round_date)

  estacao0 = estacao0.sort_values(by = ['Data_Hora', 'UmidadeRelativa'], ascending = True).drop_duplicates(subset=['Data_Hora'], keep='last')
  estacao1 = estacao1.sort_values(by = ['Data_Hora', 'UmidadeRelativa'], ascending = True).drop_duplicates(subset=['Data_Hora'], keep='last')
  estacao2 = estacao2.sort_values(by = ['Data_Hora', 'UmidadeRelativa'], ascending = True).drop_duplicates(subset=['Data_Hora'], keep='last')
  estacao3 = estacao3.sort_values(by = ['Data_Hora', 'UmidadeRelativa'], ascending = True).drop_duplicates(subset=['Data_Hora'], keep='last')
  estacao4 = estacao4.sort_values(by = ['Data_Hora', 'UmidadeRelativa'], ascending = True).drop_duplicates(subset=['Data_Hora'], keep='last')

  merge1 = estacao0.merge(estacao1, on='Data_Hora',
                          how='outer', suffixes=('_0', '_1'))
  merge2 = estacao2.merge(estacao3, on='Data_Hora',
                          how='outer', suffixes=('_2', '_3'))
  merge3 = merge1.merge(merge2, on='Data_Hora', how='outer')

  # Manualy ad suffixes to estacao4
  new_cols = []

  for col in estacao4.columns:
    if col != 'Data_Hora':
      col = col + '_4'

    new_cols.append(col)

  estacao4.columns = new_cols

  merged = merge3.merge(estacao4, on='Data_Hora', how='outer')

  logging.info(' sorting data')
  merged['Data_Hora'] = pd.to_datetime(merged['Data_Hora'], format='%Y/%m/%d %H:%M:%S')
  merged = merged.sort_values('Data_Hora').reset_index()

  # Create date_vec
  start, stop = merged['Data_Hora'].iloc[0], merged['Data_Hora'].iloc[-1]
  d,h,m = days_hours_minutes(stop - start)
  total_days = d + h/24 + m/24/60 + (1 / 24 / 4)

  date_vec= [start + timedelta(x) for x in
             np.arange(0, total_days, 1 / 24 / 4)]

  logging.info(f' {len(list(set(date_vec) - set(merged["Data_Hora"])))} missing samples')

  # Merge with date_vec
  new_df = pd.DataFrame(date_vec, columns=['Data_Hora'])
  new_df['Data_Hora'] = pd.to_datetime(new_df['Data_Hora'], yearfirst=True)
  merged = new_df.merge(merged, how='left', on='Data_Hora')

  local_cols = [col for col in merged.columns if 'Local' in col]
  for col in local_cols:
    merged.loc[:, col] = merged[col].dropna().unique()[0]

  if INCLUDE_MEAN:
    merged = include_mean(merged)

  if SAVE:
    logging.info(' saving files')
    save_path = pjoin(root, 'data/cleandata/Info pluviometricas/Merged Data/')
    create_dir(save_path)

    merged.to_csv(pjoin(save_path, 'merged.csv'),
                  decimal='.', sep=';', index=False)

  logging.info(' done!')
