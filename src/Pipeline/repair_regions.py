import logging
import os
from os.path import join as pjoin
import pandas as pd
import numpy as np
import re
from utils import *
import plotly.graph_objects as go
import xgboost
import concurrent
import time
import yaml

pd.options.plotting.backend = "plotly"

#fig = df.plot()
#fig.write_image()

logging.basicConfig(level=logging.INFO,
                    format='## Repair - %(asctime)s - %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


def read_yaml(path):
  with open(path) as file:
    return yaml.load(file, Loader=yaml.FullLoader)


def interpolation(Series, error, max_interpolation_size=5, **kwargs):
  error_reg = list_2_regions(error)
  reg_size = [i[1] - i[0] for i in error_reg]

  interpol_reg = [error_reg[i]
                  for i in range(len(reg_size)) if reg_size[i] <= max_interpolation_size]
  error_interpol = regions_2_list(interpol_reg, len(Series))

  error.loc[error_interpol] = False

  Series.fillna(-100000)
  Series.loc[error_interpol] = np.nan

  Series = Series.interpolate(method='linear', limit=max_interpolation_size)
  Series.loc[Series == -100000] = np.nan

  return Series, error.tolist(), error_interpol


def predict_region(xgb, df_recurrent, label, dcols, p, lookbackSize):
    df_recurrent = df_recurrent.copy(deep=True)
    start, stop = p[0], p[1]
    y_predict = []
    for j in range(start, stop):
      pred_ = xgb.predict(df_recurrent.loc[[j]])
      y_predict.append(pred_)
      matrix = df_recurrent.loc[j+1:j+1+lookbackSize, dcols].values
      np.fill_diagonal(matrix, pred_)
      df_recurrent.loc[j+1:j+1+lookbackSize, dcols] = matrix
    return y_predict


def repair_regions(df, label, max_region_size=None, lookbackSize=None, extra_features=None, **kwargs):
  '''
  df
  label
  max_interpolation_size
  lookbackSize
  extra_features - list
  '''

  attribute = label.split('_')[0]
  cols_attribute = [i for i in df.columns if attribute in i and 'interpol' not in i]
  df_att = df[extra_features + cols_attribute].copy(deep=True)

  # Create Delay | LookBack
  for i in range(1, lookbackSize + 1):
    df_att['delay_' + str(i)] = df_att[label].shift(i)
    df_att['delay_error' + str(i)] = df_att[label+'_error'].shift(i)

  cols_delay = [i for i in df_att.columns if 'delay' in i]

  # Eliminate faulty data
  error_cols = [
      i for i in df_att.columns if 'delay_error' in i or i == (label+'_error')]
  i = 0
  for col in error_cols:
    if i == 0:
      has_error = df_att[col]
      i += 1
    else:
      has_error = has_error | df_att[col].fillna(value=True)

  df_error = df_att[~has_error].drop(columns=error_cols)

  logging.info(label)
  logging.info(f'Dados com erro:  {has_error.sum()/len(df_att)* 100}%')

  logging.info('Training Model')

  try:
    xgb = xgboost.XGBRegressor(objective='reg:squarederror', tree_method='gpu_hist')
    xgb.fit(df_error.drop(columns=[label]), df_error[label]) # Ele tenta executar pela GPU apenas ao rodar o FIT
  except:
    xgb = xgboost.XGBRegressor(objective='reg:squarederror')
    xgb.fit(df_error.drop(columns=[label]), df_error[label]) # Ele tenta executar pela GPU apenas ao rodar o FIT

  drop_cols = [i for i in df_att.columns if 'delay_error' in i] + \
              [label] + [label+'_error']

  df_recurrent = df_att.drop(columns=drop_cols).copy(deep=True)
  df[label + '_pred'] = df[label]

  predict_regions = list_2_regions(df_att[label + '_error'])

  dcols = ['delay_' + str(i+1) for i in range(lookbackSize)]

  # Limit Regions size by max_region_size
  nr = len(predict_regions)
  reg_size = [i[1] - i[0] for i in predict_regions]
  predict_regions = [predict_regions[i]
                     for i in range(len(reg_size)) if reg_size[i] < max_region_size]
  logging.info(f'Number of regions: {nr} -> {len(predict_regions)}')

  df[label + '_repaired'] = regions_2_list(predict_regions, len(df))

  with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    futures = {executor.submit(predict_region, xgb, df_recurrent,
                               label, dcols, p, lookbackSize): p for p in predict_regions}

    for future in concurrent.futures.as_completed(futures):
      pr = futures[future]
      try:
        data = future.result()
        df.loc[pr[0]:pr[1]-1, label + '_pred' ] = data
      except Exception as exc:
        print(f'Generated an exception: {exc}')
      # else:
        # print(f'{pr[0]}-{pr[1]} finished')

  # print(f"Time: {time.time() - time_start} seconds")

  logging.info('Generating Plots')

  if not os.path.exists("images"):
    os.mkdir("images")

  number_of_plots = 10
  for i, p in enumerate(predict_regions[0:number_of_plots]):
    start, stop = p[0] - 100, p[1] + 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[label][start:stop].index,
        y=df[label][start:stop],
        mode='lines', name='Real', connectgaps=False))
    fig.add_trace(go.Scatter(
        x=df[label + '_pred'][start:stop]
        .where(df[label + '_error'][start:stop]).index,
        y=df[label + '_pred'][start:stop]
        .where(df[label + '_error'][start:stop]),
        mode='lines', name='Prediction'))

    fig.write_image(f'images/{label}_{i}.jpg')

  #df.loc[:,label] = prediction[label]


def fill_mean(df, col, **kwargs):
  select_cols = [col for col in df.columns if col.split('_')[0] in col]

  repaired_cols = f'{col}_repaired'
  error_cols    = f'{col}_error'
  select_rows = df[(error_cols == 1) & (repaired_cols = 0)]


if __name__ == '__main__':
  root = os.getcwd()
  data_path = pjoin(
      root, 'data/cleandata/Info pluviometricas/Merged Data/merged.csv')

  regions_path = pjoin(
      root, 'data/cleandata/Info pluviometricas/Merged Data/error_regions.csv')

  save_path = pjoin(root, 'data/cleandata/Info pluviometricas/Merged Data/repaired.csv')
  config_path = 'src/Pipeline/config/repair_regions.yaml'

  data = pd.read_csv(data_path,
                   sep=';',
                   dtype={'Local_0': object, 'Local_1': object,
                          'Local_2': object, 'Local_3': object})

  regions = pd.read_csv(regions_path, sep =';')

  config = read_yaml(os.path.join(root, config_path))

  df = data.merge(regions, on = 'Data_Hora')

  df[['Date', 'Time']] = df['Data_Hora'].str.split(expand=True)
  df[['Hora', 'Min', 'Seg']] = df['Time'].str.split(':', expand=True)
  df[['Ano', 'Mes', 'Dia']] = df['Date'].str.split('-', expand = True)
  df['Hora'] = df['Hora'].astype(int)
  df['Min'] = df['Min'].astype(int)
  df['Ano'] = df['Ano'].astype(int)
  df['Mes'] = df['Mes'].astype(int)
  df['Dia'] = df['Dia'].astype(int)

  r = re.compile(".*error")
  cols = list(filter(r.match, df.columns.to_list()))
  cols = [i for i in cols if 'Precipitacao' not in i]

  df_cols = pd.DataFrame(columns=['Coluna', 'Valor'])

  for col in cols:
      val = round(df[col].value_counts()[True] / df.shape[0] * 100, 2)
      df_cols.loc[df_cols.shape[0]] = [col, val]

  df_cols = df_cols.sort_values(by='Valor').reset_index(drop=True)
  df_cols.iloc[:, 0] = df_cols.iloc[:, 0].str.replace("_error", "")

  interpolation_cols = [col for col in df_cols.iloc[:,0] if 'interpolation' in config[col.split('_')[0]] ]
  regression_cols    = [col for col in df_cols.iloc[:,0] if 'regression'    in config[col.split('_')[0]] ]
  fill_mean_cols     = [col for col in df_cols.iloc[:,0] if 'fill_mean'     in config[col.split('_')[0]] ]

  ## Interpolation -------------------
  for i, col in enumerate(interpolation_cols):
    logging.info(f'## Interpolating Data {col}')

    label = col.split('_')[0]

    error = df[df_cols.iloc[i, 0] + '_error'].copy(deep=True)
    Series = df.loc[:,col].copy(deep=True)

    kwargs = config[label]['interpolation']
    n_Series, n_error, error_interpol = interpolation(
                                        Series, error, **kwargs)

    df.loc[:, df_cols.iloc[i, 0] + '_error'] = n_error
    df.loc[:, df_cols.iloc[i, 0] + '_interpol'] = error_interpol
    df.loc[:, df_cols.iloc[i, 0]] = n_Series

  ## Regression -------------------
  logging.info('\n## XGB Regression')
  for i, col in enumerate(regression_cols):
    logging.info(f'({i}/{len(regression_cols)}) Training for {col}')

    label = col.split('_')[0]
    kwargs = config[label]['regression']
    repair_regions(df, col, **kwargs)

    logging.info(f'({i}/{len(df_cols)}) Done training for {label}\n')

  ## Fill Mean ----------------------
  logging.info('\n## Fill With Mean')
  for i, col in enumerate(fill_mean_cols):
    logging.info(f'({i}/{len(fill_mean_cols)}) Training for {col}')

    label = col.split('_')[0]
    kwargs = config[label]['fill_mean']
    fill_mean(df, col, **kwargs)


  save_cols = ['Data_Hora'] + [i for i in df.columns if ('_interpol' in i) or ('_pred' in i) or ('_repaired' in i) ]
  df[save_cols].to_csv(save_path, decimal='.', sep=';', index=False)
