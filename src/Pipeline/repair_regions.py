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
from datetime import timedelta

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

  interpol_reg = [error_reg[i] for i in range(len(reg_size))
                               if reg_size[i] <= max_interpolation_size]
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

  df[label + '_regression'] = regions_2_list(predict_regions, len(df))

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


def Calculate_Dist(lat1, lon1, lat2, lon2):
    r = 6371
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2)**2 + np.cos(phi1) *\
        np.cos(phi2) * np.sin(delta_lambda / 2)**2
    res = r * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))
    return np.round(res, 2)


def idw_interpolate( row , num, label, distances):

    rest = [i for i in range(5) if i != num]
    row = row.fillna(0)

    aux_num, aux_den = 0, 0
    for r in rest:

        p = row[f'{label}_{r}']
        local_a = row[f'Local_{num}']
        local_b = row[f'Local_{r}']

        d = distances[local_a][local_b]

        aux_num += p/d * (not row[f'{label}_{r}_error'])
        aux_den += 1/d * (not row[f'{label}_{r}_error'])

    if aux_den == 0:
        return np.nan

    return aux_num/aux_den


def fill_ow( row , num , label, df_ow):

    rounded_hour = row['Data_Hora'].round('H')
    mask = pd.to_datetime(df_ow['Data_Hora']) == rounded_hour
    try:
        return df_ow.loc[mask,label].item()
    except ValueError:
        mask = pd.to_datetime(df_ow['Data_Hora']) == rounded_hour + timedelta(hours=1)
        return df_ow.loc[mask,label].item()


def get_estacaoes_dist(est):
  estacoes = list(est.index)

  distances = {k: {} for k in estacoes}

  for estacao in estacoes:

      rest = [c for c in est.index if c != estacao]
      for r in rest:
          distances[estacao][r] = Calculate_Dist(*est.loc[estacao,:].to_list(),\
                                                *est.loc[r,:].to_list())
  return distances

if __name__ == '__main__':

  # Get Paths and Load Data
  root = os.getcwd()
  data_path = pjoin(root, 'data/cleandata/Info pluviometricas/Merged Data/merged.csv')
  regions_path = pjoin(root, 'data/cleandata/Info pluviometricas/Merged Data/error_regions.csv')
  estacoes_path = pjoin(root, 'data/cleandata/Estacoes/lat_lng_estacoes.csv')

  save_path = pjoin(root, 'data/cleandata/Info pluviometricas/Merged Data/repaired.csv')

  config_path = 'src/Pipeline/config/repair_regions.yaml'
  ow_path = 'data/cleandata/OpenWeather/history_bulk.csv'

  data = pd.read_csv(data_path, sep=';',
                     dtype={'Local_0': object, 'Local_1': object,
                            'Local_2': object, 'Local_3': object})

  df_ow = pd.read_csv(ow_path, sep =';')
  df_ow['Data_Hora'] = pd.to_datetime(df_ow['Data_Hora'], yearfirst = True)
  df_ow = df_ow.drop_duplicates(subset = 'Data_Hora' )

  est = pd.read_csv(estacoes_path, sep = ';')
  est = est.iloc[:-1, :]
  est = est.set_index('Estacao')

  regions = pd.read_csv(regions_path, sep =';')

  config = read_yaml(os.path.join(root, config_path))

  df = data.merge(regions, on = 'Data_Hora')

  # Transform Datatime features
  df[['Date', 'Time']] = df['Data_Hora'].str.split(expand=True)
  df[['Hora', 'Min', 'Seg']] = df['Time'].str.split(':', expand=True)
  df[['Ano', 'Mes', 'Dia']] = df['Date'].str.split('-', expand = True)
  df['Hora'] = df['Hora'].astype(int); df['Min']  = df['Min'].astype(int)
  df['Ano']  = df['Ano'].astype(int) ; df['Mes']  = df['Mes'].astype(int)
  df['Dia']  = df['Dia'].astype(int)
  df['Data_Hora'] = pd.to_datetime(df['Data_Hora'])

  # Order Columns by Number of Errors
  r = re.compile(".*error")
  cols = list(filter(r.match, df.columns.to_list()))
  df_cols = pd.DataFrame(columns=['Coluna', 'Valor'])
  for col in cols:
      val = round(df[col].value_counts()[True] / df.shape[0] * 100, 2)
      if col.split('_')[0] in config.keys():
        df_cols.loc[df_cols.shape[0]] = [col, val]

  df_cols = df_cols.sort_values(by='Valor').reset_index(drop=True)
  df_cols.iloc[:, 0] = df_cols.iloc[:, 0].str.replace("_error", "")

  # Define Operation on Features from yaml configuration file
  interpolation_cols = [col for col in df_cols.iloc[:,0] if 'interpolation' in config[col.split('_')[0]] ]
  regression_cols    = [col for col in df_cols.iloc[:,0] if 'regression'    in config[col.split('_')[0]] ]
  idw_cols           = [col for col in df_cols.iloc[:,0] if 'idw'           in config[col.split('_')[0]] ]
  fill_ow_cols       = [col for col in df_cols.iloc[:,0] if 'fill_ow'       in config[col.split('_')[0]] ]

  idw_cols     = [col for col in idw_cols     if config[col.split('_')[0]]['idw']]     # Check if True
  fill_ow_cols = [col for col in fill_ow_cols if config[col.split('_')[0]]['fill_ow']] # Check if True

  # Start Operations
  # =========================

  ## Interpolation -------------------
  if interpolation_cols:
    logging.info('## Interpolation')
  for i, col in enumerate(interpolation_cols):
    logging.info(f'## ({i+1}/{len(regression_cols)}) Interpolating Data {col}')

    label = col.split('_')[0]

    error = df[df_cols.iloc[i, 0] + '_error'].copy(deep=True)
    Series = df.loc[:,col].copy(deep=True)

    kwargs = config[label]['interpolation']
    n_Series, n_error, error_interpol = interpolation(
                                        Series, error, **kwargs)

    df.loc[:, col+'_error'] = n_error
    df.loc[:, col+'_interpol'] = error_interpol
    df.loc[:, col+'_pred'] = n_Series
    df.loc[:, col] = df.loc[:, col+'_pred'].copy()

  if interpolation_cols:
    logging.info(f'## Done interpolating Data\n')

  ## Regression -------------------
  if regression_cols:
    logging.info('## XGB Regression')
  for i, col in enumerate(regression_cols):
    logging.info(f'({i+1}/{len(regression_cols)}) Training for {col}')

    label = col.split('_')[0]
    kwargs = config[label]['regression']
    repair_regions(df, col, **kwargs)
    # Remove Error From Repaired Columns
    df.loc[df[f'{col}_regression'], f'{col}_error'] = False

    # Update Data
    df.loc[:, col] = df.loc[:, col+'_pred'].copy()

    logging.info(f'({i+1}/{len(df_cols)}) Done training for {label}\n')

  ## IDW Interpolation ------------
  if idw_cols:
    distances = get_estacaoes_dist(est)
    logging.info('## IDW Interpolation')

  for i, col in enumerate(idw_cols):
    logging.info(f'({i+1}/{len(idw_cols)}) IDW Interpolatin for {col}')

    label, num = col.split('_')
    num = int(num)

    df.insert(df.shape[1],'temp', np.nan)
    df.loc[df[f'{col}_error'], 'temp'] = \
             df[df[f'{col}_error']].apply(idw_interpolate,
                                                args = (num, label, distances),
                                                axis = 1 ).copy()


    try:
      # If _pred does not exists creates a copy of _label
      df.insert(df.shape[1], f'{col}_pred', df[f'{col}'].copy())
    except ValueError:
      # Column already exists
      pass
    df.loc[~df['temp'].isna(), f'{col}_pred'] = df.loc[~df['temp'].isna(), 'temp']
    df.loc[~df['temp'].isna(), f'{col}_error'] = False
    df.insert(df.shape[1], f'{col}_idw', False)
    df.loc[~df['temp'].isna(), f'{col}_idw'] = True
    logging.info(f'{df[~df["temp"].isna()].shape[0]} Samples Interpolated - {col}')
    df = df.drop(columns = ['temp'])

    # Update Data
    df.loc[:, col] = df.loc[:, col+'_pred'].copy()

  if idw_cols:
    logging.info('## IDW Interpolation done\n')

  ## Fill with OpenWeather --------
  if fill_ow_cols:
    logging.info('## Imputation With OpenWeather')

  for i, col in enumerate(fill_ow_cols):
    logging.info(f'({i+1}/{len(fill_ow_cols)}) Open Weather repair for {col}')

    label, num = col.split('_')
    num = int(num)

    try:
      # If _repaired does not exists creates a copy of _label
      df.insert(df.shape[1], f'{label}_{i}_pred', df[f'{label}_{i}'].copy())
    except ValueError:
      # Column already exists
      pass

    df.loc[df[f'{label}_{i}_error'], f'{label}_{i}_pred'] = \
           df[df[f'Precipitacao_{i}_error']].apply(fill_ow, args = (i, label ,df_ow),
                                                   axis = 1 ).copy()

    df.insert(df.shape[1], f'{label}_{i}_fill_ow', False)
    df.loc[df[f'{label}_{i}_error'], f'{label}_{i}_fill_ow'] = True
    df[f'{label}_{i}_error'] = False

    # Update Data
    df.loc[:, col] = df.loc[:, col+'_pred'].copy()

  # Save Data
  # ===========
  save_cols = ['Data_Hora'] + [i for i in df.columns if (('_interpol' in i) or
                               ('_pred' in i) or ('_regression' in i) or
                               ('_idw' in i) or ('_fill_ow' in i) or ('_error' in i))]

  if os.path.exists(save_path):
    df_repaired = pd.read_csv(save_path, sep = ';')
    for col in save_cols:
      df_repaired[col] = df[col]

    df_repaired[save_cols].to_csv(save_path, decimal='.', sep=';', index=False)

  else:
    df[save_cols].to_csv(save_path, decimal='.', sep=';', index=False)
