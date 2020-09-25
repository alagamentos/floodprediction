from utils import *
import logging
import pandas as pd
import numpy as np
from os.path import join as pjoin
import os
from pathlib import Path
from datetime import timedelta
import concurrent
import xgboost
import plotly.graph_objects as go

root = Path(os.path.dirname(os.path.realpath(__file__))).parent.parent

def get_columns(df, feature):
  d = {c.replace('_error','') : df[c].sum() for c in df.columns if feature in c if '_error' in c}
  return [k for k, v in sorted(d.items(), key=lambda item: item[1])]


def get_distances(est):
  """

  Calculates distances between every given station

  :param est: DataFrame containing latitude and longitude for every station
  :type est: ``DataFrame``

  :return: Dictonary were keys are station names and value is distance between stations
  :rtype: ``dict``

  """

  estacoes = list(est.index)

  distances = {k: {} for k in estacoes}

  for estacao in estacoes:

      rest = [c for c in est.index if c != estacao]
      for r in rest:
          distances[estacao][r] = calculate_distance(*est.loc[estacao,:].to_list(),\
                                                *est.loc[r,:].to_list())
  return distances


def read_estacoes():

  estacoes_path = pjoin(root, 'data/cleandata/Estacoes/lat_lng_estacoes.csv')
  est = pd.read_csv(estacoes_path, sep = ';')
  est = est.iloc[:-1, :]
  est = est.set_index('Estacao')

  return est


def read_openweather():
  ow_path = pjoin(root,'data/cleandata/OpenWeather/history_bulk.csv')
  df_ow = pd.read_csv(ow_path, sep =';')
  df_ow['Data_Hora'] = pd.to_datetime(df_ow['Data_Hora'], yearfirst = True)
  df_ow = df_ow.drop_duplicates(subset = 'Data_Hora' )

  return df_ow


def calculate_distance(lat1, lon1, lat2, lon2):
  """

  Calculate the distance in km between to given points (pairs of latitude and longitude)

  :param lat1: Latitude value (Degrees) for the point 1
  :type lat1: ``float``

  :param lon1: Longitude value (Degrees) for the point 1
  :type lon1: ``float``

  :param lat2: Latitude value (Degrees) for the point 2
  :type lat2: ``float``

  :param lon1: Longitude value (Degrees) for the point 2
  :type lon1: ``float``

  :return: Distance in km
  :rtype: ``float``

  """
  r = 6371
  phi1 = np.radians(lat1)
  phi2 = np.radians(lat2)
  delta_phi = np.radians(lat2 - lat1)
  delta_lambda = np.radians(lon2 - lon1)
  a = np.sin(delta_phi / 2)**2 + np.cos(phi1) *\
      np.cos(phi2) * np.sin(delta_lambda / 2)**2
  res = r * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))
  return np.round(res, 2)


def _interpolation(Series, error, max_interpolation_size=5, interpolate_kwargs = {}):
  """

  Linear interpolation for a given Series

  :param Series: Series with the data to interpolate
  :type Series: ``Series``

  :param error: List containing the regions in which to interpolate
  :type error: ``list`` of ``bool``

  :param max_interpolation_size: Maximum number of samples to interpolate, defaults to 5
  :type max_interpolation_size: ``int``

  """

  error_reg = list_2_regions(error)
  reg_size = [i[1] - i[0] for i in error_reg]

  interpol_reg = [error_reg[i] for i in range(len(reg_size))
                               if reg_size[i] <= max_interpolation_size]

  logging.info(f'Regions to interpolate {len(interpol_reg)} - {Series.name}')

  interpol_reg = regions_2_list(interpol_reg, len(Series))

  # Series.fillna(-100000)
  Series.loc[interpol_reg] = np.nan

  Series = Series.interpolate(limit=max_interpolation_size, **interpolate_kwargs)
  # Series.loc[Series == -100000] = np.nan

  return Series, interpol_reg


def interpolation(df, feature, kwargs):
  """

  """

  data_columns = get_columns(df, feature)
  for col in data_columns:
    try:
      df.insert(df.shape[1], col+'_interpol', False)
    except ValueError:
      pass

    series = df.loc[:, col].copy()
    error = df.loc[:, col+'_error'].copy()
    series_c, corrected = _interpolation(series, error,  **kwargs)

    df.loc[corrected, col+'_error'] = False
    df.loc[:, col] = series_c.copy()
    df.loc[:, col+'_interpol'] = corrected.copy()

  return df


def _idw_row(row , num, label, distances):
  """

  Inverse distance weighting interpolation for a given DataFrame row

  :param row: DataFrame row
  :type row: ``Series``

  :param num: Station number to repair
  :type num: ``int``

  :param label: Feature label to repair
  :type label: ``str``

  :param distances: Dictonary containing distances between stations
  :type distances: ``dict``

  :return: Interpolated value
  :rtype: ``float``

  """

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


def idw(df, feature, kwargs):

  data_columns = get_columns(df, feature)
  est = read_estacoes()
  distances = get_distances(est)

  if not kwargs:
    return

  for col in data_columns:

    num = int(col.split('_')[1])

    df.insert(df.shape[1],'temp', np.nan)
    df.loc[df[f'{col}_error'], 'temp'] = df[df[f'{col}_error']].apply(_idw_row,
                                                                      args = (num, feature, distances),
                                                                      axis = 1 ).copy()

    df.loc[~df['temp'].isna(), col] = df.loc[~df['temp'].isna(), 'temp']
    df.loc[~df['temp'].isna(), f'{col}_error'] = False

    try:
      df.insert(df.shape[1], f'{col}_idw', False)
    except ValueError:
      pass
    df.loc[~df['temp'].isna(), f'{col}_idw'] = True

    logging.info(f'{df[~df["temp"].isna()].shape[0]} Samples Interpolated - {col}')

    df = df.drop(columns = ['temp'])

  return df


def _fill_ow_row(row , label, df_ow):
  """

  Completes faulty data with OpenWeather values, rounds faulty data to the hour.

  :param row: DataFrame row
  :type row: ``Series``

  :param label: Feature label to repair
  :type label: ``str``

  :param df_ow: OpenWeather DataFrame
  :type df_ow: ``DataFrame``

  :return: OpenWeather value
  :rtype: ``DataFrame``
  """

  rounded_hour = row['Data_Hora'].round('H')
  mask = pd.to_datetime(df_ow['Data_Hora']) == rounded_hour
  try:
    return df_ow.loc[mask,label].item()
  except ValueError:
    mask = pd.to_datetime(df_ow['Data_Hora']) == rounded_hour + timedelta(hours=1)
    return df_ow.loc[mask,label].item()


def fill_ow(df, feature, kwargs):

  data_columns = get_columns(df, feature)
  df_ow = read_openweather()

  if not kwargs:
    return

  for col in data_columns:

    df.loc[df[f'{col}_error'], col] = df[df[f'{col}_error']].apply(_fill_ow_row,
                                                                             args = (feature, df_ow),
                                                                             axis = 1 ).copy()
    try:
      df.insert(df.shape[1], f'{col}_fill_ow', False)
    except ValueError:
      pass
    df.loc[df[f'{col}_error'], f'{col}_fill_ow'] = True
    df[f'{col}_error'] = False

  return df


def _predict_region(xgb, df_recurrent, label, dcols, p, lookbackSize):
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


def _repair_regions(df, label, max_region_size=None, lookbackSize=None, extra_features=None, **kwargs):
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

  y_train = df_error[label]

  drop_columns_2 = [c for c in df_error.columns if (not 'delay' in c) and
                                                 (not c in extra_features)]
  df_error = df_error.drop(columns = drop_columns_2)

  try:
    xgb = xgboost.XGBRegressor(objective='reg:squarederror', tree_method='gpu_hist')
    xgb.fit(df_error, y_train) # Ele tenta executar pela GPU apenas ao rodar o FIT
  except:
    xgb = xgboost.XGBRegressor(objective='reg:squarederror')
    xgb.fit(df_error, y_train) # Ele tenta executar pela GPU apenas ao rodar o FIT

  drop_cols = [i for i in df_att.columns if 'delay_error' in i] + \
      [label] + [label+'_error']

  df_recurrent = df_att.drop(columns=drop_cols + drop_columns_2).copy(deep=True)
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
    futures = {executor.submit(_predict_region, xgb, df_recurrent,
                               label, dcols, p, lookbackSize): p for p in predict_regions}

    for future in concurrent.futures.as_completed(futures):
      pr = futures[future]
      try:
        data = future.result()
        df.loc[pr[0]:pr[1]-1, label + '_pred' ] = data
      except Exception as exc:
        print(f'Generated an exception: {exc} - {pr[0]} - {pr[1]}')
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

  return df


def regression(df, feature, kwargs):
  data_columns = get_columns(df, feature)

  df_copy = df.copy()

  for col in data_columns:

    id = df_copy.index
    df_copy = _repair_regions(df_copy.reset_index(drop = True), col, **kwargs)
    df_copy.index = id

    df.loc[:, col] = df_copy.loc[:, col + '_pred']

    try:
      df.insert(df.shape[1],col+'_regression', df_copy.loc[:, col+'_repaired'] )
    except ValueError:
      df.loc[:, col+'_regression'] = df_copy.loc[:, col+'_repaired']

    df.loc[df_copy[col + '_repaired'].to_list(), col + '_error'] = False

    df_copy.drop(columns = [col + '_pred', col + '_repaired'])

  return df
