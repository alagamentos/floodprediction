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
#pip install kaleido

pd.options.plotting.backend = "plotly"

#fig = df.plot()
#fig.write_image()

logging.basicConfig(level=logging.INFO,
                    format='## Repair - %(asctime)s - %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


def interpolation(Series, error, max_size=5, **kwargs):
  error_reg = list_2_regions(error)
  reg_size = [i[1] - i[0] for i in error_reg]

  interpol_reg = [error_reg[i]
                  for i in range(len(reg_size)) if reg_size[i] <= max_size]
  error_interpol = regions_2_list(interpol_reg, len(Series))

  error.loc[error_interpol] = False

  Series.fillna(-100000)
  Series.loc[error_interpol] = np.nan

  Series = Series.interpolate(method='linear', limit=max_size)
  Series.loc[Series == -100000] = np.nan

  return Series, error.tolist()


def predict_region(xgb, prediction, df_recurrent, label, dcols, p, lookbackSize):
    df_recurrent = df_recurrent.copy(deep=True)
    start, stop = p[0], p[1]
    y_predict = []
    for j in range(start, stop):
      pred_ = xgb.predict(df_recurrent.loc[[j]])
      #prediction.loc[j,label] = pred_
      y_predict.append(pred_)
      matrix = df_recurrent.loc[j+1:j+1+lookbackSize, dcols].values
      np.fill_diagonal(matrix, pred_)
      df_recurrent.loc[j+1:j+1+lookbackSize, dcols] = matrix
    #return prediction, start, stop
    return y_predict


def repair_regions(df, label, max_size=None, lookbackSize=None, extra_features=None):
  """
  df
  label
  max_size
  lookbackSize
  extra_features - list
  """

  df[['Date', 'Time']] = df['Data / Hora'].str.split(expand=True)
  df[['Hora', 'Min', 'Seg']] = df['Time'].str.split(':', expand=True)
  df['Hora'] = df['Hora'].astype(int)
  df['Min'] = df['Min'].astype(int)
  df['Seg'] = df['Seg'].astype(int)

  attribute = label.split('_')[0]
  cols_attribute = [i for i in df.columns if attribute in i]
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
  xgb = xgboost.XGBRegressor(
      objective='reg:squarederror', tree_method='gpu_hist')
  xgb.fit(df_error.drop(columns=[label]), df_error[label])

  drop_cols = [i for i in df_att.columns if 'delay_error' in i] + \
      [label] + [label+'_error']

  df_recurrent = df_att.drop(columns=drop_cols).copy(deep=True)
  prediction = df_att[[label, label + '_error']].copy(deep=True)

  predict_regions = list_2_regions(df_att[label + '_error'])

  logging.info(f'Number of regions: {len(predict_regions)}')

  dcols = ['delay_' + str(i+1) for i in range(lookbackSize)]
  # time_start = time.time()

  # for p in predict_regions[0:30]:
  #   for i in range(p[0], p[1]):
  #     pred_ = xgb.predict(df_recurrent.loc[[i]])
  #     prediction.loc[i, label] = pred_
  #     matrix = df_recurrent.loc[i + 1 : i + lookbackSize + 1, dcols].values
  #     np.fill_diagonal(matrix, pred_)
  #     df_recurrent.loc[i + 1 : i + lookbackSize + 1, dcols] = matrix

  with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    futures = {executor.submit(predict_region, xgb, prediction, df_recurrent,
                               label, dcols, p, lookbackSize): p for p in predict_regions[0:30]}

    for future in concurrent.futures.as_completed(futures):
      pr = futures[future]
      try:
        data = future.result()
        prediction.loc[pr[0]:pr[1]-1, label] = data
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
        x=df_att[label][start:stop].index,
        y=df_att[label][start:stop],
        mode='lines', name='Real', connectgaps=False))
    fig.add_trace(go.Scatter(
        x=prediction[label][start:stop]
        .where(prediction[label + '_error'][start:stop]).index,
        y=prediction[label][start:stop]
        .where(prediction[label + '_error'][start:stop]),
        mode='lines', name='Prediction'))

    fig.write_image(f'images/{label}_{i}.jpg')


if __name__ == '__main__':
  root = os.getcwd()
  data_path = pjoin(
      root, 'data/cleandata/Info pluviometricas/Merged Data/merged_wRegions.csv')

  # save_path = pjoin(
  #     root, '')

  df = pd.read_csv(data_path,
                   sep=';',
                   dtype={'Local_0': object, 'Local_1': object,
                          'Local_2': object, 'Local_3': object})

  config = {
      'UmidadeRelativa':
      {'max_size': 5,
       'lookbackSize': 30,
       'extra_features': ['Hora']},

      'PressaoAtmosferica':
      {'max_size': 5,
       'lookbackSize': 30,
       'extra_features': ['Hora']},

      'Temperatura do Ar':
      {'max_size': 5,
       'lookbackSize': 30,
       'extra_features': ['Hora']},

      'TemperaturaInterna':
      {'max_size': 5,
       'lookbackSize': 30,
       'extra_features': ['Hora']},

      'PontoDeOrvalho':
      {'max_size': 5,
       'lookbackSize': 30,
       'extra_features': ['Hora']},

      'SensacaoTermica':
      {'max_size': 5,
       'lookbackSize': 30,
       'extra_features': ['Hora']},

      'RadiacaoSolar':
      {'max_size': 5,
       'lookbackSize': 30,
       'extra_features': ['Hora']},

      'DirecaoDoVento':
      {'max_size': 5,
       'lookbackSize': 30,
       'extra_features': ['Hora']},

      'VelocidadeDoVento':
      {'max_size': 5,
       'lookbackSize': 30,
       'extra_features': ['Hora']},

      'Precipitacao':
      {'max_size': 5,
       'lookbackSize': 30,
       'extra_features': ['Hora']},
  }

r = re.compile(".*error")
cols = list(filter(r.match, df.columns.to_list()))
cols = [i for i in cols if 'Precipitacao' not in i]

df_cols = pd.DataFrame(columns=['Coluna', 'Valor'])

for col in cols:
    #print(col)
    #display(df[col].value_counts())
    val = round(df[col].value_counts()[True] / df.shape[0] * 100, 2)
    df_cols.loc[df_cols.shape[0]] = [col, val]

df_cols = df_cols.sort_values(by='Valor').reset_index(drop=True)
df_cols.iloc[:, 0] = df_cols.iloc[:, 0].str.replace("_error", "")

logging.info('## Interpolating Data')
for i in range(len(df_cols)):
  label = df_cols.iloc[i, 0]

  error = df[df_cols.iloc[i, 0] + '_error'].copy(deep=True)
  Series = df[df_cols.iloc[i, 0]].copy(deep=True)

  n_Series, n_error = interpolation(
      Series, error, **config[label.split('_')[0]])
  df.loc[:, df_cols.iloc[i, 0] + '_error'] = n_error
  df.loc[:, df_cols.iloc[i, 0]] = n_Series

logging.info('XGB')
for i in range(len(df_cols)):
  label = df_cols.iloc[i, 0]
  logging.info(f'({i}/{len(df_cols)}) Training for {label}')
  repair_regions(df, label, **config[label.split('_')[0]])
  logging.info(f'({i}/{len(df_cols)}) Done training for {label}')
