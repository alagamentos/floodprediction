from utils import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from os.path import join as pjoin
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO,
                    format='## find_regions - %(levelname)s: %(message)s ')


def get_regions(df,
                d_threshold=None,
                z_threshold=None,
                nz_threshold=None,
                margins=None,
                plot_derivative=False, derivative_kwargs={},
                plot_zeros=False, zeros_kwargs={},
                plot_final=False, final_kwargs={}):
  '''
  df
  d_threshold
  z_threshold
  plot_derivative
  plot_zeros
  plot_final
  '''

  # Get derivative threshold
  if d_threshold is not None:
    peaks = derivative_threshold(df, d_threshold,
                                 plot=plot_derivative, **derivative_kwargs)
  else:
    peaks = [False] * len(df)

  # Get constant values
  if z_threshold is not None:
    zeros = derivative_zero(df, z_threshold, False,
                            plot=plot_zeros, **zeros_kwargs)
  else:
    zeros = [False] * len(df)

  # Get non zeros constant values
  if nz_threshold is not None:
    non_zeros = derivative_zero(df, nz_threshold, True,
                                plot=plot_zeros, **zeros_kwargs)
  else:
    non_zeros = [False] * len(df)

  # Error Union
  nans = df.isna()
  error = [zeros[i] or peaks[i] or non_zeros[i] for i in range(len(peaks))]
  error_reg = list_2_regions(error)

  # Expand margins
  if margins is not None:
    error_reg = increase_margins(3, error_reg, len(peaks))

  # Include NaNs
  error = regions_2_list(error_reg, len(df))
  error = [nans[i] or error[i] for i in range(len(error))]

  # Plot Results
  if plot_final:
    error_reg = list_2_regions(error)
    plot_regions(df, error_reg, **final_kwargs)

  return error


if __name__ == '__main__':
  root = Path(__file__).resolve().parents[2]  # resolve() -> absolute path
  data_path = pjoin(
      root, 'data/cleandata/Info pluviometricas/Merged Data/merged.csv')
  save_path = pjoin(
      root, 'data/cleandata/Info pluviometricas/Merged Data/regions.csv')

  df = pd.read_csv(data_path,
                   sep=';',
                   dtype={'Local_0': object, 'Local_1': object,
                          'Local_2': object, 'Local_3': object})

  config = {
      'UmidadeRelativa':
      {'d_threshold': 12,
       'z_threshold': 3,
       'margins': 3},

      'PressaoAtmosferica':
      {'d_threshold': 50,
       'z_threshold': 7,
       'margins': 5},

      'TemperaturaDoAr':
      {'d_threshold': 6,
       'z_threshold': 4,
       'margins': 2},

      'TemperaturaInterna':
      {'d_threshold': 6,
       'z_threshold': 4,
       'margins': 3},

      'PontoDeOrvalho':
      {'d_threshold': 3.5,
       'z_threshold': 4,
       'margins': 5},

      'SensacaoTermica': {
          'd_threshold': 4,
          'z_threshold': 10,
          'margins': 3},

      'RadiacaoSolar': {
          'd_threshold': 850,
          'z_threshold': 50,
          'margins': 5,
          'nz_threshold': 3},

      'DirecaoDoVento': {
          'z_threshold': 3,
          'margins': 3},

      'VelocidadeDoVento': {
          'd_threshold': 8,
          'z_threshold': 5,
          'margins': 3},

      'Precipitacao': {}
  }

  i = 1

  for key in config.keys():
    msg = str(i) + '/' + str(len(config.keys()))
    logging.info(msg)

    cols = [i for i in df.columns if key in i]

    for col in cols:
      logging.info(col)
      new_col = col + "_error"
      df[new_col] = get_regions(df[col], **config[key])

    i += 1


  save_cols = ['Data_Hora'] + [i for i in df.columns if '_error' in i]

  df[save_cols].to_csv(save_path, decimal='.', sep=';', index=False)
  logging.info('New file created at:' + save_path)
  logging.info('Done!')
