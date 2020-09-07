from utils import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from os.path import join as pjoin
from pathlib import Path
import logging
import yaml

logging.basicConfig(level=logging.INFO,
                    format='## find_regions - %(levelname)s: %(message)s ')


def get_error_regions(df,
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

def read_yaml(path):
  with open(path) as file:
    return yaml.load(file, Loader=yaml.FullLoader)

if __name__ == '__main__':
  root = Path(__file__).resolve().parents[2]  # resolve() -> absolute path
  data_path = pjoin(
      root, 'data/cleandata/Info pluviometricas/Merged Data/merged.csv')
  save_path = pjoin(
      root, 'data/cleandata/Info pluviometricas/Merged Data/error_regions.csv')
  config_path = 'src/Pipeline/config/error_regions.yaml'

  df = pd.read_csv(data_path,
                   sep=';',
                   dtype={'Local_0': object, 'Local_1': object,
                          'Local_2': object, 'Local_3': object})

  config = read_yaml(os.path.join(root, config_path))

  i = 1

  for key in config.keys():
    msg = str(i) + '/' + str(len(config.keys()))
    logging.info(msg)

    cols = [i for i in df.columns if key in i]

    for col in cols:
      logging.info(col)
      new_col = col + "_error"
      df[new_col] = get_error_regions(df[col], **config[key])

    i += 1


  save_cols = ['Data_Hora'] + [i for i in df.columns if '_error' in i]

  df[save_cols].to_csv(save_path, decimal='.', sep=';', index=False)
  logging.info('New file created at:' + save_path)
  logging.info('Done!')
