import yaml
import pandas as pd
import sys
from repair_regions_functions import *
import logging

logging.basicConfig(level=logging.INFO,
                    format='## Repair - %(asctime)s - %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

def read_yaml(path):
  with open(path) as file:
    return yaml.load(file, Loader=yaml.FullLoader)

def include_datetime_features(df):
  df[['Date', 'Time']] = df['Data_Hora'].str.split(expand=True)
  df[['Hora', 'Min', 'Seg']] = df['Time'].str.split(':', expand=True)
  df[['Ano', 'Mes', 'Dia']] = df['Date'].str.split('-', expand = True)
  df['Hora'] = df['Hora'].astype(int); df['Min']  = df['Min'].astype(int)
  df['Ano']  = df['Ano'].astype(int) ; df['Mes']  = df['Mes'].astype(int)
  df['Dia']  = df['Dia'].astype(int)
  df['Data_Hora'] = pd.to_datetime(df['Data_Hora'])

if __name__ == "__main__":

  config_path = 'src/Pipeline/config/repair_regions.yaml'
  error_regions_path = 'data/cleandata/Info pluviometricas/Merged Data/error_regions.csv'
  merged_path = 'data/cleandata/Info pluviometricas/Merged Data/merged.csv'

  save_path = pjoin(root, 'data/cleandata/Info pluviometricas/Merged Data/repaired.csv')

  regions = pd.read_csv(error_regions_path, sep = ';')
  merged = pd.read_csv(merged_path, sep = ';')

  df = regions.merge(merged, on = 'Data_Hora')
  include_datetime_features(df)

  # Transform datetime features

  try:
    df = df.drop(columns = ['index'])
  except:
    pass

  config = read_yaml(config_path)

  available_functions = [idw, fill_ow, interpolation, regression]

  functions = {}
  for fun in available_functions:
    functions[fun.__name__] = fun

  for i, feature in enumerate(config.keys()):
    logging.info('='*30)
    logging.info(f'{i+1}/{len(config.keys())} - Starting {feature}')
    for j, (cfunction, ckwargs) in enumerate(config[feature].items()):
      logging.info(f'{j+1}/{len(config[feature].keys())} - Applying {cfunction} to {feature} with {ckwargs}')
      df = functions[cfunction](df, feature, ckwargs)

    pred_cols = [c.replace('_error', '') for c in df.columns if '_error' in c]
    for c in pred_cols:
      df.rename(columns = {c:f'{c}_error'})
    logging.info(f'{feature} Done')
    logging.info('='*30 + '\n')

  drop_cols = [c for c in df.columns if 'Local' in c]
  drop_cols += ['Hora', 'Min', 'Seg', 'Ano', 'Mes', 'Dia', 'Time', 'Date']

  for c in drop_cols:
    try:
      df = df.drop(columns = [c])
    except:
      pass

  # Save
  df.to_csv(save_path, sep = ';', index = False)

