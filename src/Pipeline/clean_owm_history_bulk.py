import pandas as pd
import numpy as np
import os
from os import mkdir
from os.path import join as pjoin
import logging


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO, format='## Clean OpenWeather - %(levelname)s: %(message)s')

  root = os.getcwd()
  load_path = pjoin(root, 'data/rawdata/OpenWeather/history_bulk.csv')
  save_path = pjoin(root, 'data/cleandata/OpenWeather/')

  if not os.path.exists(save_path):
    mkdir(save_path)
    logging.info(f' creating directory: {save_path}')

  save_path += 'history_bulk.csv'

  df = pd.read_csv(load_path)

  # Format Date and Time
  logging.info(f' Removing time zones')
  df.insert(0, 'Data_Hora', np.nan)
  df['Data_Hora'] = pd.to_datetime(df['dt_iso'].str[:-10])
  df['Data_Hora'] = df.apply(lambda x: x['Data_Hora'] +
                             pd.Timedelta(hours=x['timezone'] / 3600), axis=1)
  df = df[df['Data_Hora'] > '2010-01-01']

  # Select and Rename Columns
  drop_cols = ['dt', 'dt_iso', 'timezone', 'city_name',
               'lat', 'lon', 'weather_main', 'weather_id', 'weather_icon',
               'snow_1h', 'snow_3h', 'rain_3h', 'sea_level', 'grnd_level']

  rename_cols = {'pressure': 'PressaoAtmosferica',
                 'humidity': 'UmidadeRelativa',
                 'wind_speed': 'VelocidadeDoVento',
                 'wind_deg': 'DirecaoDoVento',
                 'rain_1h': 'Precipitacao',
                 'feels_like': 'SensacaoTermica',
                 'temp': 'TemperaturaDoAr'}

  df = df.drop(columns=drop_cols).rename(columns=rename_cols)

  # Fill Data
  df['Precipitacao'] = df['Precipitacao'].fillna(0)

  logging.info(f' saving file')
  df.to_csv(save_path, sep=';', index=False)
