import pandas as pd
import logging

logging.basicConfig(level=logging.INFO,
                    format='## get_labels_hour - %(levelname)s: %(message)s ')

if __name__== '__main__':

  rain_threshold = 2 # mm/h

  open_weather_path = 'data/cleandata/OpenWeather/history_bulk.csv'
  ip_path = 'data/cleandata/Info pluviometricas/Merged Data/merged.csv'
  ords_path = 'data/cleandata/Ordens de serviço/labels_day.csv'

  save_path = 'data/cleandata/Ordens de serviço/labels_hour.csv'

  ow = pd.read_csv(open_weather_path, sep = ';')

  ip = pd.read_csv(ip_path,
                  sep = ';',
                  dtype = {'Local_0': object, 'Local_1':object,
                            'Local_2':object,  'Local_3':object})

  ords = pd.read_csv(ords_path,
                     sep = ';')


  ## TO DO:
  ## Merge (merged.csv) With repaired data (repaired.csv)
  ## - Use correct data for removing OrdensServico
  ## =======================

  # OrdensServico
  ords['Data'] = pd.to_datetime(ords['Data'], yearfirst = True)
  for i in range(5):
      ords[f'LocalMax_{i}'] =  ords[f'LocalMax']


  # InfoPluviometrica
  ip['Data_Hora'] = pd.to_datetime(ip['Data_Hora'], yearfirst=True)
  ip.insert(0, 'Data', ip.loc[:,'Data_Hora'].dt.date)
  ip.insert(0, 'Hora', ip.loc[:,'Data_Hora'].dt.hour)

  precipitacao_cols = [c for c in ip.columns if 'Precipitacao'in c]
  precipitacao_cols += ['Data', 'Hora']

  df_p = ip[ip.loc[:,'Data_Hora'].dt.minute == 0][precipitacao_cols].reset_index(drop=True)
  df_p['Data'] = pd.to_datetime(df_p['Data'], yearfirst = True)


  # OpenWeather
  ow['Data_Hora'] = pd.to_datetime(ow['Data_Hora'], yearfirst=True)
  ow.insert(0, 'Data', ow.loc[:,'Data_Hora'].dt.date)
  ow.insert(0, 'Hora', ow.loc[:,'Data_Hora'].dt.hour)
  ow = ow[~ow['Data_Hora'].duplicated(keep = 'first')]
  ow = ow[['Data','Hora','Precipitacao']]
  ow['Data'] = pd.to_datetime(ow['Data'], yearfirst = True)


  # Merge

  # Merge Infopluviometrica with OpenWeather
  df_m = df_p.merge(ow, how = 'outer', on = ['Data','Hora']).sort_values(by = ['Data', 'Hora'])

  # Merge with OrdensServico
  df_m = df_m.merge(ords, on = 'Data', how = 'outer')

  # Clean Merged
  df_m = df_m.fillna(0)
  df_m = df_m.rename(columns = {'Precipitacao':'Precipitacao_5'})
  df_m.insert(0,'Data_Hora', 0 )
  df_m['Data_Hora'] = pd.to_datetime(df_m['Data'].astype(str) + ' ' +
                                    df_m['Hora'].astype(str) + ':00:00', yearfirst=True)
  df_m = df_m.rename(columns = {'LocalMax_ow':'LocalMax_5'})

  # Remove OrdemServico when rain is under threshold
  for i in range(6):
    n_remove = len(df_m.loc[(df_m[f'Precipitacao_{i}'].fillna(0) < rain_threshold) &
                            (df_m[f'LocalMax_{i}'] == 1), f'LocalMax_{i}'])
    logging.info(f'Removing {n_remove} OrdensServico from LocalMax')
    df_m.loc[df_m[f'Precipitacao_{i}'].fillna(0) < rain_threshold, f'LocalMax_{i}'] = 0


  lm_cols = [c for c in df_m.columns if 'LocalMax_' in c]
  n_remove = len(df_m.loc[(df_m[lm_cols].max(axis = 1) == 0) &
                          (df_m['LocalMax'] == 1), 'LocalMax'])
  logging.info(f'Removing {n_remove} OrdensServico from LocalMax')
  df_m.loc[(df_m[lm_cols].max(axis = 1) == 0) &
                          (df_m['LocalMax'] == 1), 'LocalMax'] = 0

  # Clean up output
  interest_cols = [c for c in df_m.columns if 'Local' in c]
  df_m = df_m[['Data_Hora']  + interest_cols]
  df_m.head()

  # Export CSV
  saving_info = f'Saving labels_day.csv to path:\n\t\t | - {save_path}'
  logging.info(saving_info)
  df_m.to_csv(save_path, sep = ';', index = False)
