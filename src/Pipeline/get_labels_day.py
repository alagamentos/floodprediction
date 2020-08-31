import pandas as pd
import logging
from utils import *

logging.basicConfig(level=logging.INFO,
                    format='## get_labels - %(levelname)s: %(message)s ')

def Calculate_Dist(lat1, lon1, lat2, lon2):
    r = 6371
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2)**2 + np.cos(phi1) *\
        np.cos(phi2) *   np.sin(delta_lambda / 2)**2
    res = r * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))
    return np.round(res, 2)

def get_distances(estacoes, ord_serv):
    for index, row in ord_serv.iterrows():
        dist = estacoes.apply(lambda x:
                           Calculate_Dist(row['lat'], row['lng'],
                                            x['lat'],   x['lng']),
                           axis=1)
        ord_serv.loc[index,'Distance'], arg = dist.min(), dist.argmin()
        ord_serv.loc[index,'Estacao'] = estacoes.iloc[arg,0]

    return ord_serv

if __name__== '__main__':

  rain_threshold = 10 # mm/dia
  distance_threshold = 4.5 # km - Max for each station

  open_weather_path = 'data/cleandata/OpenWeather/history_bulk.csv'
  ip_path = 'data/cleandata/Info pluviometricas/Merged Data/merged.csv'
  ords_path = 'data/cleandata/Ordens de serviço/Enchentes_LatLong.csv'
  est_path = 'data/cleandata/Estacoes/lat_lng_estacoes.csv'

  save_path = 'data/cleandata/Ordens de serviço/labels_day.csv'

  ow = pd.read_csv(open_weather_path, sep = ';',
                  parse_dates = ['Data_Hora'])

  ip = pd.read_csv(ip_path,
                  sep = ';',
                  dtype = {'Local_0': object, 'Local_1':object,
                            'Local_2':object,  'Local_3':object},
                  parse_dates = ['Data_Hora'])

  ords = pd.read_csv(ords_path,
                  sep = ';')

  est = pd.read_csv( est_path, sep = ';')
  est = est.iloc[:-1]


  ## TO DO:
  ## Merge (merged.csv) With repaired data (repaired.csv)
  ## - Use correct data for removing OrdensServico
  ## =======================

  # Group OrdensServico by Date - Count()
  ords['Data'] = pd.to_datetime(ords['Data'], yearfirst=True)
  ords_gb = ords.fillna(0).groupby('Data').count().max(axis = 1).to_frame().reset_index()
  ords_gb.columns = ['Data', 'OrdensServico']

  # Group Precipitacao by Date - Sum()
  precipitacao_cols = ['Data'] + [c for c in ip.columns if 'Precipitacao'in c]
  ip.insert(0,'Data', ip.loc[:,'Data_Hora'].dt.date)
  ip.insert(0,'Time', ip.loc[:,'Data_Hora'].dt.time)
  df_p = ip[ip.loc[:,'Data_Hora'].dt.minute == 0][precipitacao_cols].groupby('Data').sum().reset_index()
  df_p['Data'] = pd.to_datetime(df_p['Data'], yearfirst=True)

  # Group OpenWeather by Date
  ow.insert(0,'Data', ow.loc[:,'Data_Hora'].dt.date)
  ow.insert(0,'Time', ow.loc[:,'Data_Hora'].dt.time)
  ow_gb = ow.groupby('Data').sum()[['Precipitacao']].reset_index()
  ow_gb['Data'] = pd.to_datetime(ow_gb['Data'], yearfirst = True)
  ow_gb.columns = ['Data', 'Precipitacao_ow']

  # Merge
  df_m = df_p.merge(ords_gb, how = 'outer', on='Data')
  df_m['OrdensServico'] = df_m['OrdensServico'].fillna(0)
  df_m = ow_gb.merge(df_m, on='Data', how = 'outer')

  # LocalMax
  regions = list_2_regions(df_m['OrdensServico'].fillna(0) > 0)
  b_list = regions_2_list(regions, len(df_m))
  regions = list_2_regions(b_list)

  df_m.loc[:, 'LocalMax'] = 0

  df_m['OrdensServico'] = df_m['OrdensServico'].fillna(0)
  for r in regions:
      for i in range(r[0], r[1]+1):
          id_max = df_m.loc[i-3: i+3, 'OrdensServico'].idxmax()
          if i == id_max:
              df_m.loc[i-3: i+3, 'LocalMax'] = 0
              df_m.loc[i, 'LocalMax'] = 1

  total_os = (df_m['OrdensServico'].fillna(0) > 0).sum()
  logging.info(f'{total_os} Ordens de Serviço')
  # Remove LocalMax when rain(mm/day) is less than rain_threshold
  df_m = df_m.rename(columns = {'Precipitacao_ow':'Precipitacao_5'})
  for i in range(6):
      df_m.loc[:, f'LocalMax_{i}'] = df_m['LocalMax']
      df_m.loc[df_m[f'Precipitacao_{i}'] < rain_threshold, f'LocalMax_{i}'] = 0

      n_remove = len(df_m.loc[(df_m[f'Precipitacao_{i}'] >= rain_threshold) &
                              (df_m['LocalMax'] == 1), f'LocalMax_{i}'])
      logging.info(f'Removing {n_remove} OrdensServico from Local_{i}')

  # Separate OrdensServico for each station
  # Station over threshold distance is called "Null"
  ord_serv_region = get_distances(est, ords)
  ord_serv_region.loc[ord_serv_region['Distance'] > distance_threshold, 'Estacao'] = 'Null'
  # Group by Date
  ord_serv_region['Data'] = pd.to_datetime(ord_serv_region['Data'], yearfirst=True)
  ords_gbr = ords.fillna(0).groupby(['Data', 'Estacao']).count().max(axis = 1).to_frame().reset_index()
  ords_gbr.columns = ['Data', 'Estacao', 'OrdensServico']

  # Rename each station respective to its local value
  local_cols = [i for i in ip.columns if 'Local' in i]
  map_estacoes = dict(zip(ip.loc[0,local_cols].values, ip[local_cols].columns))
  # Null -> 5
  map_estacoes['Null'] = 'Local_5'

  # Merge
  ords_gbr = df_m[['Data']].merge(ords_gbr, on = 'Data', how = 'outer')
  ords_gbr = ords_gbr.dropna()
  ords_gbr['Estacao'] = ords_gbr['Estacao'].map(map_estacoes)

  # Create Local Columns
  for i in range(6):
      df_m.insert(len(df_m.columns), f'Local_{i}', 0)

  # Populate Columns with number of OrdensServico for each station
  for d in ords_gbr['Data']:
      aux = ords_gbr[ords_gbr['Data'] == d]
      for local in aux.Estacao.dropna().unique():
          df_m.loc[df_m['Data'] == d, local] = aux.loc[(aux['Estacao'] == local),\
                                                      'OrdensServico'].values[0]

  # Rename columns
  df_m = df_m.rename(columns = {'LocalMax_5':'LocalMax_ow', 'Local_5':'Local_Null'})

  # Select columns of interest different than zero
  interest_cols = [c for c in df_m.columns if 'Local' in c]
  df_m = df_m[['Data']  + interest_cols]
  df_m = df_m[df_m[interest_cols].sum(axis = 1) > 0]

  # Export CSV
  saving_info = f'Saving labels_day.csv to path:\n\t\t\t |-{save_path}'
  logging.info(saving_info)
  df_m.to_csv(save_path, sep = ';', index = False)
