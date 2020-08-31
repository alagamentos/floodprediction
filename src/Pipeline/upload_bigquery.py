import pandas_gbq
from google.oauth2 import service_account
import pandas as pd

PROJECT_ID = 'temporal-285820'
TABLE_merged = 'info_pluviometrica.merged'
TABLE_error_regions = 'info_pluviometrica.regions'
TABLE_repaired = 'info_pluviometrica.repaired'
TABLE_lat_lng_estacoes = 'estacoes.lat_lng_estacoes'
TABLE_openwehather_hisotry = 'openweather.history'

CREDENTIALS = service_account.Credentials.from_service_account_file('key/temporal-285820-cde76c259484.json')
pandas_gbq.context.credentials = CREDENTIALS


df_merged = pd.read_csv('data/cleandata/Info pluviometricas/Merged Data/merged.csv', sep=';',
                        dtype={'Local_0': object, 'Local_1': object, 'Local_2': object, 'Local_3': object})
pandas_gbq.to_gbq(df_merged, TABLE_merged, project_id=PROJECT_ID, credentials=CREDENTIALS, if_exists='replace')
print('merged done!')


df_regions = pd.read_csv('data/cleandata/Info pluviometricas/Merged Data/error_regions.csv', sep=';')
pandas_gbq.to_gbq(df_regions, TABLE_error_regions, project_id=PROJECT_ID,
                  credentials=CREDENTIALS, if_exists='replace')
print('error_regions done!')


df_repaired = pd.read_csv('data/cleandata/Info pluviometricas/Merged Data/repaired.csv', sep=';')
pandas_gbq.to_gbq(df_repaired, TABLE_repaired, project_id=PROJECT_ID, credentials=CREDENTIALS, if_exists='replace')
print('repaired done!')


df_lat_lng_estacoes = pd.read_csv('data/cleandata/Estacoes/lat_lng_estacoes.csv', sep=';')
pandas_gbq.to_gbq(df_lat_lng_estacoes, TABLE_lat_lng_estacoes, project_id=PROJECT_ID,
                  credentials=CREDENTIALS, if_exists='replace')
print('lat_lng_estacoes done!')


# df_owm_history_bulk = pd.read_csv('data/cleandata/OpenWeather/history_bulk.csv', sep=';')

# pandas_gbq.to_gbq(df_owm_history_bulk, TABLE_openwehather_hisotry,
#                   project_id=PROJECT_ID, credentials=CREDENTIALS, if_exists='replace')
# print('openweathermap history bulk done!')
