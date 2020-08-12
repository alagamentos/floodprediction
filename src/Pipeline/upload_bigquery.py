import pandas_gbq
from google.oauth2 import service_account
import pandas as pd

PROJECT_ID = 'temporal-285820'
TABLE_merged = 'info_pluviometrica.merged'
TABLE_merged_wregions = 'info_pluviometrica.regions'
TABLE_repaired = 'info_pluviometrica.repaired'
TABLE_lat_lng_estacoes = 'estacoes.lat_lng_estacoes'

CREDENTIALS   = service_account.Credentials.from_service_account_file('key/temporal-285820-cde76c259484.json')
pandas_gbq.context.credentials = CREDENTIALS

# merged.csv
df_merged = pd.read_csv('data/cleandata/Info pluviometricas/Merged Data/merged.csv',
                        sep=';',
                        dtype={'Local_0': object, 'Local_1': object,
                               'Local_2': object, 'Local_3': object})

pandas_gbq.to_gbq(df_merged, TABLE_merged, project_id=PROJECT_ID, credentials=CREDENTIALS, if_exists='replace')
print('df_merged done!')


# merged_wRegions.csv
df_regions = pd.read_csv('data/cleandata/Info pluviometricas/Merged Data/error_regions.csv',
                                 sep=';')

pandas_gbq.to_gbq(df_regions, TABLE_merged_wregions, project_id=PROJECT_ID, credentials=CREDENTIALS, if_exists='replace')
print('merged_wRegions done!')


# merged_Repaired.csv
df_repaired = pd.read_csv('data/cleandata/Info pluviometricas/Merged Data/repaired.csv',
            sep = ';')

pandas_gbq.to_gbq(df_repaired, TABLE_repaired, project_id=PROJECT_ID, credentials=CREDENTIALS, if_exists='replace')
print('regions done!')

# # lat_lng_estacoes
# df_lat_lng_estacoes = pd.read_csv( 'data/cleandata/Estacoes/lat_lng_estacoes.csv',
#             sep = ';')

# pandas_gbq.to_gbq(df_lat_lng_estacoes, TABLE_lat_lng_estacoes, project_id=PROJECT_ID, credentials=CREDENTIALS, if_exists='replace')
# print('lat_lng_estacoes done!')
