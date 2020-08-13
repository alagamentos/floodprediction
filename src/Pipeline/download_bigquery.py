import pandas_gbq
from google.oauth2 import service_account
import pandas as pd
import os

# Windows
import google.cloud.bigquery_storage_v1.client
from functools import partialmethod
# Set a two hours timeout
google.cloud.bigquery_storage_v1.client.BigQueryReadClient.read_rows = \
    partialmethod(google.cloud.bigquery_storage_v1.client.BigQueryReadClient.read_rows, timeout=3600*2)

PROJECT_ID = 'temporal-285820'
TABLE_merged = 'info_pluviometrica.merged'
TABLE_regions = 'info_pluviometrica.error_regions'
TABLE_repaired = 'info_pluviometrica.repaired'
TABLE_lat_lng_estacoes = 'estacoes.lat_lng_estacoes'

CREDENTIALS = service_account.Credentials.from_service_account_file('key/temporal-285820-cde76c259484.json')

df_merged = pd.read_gbq(f'SELECT * FROM {PROJECT_ID}.{TABLE_merged}', credentials=CREDENTIALS, project_id=PROJECT_ID)
df_merged.to_csv('data/cleandata/Info pluviometricas/Merged Data/merged.csv',  decimal='.', sep=';', index=False)

df_regions = pd.read_gbq(f'SELECT * FROM {PROJECT_ID}.{TABLE_regions}', credentials=CREDENTIALS, project_id=PROJECT_ID)
df_regions.to_csv('data/cleandata/Info pluviometricas/Merged Data/error_regions.csv',  decimal='.', sep=';', index=False)

df_repaired = pd.read_gbq(f'SELECT * FROM {PROJECT_ID}.{TABLE_repaired}', credentials=CREDENTIALS, project_id=PROJECT_ID)
df_repaired.to_csv('data/cleandata/Info pluviometricas/Merged Data/repaired.csv',  decimal='.', sep=';', index=False)

lat_lng_estacoes_path = 'data/cleandata/Estacoes/lat_lng_estacoes.csv'

if not os.path.exists(lat_lng_estacoes_path):
  os.mkdir('data/cleandata/Estacoes')
  df_lng_lng_estacoes = pd.read_gbq(f'SELECT * FROM {PROJECT_ID}.{TABLE_lat_lng_estacoes}', credentials=CREDENTIALS, project_id=PROJECT_ID)
  df_lng_lng_estacoes.to_csv(lat_lng_estacoes_path,  decimal='.', sep=';', index=False)
