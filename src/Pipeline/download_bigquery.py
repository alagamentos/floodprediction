import pandas_gbq
from google.oauth2 import service_account
import pandas as pd

PROJECT_ID = 'temporal-285820'
TABLE_merged = 'info_pluviometrica.merged'
TABLE_merged_wregions = 'info_pluviometrica.regions'
TABLE_repaired = 'info_pluviometrica.repaired'

CREDENTIALS   = service_account.Credentials.from_service_account_file('key/temporal-285820-cde76c259484.json')

df_merged = pd.read_gbq('SELECT * FROM {PROJECT_ID}.{TABLE_merged}', credentials=CREDENTIALS, project_id=PROJECT_ID)
df_merged.to_csv('data/cleandata/Info pluviometricas/Merged Data/merged.csv',  decimal='.', sep=';', index=False)

df_merged_wRegions = pd.read_gbq('SELECT * FROM {PROJECT_ID}.{TABLE_merged_wregions}', credentials=CREDENTIALS, project_id=PROJECT_ID)
df_merged_wRegions.to_csv('data/cleandata/Info pluviometricas/Merged Data/regions.csv',  decimal='.', sep=';', index=False)

df_merged_wRegions = pd.read_gbq('SELECT * FROM {PROJECT_ID}.{TABLE_repaired}', credentials=CREDENTIALS, project_id=PROJECT_ID, index_col = [0])
df_merged_wRegions.to_csv('data/cleandata/Info pluviometricas/Merged Data/repaired.csv',  decimal='.', sep=';', index=False)
