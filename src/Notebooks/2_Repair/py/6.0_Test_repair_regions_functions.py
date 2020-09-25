#!/usr/bin/env python
# coding: utf-8

# In[ ]:


feature = 'RadiacaoSolar'


# In[ ]:



import yaml
def read_yaml(path):
    with open(path) as file:
        return yaml.load(file, Loader=yaml.FullLoader)
    
config_path = '../../../src/Pipeline/config/repair_regions.yaml'
config = read_yaml(config_path)


import plotly.graph_objects as go
from plotly.offline import init_notebook_mode
from plotly.subplots import make_subplots
init_notebook_mode()

import pandas as pd
import sys
sys.path.insert(1, '../../Pipeline')

from repair_regions_functions import *

p1 = '../../../data/cleandata/Info pluviometricas/Merged Data/error_regions.csv'
p2 = '../../../data/cleandata/Info pluviometricas/Merged Data/merged.csv'

regions = pd.read_csv(p1, sep = ';')
merged = pd.read_csv(p2, sep = ';')

df = regions.merge(merged, on = 'Data_Hora')
regions['Data_Hora'] = pd.to_datetime(regions['Data_Hora'])
merged['Data_Hora'] = pd.to_datetime(merged['Data_Hora'])

# Transform Datatime features
df[['Date', 'Time']] = df['Data_Hora'].str.split(expand=True)
df[['Hora', 'Min', 'Seg']] = df['Time'].str.split(':', expand=True)
df[['Ano', 'Mes', 'Dia']] = df['Date'].str.split('-', expand = True)
df['Hora'] = df['Hora'].astype(int); df['Min']  = df['Min'].astype(int)
df['Ano']  = df['Ano'].astype(int) ; df['Mes']  = df['Mes'].astype(int)
df['Dia']  = df['Dia'].astype(int)
df['Data_Hora'] = pd.to_datetime(df['Data_Hora'])

df = df.drop(columns = ['index'])


# In[ ]:


p3 = '../../../data/cleandata/Info pluviometricas/Merged Data/repaired.csv'
df = pd.read_csv(p3, sep = ';')
df['Data_Hora'] = pd.to_datetime(df['Data_Hora'])


# In[ ]:


ano = 2015
df = df#[df['Data_Hora'].dt.year == ano]

for c in [c for c in df.columns if '_repaired' in c]:
    df.rename(columns = {c:c.replace('_repaired','')})


# In[ ]:


if 'interpolation' in config[feature]:
    kwargs_interpol = config[feature]['interpolation']
    print(f'Applying interpolation on {feature} with {config[feature]["interpolation"]}')
    df = interpolation(df, feature, kwargs_interpol)
if 'idw' in config[feature]:
    kwargs_idw = config[feature]['idw']
    print(f'Applying idw on {feature} with {kwargs_idw}')
    df = idw(df, feature, kwargs_idw)
if 'regression' in config[feature]:
    kwargs_regression = config[feature]['regression']
    print(f'Applying regression on {feature} with {kwargs_regression}')
    df = regression(df, feature, kwargs_regression)
if 'fill_ow' in config[feature]:
    kwargs_fill_ow = config[feature]['fill_ow']
    print(f'Applying fill_ow on {feature} with {kwargs_fill_ow}')
    df = fill_ow(df, feature, kwargs_fill_ow)


# In[ ]:


#df = df[df['Data_Hora'].dt.year == ano]

fig = make_subplots(5,1, shared_xaxes=True)

for i in range(5):
    fig.add_trace(go.Scatter(
                    x=df['Data_Hora'],
                    y=df[f'{feature}_{i}'],
                    line = dict(color='#616161'),
                    ), 
                  col= 1 ,
                  row= i + 1)

    try:
        fig.add_trace(go.Scatter(
                        x=df['Data_Hora'],
                        y=df[f'{feature}_{i}'].where(df[f'{feature}_{i}_interpol']),
                        line = dict(color='yellow')
                        ), 
                      col= 1 ,
                      row= i + 1)
    except:
        pass
    
    try:
        fig.add_trace(go.Scatter(
                        x=df['Data_Hora'],
                        y=df[f'{feature}_{i}'].where(df[f'{feature}_{i}_idw']),
                        line = dict(color='blue'),
                        ), 
                      col= 1 ,
                      row= i + 1)
    except:
        pass
    
    try:
        fig.add_trace(go.Scatter(
                        x=df['Data_Hora'],
                        y=df[f'{feature}_{i}'].where(df[f'{feature}_{i}_regression']),
                        line = dict(color='green'),
                        ), 
                      col= 1 ,
                      row= i + 1)
    except:
        pass
    
    try:
        fig.add_trace(go.Scatter(
                        x=df['Data_Hora'],
                        y=df[f'{feature}_{i}'].where(df[f'{feature}_{i}_fill_ow']),
                        line = dict(color='orange'),
                        ), 
                      col= 1 ,
                      row= i + 1)
    except:
        pass
    
    try:
        fig.add_trace(go.Scatter(
                        x=df['Data_Hora'],
                        y=df[f'{feature}_{i}'].where(df[f'{feature}_{i}_error']),
                        line = dict(color='red'),
                        ), 
                      col= 1 ,
                      row= i + 1)
    except:
        pass
    
fig.show()


# In[ ]:




