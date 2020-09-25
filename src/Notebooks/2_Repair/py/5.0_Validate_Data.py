#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd 
pd.set_option('display.max_columns', 500)

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode
from datetime import datetime
init_notebook_mode()


# In[ ]:


p1 = '../../../data/cleandata/Info pluviometricas/Merged Data/repaired.csv'
p2 = '../../../data/cleandata/Info pluviometricas/Merged Data/error_regions.csv'
p3 = '../../../data/cleandata/Info pluviometricas/Merged Data/merged.csv'
repaired = pd.read_csv(p1, sep = ';')
regions = pd.read_csv(p2, sep = ';')
merged = pd.read_csv(p3, sep = ';')
repaired['Data_Hora'] = pd.to_datetime(repaired['Data_Hora'], yearfirst = True)
regions['Data_Hora'] = pd.to_datetime(regions['Data_Hora'],   yearfirst = True)
merged['Data_Hora'] = pd.to_datetime(merged['Data_Hora'],   yearfirst = True)


# In[ ]:


print(*[c.split('_')[0] for c in repaired.columns if '0_pred' in c])


# In[ ]:


label = 'PressaoAtmosferica'


# ## Validate Regions

# In[ ]:



# ano = 2015
# df = merged[merged['Data_Hora'].dt.year == ano]
# dfr = regions[regions['Data_Hora'].dt.year == ano]

# fig = make_subplots(5,1, shared_xaxes=True)

# for i in range(5):
#     fig.add_trace(go.Scatter(
#                     x=df['Data_Hora'],
#                     y=df[f'{label}_{i}'],
#                     line = dict(color='#616161'),
#                     ), 
#                   col= 1 ,
#                   row= i + 1)

#     fig.add_trace(go.Scatter(
#                     x=df['Data_Hora'],
#                     y=df[f'{label}_{i}'].where(dfr[f'{label}_{i}_error']),
#                     line = dict(color='red'),
#                     ), 
#                   col= 1 ,
#                   row= i + 1)



# fig.write_html(f'../../../images/{label}_Regions_{str(today)}.html')


# ## Validate repair_regions

# In[ ]:


df = repaired#[repaired['Data_Hora'].dt.year == 2015]
today = datetime.now()

for label in [c.split('_')[0] for c in repaired.columns if '0_error' in c]:
    print(label)

    fig = make_subplots(5,1, shared_xaxes=True)

    for i in range(5):
        fig.add_trace(go.Scatter(
                        x=df['Data_Hora'],
                        y=df[f'{label}_{i}'],
                        line = dict(color='#616161'),
                        legendgroup="value"
                        ), 
                          col= 1 ,
                          row= i + 1)
        try:
            fig.add_trace(go.Scatter(
                            x=df['Data_Hora'],
                            y=df[f'{label}_{i}'].where(df[f'{label}_{i}_interpol']),
                            line = dict(color='yellow'),
                            legendgroup="Interpolation",
                            name="Interpolation"
                            ), 
                              col= 1 ,
                              row= i + 1)
        except:
            pass

        try:
            fig.add_trace(go.Scatter(
                            x=df['Data_Hora'],
                            y=df[f'{label}_{i}'].where(df[f'{label}_{i}_regression']),
                            line = dict(color='green'),
                            legendgroup="Regression",
                            name="Regression"
                            ), 
                              col= 1 ,
                              row= i + 1)
        except:
            pass

        try:
            fig.add_trace(go.Scatter(
                            x=df['Data_Hora'],
                            y=df[f'{label}_{i}'].where(df[f'{label}_{i}_idw']),
                            line = dict(color='blue'),
                            legendgroup="IDW Interpolation",
                            name="IDW Interpolation"
                            ), 
                              col= 1 ,
                              row= i + 1)
        except:
            pass

        try:
            fig.add_trace(go.Scatter(
                            x=df['Data_Hora'],
                            y=df[f'{label}_{i}'].fillna(0).where(df[f'{label}_{i}_fill_ow']),
                            line = dict(color='orange'),
                            legendgroup="OpenWeather",
                            name="OpenWeather"
                            ), 
                              col= 1 ,
                              row= i + 1)
        except:
            pass
        try:
            fig.add_trace(go.Scatter(
                            x=df['Data_Hora'],
                            y=df[f'{label}_{i}'].fillna(0).where(df[f'{label}_{i}_error']),
                            line = dict(color='red'),
                            legendgroup="Error",
                            name="Error"
                            ), 
                              col= 1 ,
                              row= i + 1)
        except:
            pass
    fig.write_html(f'../../../images/{label}_repaired_{str(today)}.html')

