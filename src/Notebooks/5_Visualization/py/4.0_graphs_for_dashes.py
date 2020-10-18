#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system("ls '../../../data/cleandata/Estacoes/lat_lng_estacoes.csv'")


# In[ ]:


import pandas as pd

est = pd.read_csv('../../../data/cleandata/Estacoes/lat_lng_estacoes.csv', sep = ';')


# In[ ]:


label = pd.read_csv('../../../data/cleandata/Ordens de serviço/Enchentes_LatLong.csv', sep = ';')
# label = label[['Data','lat','lng']]
# label['latlng'] = label['lat'].astype(str)  + ',' +   label['lng'].astype(str)


# In[ ]:


import plotly.graph_objects as go


# In[ ]:


label['Data'] = pd.to_datetime(label['Data'], yearfirst = True)
gb = label.groupby('Data').count()
gb


# In[ ]:


import plotly.express as px

gb = label.groupby('Data').count()

fig = go.Figure()
fig.add_trace(go.Bar(
                 x = gb.index,
                 y = gb['lat'],
                width = [1000 * 3600 * 24 * 0.9] * len(gb.index)
                    ),
             )

fig.update_layout(bargap = 0)
fig.update_traces(marker_color='black', marker_line_color='#3b3b3b',
                  marker_line_width=1, opacity=1)
fig.show()


# In[ ]:


token = 'pk.eyJ1IjoiZmlwcG9saXRvIiwiYSI6ImNqeXE4eGp5bjFudmozY3A3M2RwbzYxeHoifQ.OdNEEm5MYvc2AS4iO_X3Pw'


# In[ ]:


import plotly.express as px
px.set_mapbox_access_token(token)
df = px.data.carshare()
fig = px.scatter_mapbox(df, lat="centroid_lat", lon="centroid_lon",     color="peak_hour", size="car_hours",
                  color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=10)
fig.show()


# In[ ]:


fig = go.Figure()


fig.add_trace(go.Scattermapbox(
    lat=est['lat'],
    lon=est['lng'],
    mode='markers',
    marker=go.scattermapbox.Marker(
        size=14,
        color = 'black',
        symbol = 'circle'
    ),
   text=['Santo Amaro'],
            ))

# fig.add_trace(go.Densitymapbox(
#                     lat=label.lat,
#                     lon=label.lng,
#                     z=[1] * label.shape[0],
#                     radius=10,
#                     colorscale = 'Blues',
#                     opacity = 0.75,
#     showscale=False
#                 ))

fig.update_layout(
    mapbox_style="open-street-map",
    hovermode='closest',
    mapbox=dict(
        bearing=0,
        center=go.layout.mapbox.Center(
            lat=est.iloc[-1,1],
            lon=est.iloc[-1,2],
            style = 'light'
        ),
#         style='dark',
        pitch=0,
        zoom=12
    ),
    width = 750,
    height = 500,
    showlegend = False,
                )


fig.show()


# In[ ]:


fig = go.Figure(go.Scattermapbox(
        lat=['45.5017'],
        lon=['-73.5673'],
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=14
        ),
        text=['Montreal'],
    ))
fig.show()

