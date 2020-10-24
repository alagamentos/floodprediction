import plotly.graph_objects as go
from plotly.subplots import make_subplots

from cptec import get_prediction, get_polygon

from urllib.request import urlopen
import json
from shapely.geometry import Polygon

token = 'pk.eyJ1IjoiZmlwcG9saXRvIiwiYSI6ImNqeXE4eGp5bjFudmozY3A3M2RwbzYxeHoifQ.OdNEEm5MYvc2AS4iO_X3Pw'


path = 'https://raw.githubusercontent.com/tbrugz/geodata-br/master/geojson/geojs-35-mun.json'

with urlopen(path) as response:
    counties = json.load(response)
SA = [ i for i in counties['features'] if i['properties']['name'] == 'Santo André' ][0]
SA_polygon = Polygon(SA['geometry']['coordinates'][0])

value_dict = {
              'Sem Alerta': 0,
              'Aviso de Observação':1,
              'Aviso de Atenção': 2,
              'Aviso Especial': 3,
              'Aviso Extraordinário de Risco Iminente':4,
              'Aviso Cessado': 5
              }
inv_value_dict = {
              0:'Sem Alerta',
              1:'Aviso de Observação',
              2:'Aviso de Atenção',
              3:'Aviso Especial',
              4:'Aviso Extraordinário de Risco Iminente',
              5:'Aviso Cessado',
              }


SA_layer = dict(sourcetype = 'geojson',
             source = SA ,
             below='',
             type = 'fill',
             opacity=0.25,
             color = 'white')

x_pred, y_pred = {}, {}

x_pred['bam'], y_pred['bam'] = get_prediction('bam')
x_pred['wrf'], y_pred['wrf'] = get_prediction('wrf')

polygon_dict = get_polygon()

color_dict = {'Aviso de Observação':'#FAFF06',
              'Aviso de Atenção': '#F2C004',
              'Aviso Especial': '#E92C00',
              'Aviso Extraordinário de Risco Iminente': '#000000',
              'Aviso Cessado': '#C3C3C3'
              }

def get_geojson_polygon(lons, lats, color='blue'):
    if len(lons) != len(lats):
        raise ValueError('the legth of longitude list  must coincide with that of latitude')
    geojd = {"type": "FeatureCollection"}
    geojd['features'] = []
    coords = []
    for lon, lat in zip(lons, lats):
        coords.append((lon, lat))
    coords.append((lons[0], lats[0]))  #close the polygon
    geojd['features'].append({ "type": "Feature",
                               "geometry": {"type": "Polygon",
                                            "coordinates": [coords] }})
    layer=dict(sourcetype = 'geojson',
             source =geojd,
             below='',
             type = 'fill',
             opacity=0.25,
             color = color)
    return layer

def make_data_repair_plots(merged, error, repaired, col, est, year, month):
  year, month = int(year), int(month)
  repaired_plot = repaired.loc[ (repaired['Data_Hora'].dt.year == year) &
                                (repaired['Data_Hora'].dt.month == month),
                                 ['Data_Hora', f'{col}_{est}'] ]

  merged_plot = merged.loc[ (merged['Data_Hora'].dt.year == year) &
                            (merged['Data_Hora'].dt.month == month),
                            ['Data_Hora', f'{col}_{est}'] ]

  error_plot = error.loc[(error['Data_Hora'].dt.year == year ) &
                         (error['Data_Hora'].dt.month == month),
                         ['Data_Hora', f'{col}_{est}_error'] ]


  plots = make_subplots(2,1, shared_xaxes=True)
  plots.add_trace(go.Scatter(
              x = merged_plot['Data_Hora'],
              y = merged_plot[f'{col}_{est}'],
              ), col = 1, row = 1)
  plots.add_trace(go.Scatter(
              x = merged_plot['Data_Hora'].where(error_plot[f'{col}_{est}_error']),
              y = merged_plot[f'{col}_{est}'].fillna(0).where(error_plot[f'{col}_{est}_error']),
              line = dict(color = 'red')
              ), col = 1, row = 1)
  plots.add_trace(go.Scatter(
              x = repaired_plot['Data_Hora'],
              y = repaired_plot[f'{col}_{est}'],
              ), col = 1, row = 2)
  plots.update_layout(transition_duration=500)

  return plots

def make_mapa_plot(label_copy, est):

  mapa = go.Figure()

  mapa.add_trace(go.Scattermapbox(
      lat=est['lat'],
      lon=est['lng'],
      mode='markers',
      marker=go.scattermapbox.Marker(
          size=14,
          color = 'green',
          symbol = 'marker'
      ),
    text=['Santo Amaro'],
              ))

  mapa.add_trace(go.Densitymapbox(
                      lat=label_copy['lat'],
                      lon=label_copy['lng'],
                      z=[1] * label_copy.shape[0],
                      radius=5,
                      colorscale = 'Blues',
                      reversescale=True,
                      opacity = 0.75,
      showscale=False
                  ))

  mapa.update_layout(
      hovermode='closest',
      mapbox=dict(
          accesstoken=token,
          bearing=0,
          center=go.layout.mapbox.Center(
            lat=-23.652598,
            lon=-46.527872,
        ),
        style='dark',
        pitch=0,
        zoom=11
      ),
      width = 500,
      height = 550,
      showlegend = False,
                  )

  return mapa

def make_rain_ordem_servico_plot(gb_label_plot, rain_sum_plot):

  ordem_servico_figure = make_subplots(2,1, shared_xaxes=True,
                                       vertical_spacing = 0.1,
                                       subplot_titles=('Ordens de Serviço',
                                                       'Precipitação'))
  ordem_servico_figure.add_trace(go.Bar(
                                    x = gb_label_plot['Data'] ,
                                    y = gb_label_plot['count']),
                                  row = 1, col = 1
                                )
  ordem_servico_figure.add_trace(go.Bar(
                    x = rain_sum_plot['Data'],
                    y = rain_sum_plot['Precipitacao_2'] ,),
              row = 2, col = 1,
             )
  ordem_servico_figure.update_layout(bargap = 0)
  ordem_servico_figure.update_traces(marker_color='black',
                                     marker_line_color='#3b3b3b',
                                     marker_line_width=1,
                                     opacity=1)

  return ordem_servico_figure

def make_cptec_prediction(model):

  x, y = x_pred[model], y_pred[model]

  # Cptec Prediction -----------------------------------

  cptec_fig = make_subplots(2,2, shared_xaxes = True)

  # Precipitação
  cptec_fig.add_trace(go.Scatter(
                        x = x['precipitacao_acc'],
                        y = y['precipitacao_acc'],
                                ),
                                row = 1, col = 1
                    )
  cptec_fig.add_trace(go.Bar(
                        x = x['precipitacao'],
                        y = y['precipitacao'],
                                ),
                                row = 1, col = 1
                    )

  # Temperatura
  cptec_fig.add_trace(go.Scatter(
                        x = x['temperatura'],
                        y = y['temperatura'],
                                ),
                                row = 1, col = 2
                    )
  cptec_fig.add_trace(go.Scatter(
                        x = x['temperatura_aparente'],
                        y = y['temperatura_aparente'],
                                ),
                                row = 1, col = 2
                    )

  # Umidade Relativa
  cptec_fig.add_trace(go.Scatter(
                        x = x['umidade_relativa'],
                        y = y['umidade_relativa'],
                                ),
                                row = 2, col = 1
                    )

  # Pressao
  cptec_fig.add_trace(go.Scatter(
                        x = x['pressao'],
                        y = y['pressao'],
                                ),
                                row = 2, col = 2
                    )

  return cptec_fig

def make_cptec_polygon(time):

  geom = polygon_dict[time]['geom']
  title = polygon_dict[time]['title']

  mylayers = []

  value = 0

  for t, g in zip(title,geom):

    warning_poly = Polygon(geom[0])
    if warning_poly.intersects(SA_polygon):
      if value_dict[t] > value:
        value = value_dict[t]

    lat = [p[1] for p in g]
    lon = [p[0] for p in g]

    try:
      warning_color = color_dict[t]
    except KeyError:
      warning_color = 'blue'

    mylayers.append(get_geojson_polygon(lon, lat, warning_color))

  mylayers.append(SA_layer)

  fig = go.Figure()
  fig.add_trace(go.Scattermapbox(
                lat=[-23.7052598],
                lon=[-46.4497872],
              mode='markers',
              marker=go.scattermapbox.Marker(
                  size=1
                      ),
                  ))

  fig.update_layout(
      hovermode='closest',
      mapbox=dict(
          accesstoken=token,
          bearing=0,
          center=go.layout.mapbox.Center(
              lat=-23.7052598,
              lon=-46.4497872,
          ),
          style='dark',
          pitch=0,
          zoom=9
      ),
      width = 750,
      height = 750,
      showlegend = False,
                  )

  fig.layout.update(mapbox_layers=mylayers)

  return fig, inv_value_dict[value]

