import plotly.graph_objects as go
from plotly.subplots import make_subplots
from os import path
import xgboost
from cptec import get_prediction, get_polygon
from urllib.request import urlopen
import json
from shapely.geometry import Polygon
import pandas as pd

# Color palette
MIDNIGHT_BLUE = '#222437'
TURQUOISE = '#1ad5c3'
PURPLE = '#6201ed'
RED = '#fd3f6e'
YELLOW = '#eab009'
BLUE = '#2c50ed'
SLATEBLUE = '#6646ed'
LIGHT_PINK = '#ecbdca'

BG_DARK = MIDNIGHT_BLUE
PLOT_PRI = TURQUOISE
PLOT_SEC = PURPLE
PLOT_TER = RED
PLOT_QUA = YELLOW
PLOT_QUI = BLUE

subplots_vertical_spacing = 0.22

dict_months = {
     1: u'Janeiro',
     2: u'Fevereiro',
     3: u'Março',
     4: u'Abril',
     5: u'Maio',
     6: u'Junho',
     7: u'Julho',
     8: u'Agosto',
     9: u'Setembro',
     10: u'Outubro',
     11: u'Novembro',
     12: u'Dezembro'
}
plot_layout_kwargs = dict(template='plotly_dark',
                          paper_bgcolor=BG_DARK,
                          plot_bgcolor=BG_DARK)

token = 'pk.eyJ1IjoiZmlwcG9saXRvIiwiYSI6ImNqeXE4eGp5bjFudmozY3A3M2RwbzYxeHoifQ.OdNEEm5MYvc2AS4iO_X3Pw'

xgb_path = 'model/Identificacao_0H.json'

x_pred, y_pred = {}, {}

x_pred['bam'], y_pred['bam'] = get_prediction('bam')
x_pred['wrf'], y_pred['wrf'] = get_prediction('wrf')

polygon_dict, SA_polygon, SA_layer = get_polygon()

color_dict = {
  'Aviso de Observação': LIGHT_PINK,
  'Aviso de Atenção': YELLOW,
  'Aviso Especial': RED,
  'Aviso Extraordinário de Risco Iminente': BLUE,
  'Aviso Cessado': '#C3C3C3'
}

# XGBoost
def get_xgb_predictions(model):
  x_data_t = x_pred[model]
  y_data = y_pred[model]

  df = pd.DataFrame(columns=['Mes', 'Dia', 'Local', 'Precipitacao', 'Data_Hora'])

  df['Data_Hora'] = x_data_t['precipitacao']
  df['Data_Hora'] = pd.to_datetime(df['Data_Hora'], yearfirst=True)
  df['Precipitacao'] = y_data['precipitacao']

  df['Dia'] = df['Data_Hora'].dt.day
  df['Mes'] = df['Data_Hora'].dt.month
  df['Local'] = 1

  df = df.drop(columns='Data_Hora')

  sum_precipitacao = df.groupby(['Dia', 'Mes']).sum().reset_index().drop(columns=['Local'])
  sum_precipitacao = sum_precipitacao.rename(columns={'Precipitacao': 'PrecSum'})

  X = df.merge(sum_precipitacao, on=['Dia', 'Mes'], how='inner')

  return X


def predict(model, xgb_path):
  X = get_xgb_predictions(model)

  model = xgboost.Booster()
  model.load_model(xgb_path)

  data = xgboost.DMatrix(data=X)
  return model.predict(data)


y_xgb = {}
y_xgb['bam'] = predict('bam', xgb_path)
y_xgb['wrf'] = predict('wrf', xgb_path)


def get_geojson_polygon(lons, lats, color='blue'):
    if len(lons) != len(lats):
        raise ValueError('the legth of longitude list  must coincide with that of latitude')
    geojd = {"type": "FeatureCollection"}
    geojd['features'] = []
    coords = []
    for lon, lat in zip(lons, lats):
        coords.append((lon, lat))
    coords.append((lons[0], lats[0]))  # close the polygon
    geojd['features'].append({"type": "Feature",
                              "geometry": {"type": "Polygon",
                                           "coordinates": [coords]}})
    layer = dict(sourcetype='geojson',
                 source=geojd,
                 below='',
                 type='fill',
                 opacity=0.25,
                 color=color)
    return layer

# Graphs
def make_data_repair_plots(merged, error, repaired, col, est, year, month):
  year, month = int(year), int(month)
  repaired_plot = repaired.loc[(repaired['Data_Hora'].dt.year == year) &
                               (repaired['Data_Hora'].dt.month == month),
                               ['Data_Hora', f'{col}_{est}']]

  merged_plot = merged.loc[(merged['Data_Hora'].dt.year == year) &
                           (merged['Data_Hora'].dt.month == month),
                           ['Data_Hora', f'{col}_{est}']]

  error_plot = error.loc[(error['Data_Hora'].dt.year == year) &
                         (error['Data_Hora'].dt.month == month),
                         ['Data_Hora', f'{col}_{est}_error']]

  plots = make_subplots(2, 1, shared_xaxes=True,
                        subplot_titles=('Dados Originais',
                                        'Dados Corrigidos'))
  plots.add_trace(go.Scatter(
      x=merged_plot['Data_Hora'],
      y=merged_plot[f'{col}_{est}'],
      line=dict(color=PLOT_SEC)
  ), col=1, row=1)
  plots.add_trace(go.Scatter(
      x=merged_plot['Data_Hora'].where(error_plot[f'{col}_{est}_error']),
      y=merged_plot[f'{col}_{est}'].fillna(0).where(error_plot[f'{col}_{est}_error']),
      line=dict(color=PLOT_TER)
  ), col=1, row=1)
  plots.add_trace(go.Scatter(
      x=repaired_plot['Data_Hora'],
      y=repaired_plot[f'{col}_{est}'],
      line=dict(color=PLOT_PRI)
  ), col=1, row=2)
  plots.update_layout(showlegend=False,
                      transition_duration=500,
                      margin=dict(l=50, r=30, t=40, b=30),
                      **plot_layout_kwargs)

  try:
    ymax, ymin = max(repaired_plot[f'{col}_{est}']), min(repaired_plot[f'{col}_{est}'])

    if col == 'PressaoAtmosferica':
      plots.update_yaxes(range=[ymin, ymax], col=1, row=1)
  except ValueError:
    pass

  return plots


def make_mapa_plot(label_copy, est):
  mapa = go.Figure()

  mapa.add_trace(go.Scattermapbox(
      lat=est['lat'],
      lon=est['lng'],
      mode='markers',
      marker=go.scattermapbox.Marker(
          size=14,
          color='#404042',
          symbol='marker'
      ),
      text=est['Estacao'],
  ))

  mapa.add_trace(go.Densitymapbox(
      lat=label_copy['lat'],
      lon=label_copy['lng'],
      z=[1] * label_copy.shape[0],
      radius=5,
      colorscale='Tealgrn',
      reversescale=False,
      opacity=0.75,
      showscale=False
  ))

  mapa.update_layout(
      hovermode='closest',
      mapbox=dict(
          accesstoken=token,
          bearing=0,
          center=go.layout.mapbox.Center(
              lat=-23.665688,
              lon=-46.517582,
          ),
          style='light',
          pitch=0,
          zoom=11.5
      ),
      width=500,
      height=550,
      showlegend=False,
      margin=dict(l=0, r=0, t=0, b=0),
      **plot_layout_kwargs
  )

  return mapa


def make_rain_ordem_servico_plot(gb_label_plot, rain_sum_plot):
  ordem_servico_figure = make_subplots(2, 1, shared_xaxes=True,
                                       vertical_spacing= subplots_vertical_spacing,
                                       subplot_titles=('Ordens de Serviço por Dia',
                                                       'Precipitação por Dia'))
  ordem_servico_figure.add_trace(go.Bar(
      x=gb_label_plot['Data'],
      y=gb_label_plot['count']),
      row=1, col=1
  )
  ordem_servico_figure.add_trace(go.Bar(
      x=rain_sum_plot['Data'],
      y=rain_sum_plot['Precipitacao_2'],),
      row=2, col=1,
  )
  ordem_servico_figure.update_layout(showlegend=False,
                                     bargap=0,
                                     margin=dict(l=40, r=20, t=40, b=30),
                                     **plot_layout_kwargs)

  ordem_servico_figure.update_traces(marker_color=PLOT_PRI,
                                     marker_line_color=PLOT_PRI,
                                     marker_line_width=1,
                                     opacity=1,
                                     col=1, row=1)
  ordem_servico_figure.update_traces(marker_color=PLOT_SEC,
                                     marker_line_color=PLOT_SEC,
                                     marker_line_width=1,
                                     opacity=1,
                                     col=1, row=2)

  return ordem_servico_figure


def make_rain_ordem_servico_plot_grouped_by(gb_label_plot_, rain_sum_plot_):

  gb_label_plot =  gb_label_plot_.copy()
  rain_sum_plot =  rain_sum_plot_.copy()

  gb_label_plot['Mes'] = gb_label_plot['Data'].dt.month
  rain_sum_plot['Mes'] = rain_sum_plot['Data'].dt.month

  gb_label_plot['Ano'] = gb_label_plot['Data'].dt.year
  rain_sum_plot['Ano'] = rain_sum_plot['Data'].dt.year

  gb_label_plot_gb = gb_label_plot.groupby(['Mes','Ano']).sum().reset_index()
  rain_sum_plot_gb = rain_sum_plot.groupby(['Mes','Ano']).sum().reset_index()

  df_label = gb_label_plot_gb.groupby('Mes').mean().drop(columns = ['Ano']).reset_index()
  df_rain = rain_sum_plot_gb.groupby('Mes').mean().drop(columns = ['Ano']).reset_index()

  df_label = df_label.sort_values(by = 'Mes', ascending = True)
  df_rain = df_rain.sort_values(by = 'Mes', ascending = True)

  df_label['Mes'] = df_label['Mes'].map(dict_months)
  df_rain['Mes'] = df_rain['Mes'].map(dict_months)


  ordem_servico_gb_figure = make_subplots(2, 1, shared_xaxes=True,
                                          vertical_spacing=subplots_vertical_spacing,
                                          subplot_titles=('Média Mensal das Ordens de Serviço',
                                                          'Média Mensal da Precipitação'))

  ordem_servico_gb_figure.add_trace(go.Bar(
      x=df_rain['Mes'],
      y=df_rain['Precipitacao_2'],),
      row=2, col=1,
  )

  ordem_servico_gb_figure.add_trace(go.Bar(
      x=df_label['Mes'],
      y=df_label['count']),
      row=1, col=1
  )

  ordem_servico_gb_figure.update_layout(showlegend=False,
                                     bargap=0.2,
                                     margin=dict(l=40, r=20, t=40, b=30),
                                     **plot_layout_kwargs)

  ordem_servico_gb_figure.update_traces(marker_color=PLOT_PRI,
                                     marker_line_color=PLOT_PRI,
                                     marker_line_width=1,
                                     opacity=1,
                                     col=1, row=1)
  ordem_servico_gb_figure.update_traces(marker_color=PLOT_SEC,
                                     marker_line_color=PLOT_SEC,
                                     marker_line_width=1,
                                     opacity=1,
                                     col=1, row=2)

  return ordem_servico_gb_figure


def make_cptec_prediction(model):
  x, y = x_pred[model], y_pred[model]

  # Cptec Prediction -----------------------------------
  subplot_titles = ("Precipitação",
                    "Temperatura",
                    "Umidade Relativa",
                    "Pressão Atmosférica")

  cptec_fig = make_subplots(2, 2, shared_xaxes=True,
                            vertical_spacing = subplots_vertical_spacing,
                            subplot_titles=subplot_titles)

  # Precipitação
  # cptec_fig.add_trace(go.Scatter(
  #                       x = x['precipitacao_acc'],
  #                       y = y['precipitacao_acc'],
  #                       name = 'Precipitação',
  #                       line = dict(color = PREP_ACC)
  #                               ),
  #                               row = 1, col = 1
  #                   )
  cptec_fig.add_trace(go.Bar(
      x=x['precipitacao'],
      y=y['precipitacao'],
      name='Precipitação Acumulada',
      marker=dict(color=PLOT_PRI)
  ),
      row=1, col=1
  )

  # Temperatura
  cptec_fig.add_trace(go.Scatter(
      x=x['temperatura'],
      y=y['temperatura'],
      name='Temperatura',
      line=dict(color=PLOT_TER)
  ),
      row=1, col=2
  )
  cptec_fig.add_trace(go.Scatter(
      x=x['temperatura_aparente'],
      y=y['temperatura_aparente'],
      name='Sensação térmica',
      line=dict(color=PLOT_SEC)
  ),
      row=1, col=2
  )

  # Umidade Relativa
  cptec_fig.add_trace(go.Scatter(
      x=x['umidade_relativa'],
      y=y['umidade_relativa'],
      name='Umidade Relativa',
      line=dict(color=PLOT_QUI)
  ),
      row=2, col=1
  )

  # Pressao
  cptec_fig.add_trace(go.Scatter(
      x=x['pressao'],
      y=y['pressao'],
      name='Pressão Atmosférica',
      line=dict(color=PLOT_QUA)
  ),
      row=2, col=2
  )

  cptec_fig.update_layout(legend=dict(
      orientation="h",
      yanchor="bottom",
      y=-0.3,
      xanchor="center",
      x=0.5
  ),
      margin=dict(l=40, r=30, t=40, b=30),
      **plot_layout_kwargs)

  return cptec_fig


def make_cptec_polygon(time):
  geom = polygon_dict[time]['geom']
  title = polygon_dict[time]['title']
  aviso = polygon_dict[time]['aviso']

  mylayers = []

  for t, g in zip(title, geom):
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
          style='light',
          pitch=0,
          zoom=9
      ),
      margin=dict(l=0, r=0, t=0, b=0),
      width=750,
      height=750,
      showlegend=False,
      **plot_layout_kwargs
  )

  fig.layout.update(mapbox_layers=mylayers)

  return fig, aviso


def make_prob_graph(model):
  y = y_xgb[model]
  m = max(y_pred[model]['precipitacao'])
  y_max = m if 5*m > 50 else 5*m

  fig = make_subplots(specs=[[{"secondary_y": True}]])

  fig.add_trace(go.Bar(
      y=y * 100,
      x=x_pred[model]['precipitacao'],
      name='Chance de alagamento',
      marker=dict(
          color=y,
          cmin=0,
          cmax=1,
          colorscale=[
              [0, PLOT_PRI],
              [1, PLOT_TER]]
      ),
  ), secondary_y=False,
  )

  fig.add_trace(go.Scatter(
      y=y_pred[model]['precipitacao'],
      x=x_pred[model]['precipitacao'],
      name='Chuva [mm]',
      line=dict(color=PLOT_QUI, width=3),
  ), secondary_y=True,
  )
  fig.update_yaxes(range=[0, 100],
                   title='Probabilidade de alagamento [%]',
                   secondary_y=False,
                   titlefont=dict(color=PLOT_PRI),
                   tickfont=dict(color=PLOT_PRI),)
  fig.update_yaxes(range=[0, y_max],
                   title='Precipitacao [mm]',
                   secondary_y=True,
                   titlefont=dict(color=PLOT_QUI),
                   tickfont=dict(color=PLOT_QUI),)
  fig.update_layout(showlegend=False,
                    title_x=0.5,
                    title_text="Previsão de Alagamento",
                    margin=dict(l=90, r=0, t=70, b=50),
                    **plot_layout_kwargs)

  return fig
