
#%%
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import os

from cptec import get_prediction, get_polygon

from graphs import make_data_repair_plots, make_mapa_plot, \
    make_rain_ordem_servico_plot, make_cptec_prediction, \
    make_cptec_polygon, make_prob_graph

root = Path(os.path.dirname(os.path.realpath(__file__))).parent.parent


# Prepdata ----------------------------------
merged_path = root / 'data/cleandata/Info pluviometricas/Merged Data/merged.csv'
repaired_path = root / 'data/cleandata/Info pluviometricas/Merged Data/repaired.csv'
error_path = root / 'data/cleandata/Info pluviometricas/Merged Data/error_regions.csv'
estacoes_path = root / 'data/cleandata/Estacoes/lat_lng_estacoes.csv'
oservico_path = root / 'data/cleandata/Ordens de serviço/Enchentes_LatLong.csv'

merged = pd.read_csv(merged_path, sep=';')
repaired = pd.read_csv(repaired_path, sep=';')
error = pd.read_csv(error_path, sep=';')
est = pd.read_csv(estacoes_path, sep=';').iloc[:-1, :]
label = pd.read_csv(oservico_path, sep=';')

# Label
merged['Data_Hora'] = pd.to_datetime(merged['Data_Hora'], yearfirst=True)
repaired['Data_Hora'] = pd.to_datetime(repaired['Data_Hora'], yearfirst=True)
error['Data_Hora'] = pd.to_datetime(error['Data_Hora'], yearfirst=True)
label['Data'] = pd.to_datetime(label['Data'], yearfirst=True)
gb_label = label.groupby('Data').count().reset_index()[['Data', 'lat']]
gb_label.columns = ['Data', 'count']

# Precipitacao Hora em Hora
precipitacao = repaired[repaired['Data_Hora'].dt.minute == 0].copy()
precipitacao['Data_Hora'] = pd.to_datetime(repaired['Data_Hora'], yearfirst=True)
precipitacao['Data'] = pd.to_datetime(precipitacao['Data_Hora'].dt.date, yearfirst=True)
rain_sum = precipitacao.groupby('Data').sum().reset_index()[['Data', 'Precipitacao_2']]

list_of_years = list(range(2011, 2020, 1))
year_options = [{'label': str(y), 'value': y} for y in list_of_years]
year_options_slider = {y: str(y) for y in list_of_years}

#%%
dict_months = {
    u'Janeiro': 1,
    u'Fevereiro': 2,
    u'Março': 3,
    u'Abril': 4,
    u'Maio': 5,
    u'Junho': 6,
    u'Julho': 7,
    u'Agosto': 8,
    u'Setembro': 9,
    u'Outubro': 10,
    u'Novembro': 11,
    u'Dezembro': 12
}
months_options = [{'label': k, 'value': int(v)} for k, v in dict_months.items()]

# Startup app -------------------------------------
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(
    __name__,
    external_stylesheets=external_stylesheets,
    url_base_pathname='/',
    assets_external_path='/assets/',
    assets_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets/')
)


# Callbacks ----------------------------------------
@app.callback(
    [Output('mes', 'options'), Output('mes', 'value')],
    [Input('ano', 'value')],
    state=[State('mes', 'value')]
)
def update_mont_options(ano, mes):
  if ano == 2019:
    year_options = [{'label': k, 'value': int(v)} for k, v in dict_months.items() if int(v) < 10]
    if int(mes) >= 10:
      mes = str(9)
    return year_options, mes
  else:
    return [{'label': k, 'value': int(v)} for k, v in dict_months.items()], mes


@app.callback(
    Output('plots', 'figure'),
    [Input('update-button', 'n_clicks')],
    state=[
        State('metrica', 'value'),
        State('estacao', 'value'),
        State('ano', 'value'),
        State('mes', 'value'),
    ]
)
def update_graphs(n_clicks, col, est, year, month):
  return make_data_repair_plots(merged, error, repaired, col, est, year, month)


@app.callback(
    [Output('count', 'children'), Output('mapa', 'figure'), Output('os-subplots', 'figure')],
    [Input('year-slider', 'value')],
)
def update_map(date_range):

  # Ordens de serviço
  label_plot = label.loc[(label['Data'].dt.year >= date_range[0]) &
                         (label['Data'].dt.year <= date_range[1]), :]

  # Grouped by ordens de serviço
  gb_label_plot = gb_label.loc[(gb_label['Data'].dt.year >= date_range[0]) &
                               (gb_label['Data'].dt.year <= date_range[1]), :]

  # Grouped by rain
  rain_sum_plot = rain_sum.loc[(rain_sum['Data'].dt.year >= date_range[0]) &
                               (rain_sum['Data'].dt.year <= date_range[1]), :]

  mapa = make_mapa_plot(label_plot, est)
  ordem_servico_figure = make_rain_ordem_servico_plot(gb_label_plot, rain_sum_plot)
  count = label_plot.shape[0]

  return f'Total de ordens de serviço: {count}', mapa, ordem_servico_figure


@app.callback(
    [Output('cptec-poly', 'figure'), Output('warning', 'children')],
    [Input('radio-poly', 'value')],
)
def update_polygon_map(time):
  fig, text = make_cptec_polygon(time)
  return fig, text


@app.callback(
    Output('cptec', 'figure'),
    [Input('radio-model-tab3', 'value')],
)
def update_cptec_predictions(model):
  return make_cptec_prediction(model)


@app.callback(
    Output('prob-graph', 'figure'),
    [Input('radio-model-tab4', 'value')],
)
def update_prob_graph(model):
  return make_prob_graph(model)


# Startup figures -----------------------------------
data_plots_fig = make_subplots(2, 1, shared_xaxes=True)
mapa = go.Figure()
ordemservico_fig = make_subplots(2, 1, shared_xaxes=True)

cptec_fig = make_subplots(2, 2, shared_xaxes=True)
cptec_poly_fig = make_cptec_polygon('Hoje')
prob_fig = go.Figure()

# Single Components ---------------------------------
# Tab 1 components
metricas_dropdown = dcc.Dropdown(
    options=[
        {'label': u'Radiação Solar', 'value': 'RadiacaoSolar'},
        {'label': u'Velocidade Do Vento', 'value': 'VelocidadeDoVento'},
        {'label': u'Umidade Relativa', 'value': 'UmidadeRelativa'},
        {'label': u'Pressão Atmosférica', 'value': 'PressaoAtmosferica'},
        {'label': u'Temperatura', 'value': 'TemperaturaDoAr'},
        {'label': u'Ponto De Orvalho', 'value': 'PontoDeOrvalho'},
        {'label': u'Precipitação', 'value': 'Precipitacao'},
    ],
    value='RadiacaoSolar',
    id='metrica',
    clearable=False
)
estacao_dropdown = dcc.Dropdown(
    options=[
        {'label': u'Camilópolis', 'value': '0'},
        {'label': u'Erasmo', 'value': '1'},
        {'label': u'Paraíso', 'value': '2'},
        {'label': u'RM', 'value': '3'},
        {'label': u'Vitória', 'value': '4'},
    ],
    value='0',
    multi=False,
    id='estacao',
    clearable=False
)
year_dropdown = dcc.Dropdown(
    options=year_options,
    value='2019',
    multi=False,
    id='ano',
    clearable=False
)
mes_dropdown = dcc.Dropdown(
    options=months_options,
    value='9',
    multi=False,
    id='mes',
    clearable=False
)
atualizar_button = html.Button(
    'Atualizar',
    id='update-button',
    n_clicks=0,
    className='tab1-update-btn'
)
data_subplots = dcc.Graph(
    id='plots',
    figure=data_plots_fig,
    style={'width': '80%'}
)

# Tab 2 components
map_figure = dcc.Graph(
    id='mapa',
    figure=mapa
)
year_slider = dcc.RangeSlider(
    min=list_of_years[0],
    max=list_of_years[-1],
    step=None,
    marks=year_options_slider,
    value=[list_of_years[0], list_of_years[-1]],
    id='year-slider',
    className='tab2-graphs-slider'
)
ordemservico_figure = dcc.Graph(
    id='os-subplots',
    figure=ordemservico_fig
)

# Tab 3 components
radio_button_model_tab3 = dcc.RadioItems(
    options=[
        {'label': 'WRF 05x05 km', 'value': 'wrf'},
        {'label': 'BAM 20x20 km', 'value': 'bam'},
    ],
    id='radio-model-tab3',
    value='wrf',
    labelStyle={'display': 'inline-block', 'marginLeft': '1em'}
)
cptec_figure = dcc.Graph(
    id='cptec',
    figure=cptec_fig,
    style={'width': '90%', 'marginTop': '1.5em'}
)
cptec_poly_figure = dcc.Graph(
    id='cptec-poly',
    figure=cptec_poly_fig
)
radio_button_poly = dcc.RadioItems(
    options=[
        {'label': 'Hoje', 'value': 'Hoje'},
        {'label': '48 horas', 'value': '48 horas'},
        {'label': '72 horas', 'value': '72 horas'}
    ],
    id='radio-poly',
    value='Hoje',
    labelStyle={'display': 'inline-block', 'marginLeft': '1em'}
)

# Tab 4 components
radio_button_model_tab4 = dcc.RadioItems(
    options=[
        {'label': 'WRF 05x05 km', 'value': 'wrf'},
        {'label': 'BAM 20x20 km', 'value': 'bam'},
    ],
    id='radio-model-tab4',
    value='wrf',
    labelStyle={'display': 'inline-block', 'marginLeft': '1em'}
)
prediction_prob_figure = dcc.Graph(
    id='prob-graph',
    figure=prob_fig,
    style={'width': '90%', 'marginTop': '1.5em'}
)

# App layout ----------------------------------------
root_layout = html.Div(className='root', children=[
    dcc.Tabs([
        # Tab 1
        dcc.Tab(label='Dados Históricos', className='tab1', children=[
            html.Div(className='tab1-container', children=[
                html.Div(className='tab1-form', children=[
                    html.H5('Filtros', className='tab1-form-title'),
                    html.Div(className='tab1-dropdown-wrapper', children=[
                        html.Div(className='tab1-dropdown-group', children=[
                            html.Label('Métrica', className='tab1-dropdown-labels'),
                            metricas_dropdown,
                            html.Label(u'Estação Meteorológica', className='tab1-dropdown-labels'),
                            estacao_dropdown,
                        ]),
                        html.Div(className='tab1-dropdown-group', children=[
                            html.Label(u'Ano', className='tab1-dropdown-labels'),
                            year_dropdown,
                            html.Label(u'Mês', className='tab1-dropdown-labels'),
                            mes_dropdown,
                        ]),
                    ]),
                    atualizar_button
                ]),
                data_subplots,
            ])
        ]),

        # Tab 2
        dcc.Tab(label='Histórico de Alagamentos', className='tab2', children=[
            html.Div(className='tab2-container', children=[
                html.Div(className='tab2-map-wrapper', children=[
                    html.Div(className='tab2-map-info', children='Regiões de Alagamento'),
                    html.Div(className='tab2-map-filter', children=[
                        map_figure,
                    ]),
                    html.Div(id='count', className='tab2-map-info'),
                ]),
                html.Div(className='tab2-graphs-wrapper', children=[
                    ordemservico_figure,
                    year_slider,
                ])
            ]),
        ]),

        # Tab 3
        dcc.Tab(label='Previsão do Tempo', className='tab3', children=[
            html.Div(className='tab3-container', children=[
                html.Div(className='tab3-model-select', children=[
                    html.Label('Selecione o modelo de previsão:'),
                    radio_button_model_tab3,
                ]),
                cptec_figure,
                html.Div(className='tab3-model-select', children=[
                    html.Label('Selecione o modelo de previsão:'),
                    radio_button_poly,
                ]),
                html.Div(id='warning', className='tab3-model-info'),
                html.Div(className='tab3-map-filter', children=[
                    cptec_poly_figure,
                ]),
            ]),
        ]),

        #Tab 4
        dcc.Tab(label='Previsão de Alagamento', className='tab4', children=[
            html.Div(className='tab4-container', children=[
                html.Div(className='tab4-model-select', children=[
                    html.Label('Selecione o modelo de previsão:'),
                    radio_button_model_tab4,
                ]),
                prediction_prob_figure,
            ])
        ])
    ])
])

app.layout = root_layout

if __name__ == '__main__':
  app.run_server(debug=True, port=8060,  threaded=True)
