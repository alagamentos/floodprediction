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
    make_cptec_polygon, make_prob_graph, make_rain_ordem_servico_plot_grouped_by, BG_DARK

import os


# Prepdata ----------------------------------

estacoes_path = 'data/lat_lng_estacoes.csv'
ordemservico_path = 'data/Enchentes_LatLong.csv'
precipitacao_path = 'data/Precipitacao.csv'

est = pd.read_csv(estacoes_path, sep=';').iloc[:-1, :]
label = pd.read_csv(ordemservico_path, sep=';')
precipitacao = pd.read_csv(precipitacao_path, sep=';')

# Label
label['Data'] = pd.to_datetime(label['Data'], yearfirst=True)

gb_label = label.groupby('Data').count().reset_index()[['Data', 'lat']]
gb_label.columns = ['Data', 'count']
gb_label['Ano'] = gb_label['Data'].dt.year
gb_label['Mes'] = gb_label['Data'].dt.month

# Precipitacao Hora em Hora
precipitacao['Data_Hora'] = pd.to_datetime(precipitacao['Data_Hora'], yearfirst=True)
precipitacao['Data'] = pd.to_datetime(precipitacao['Data_Hora'].dt.date, yearfirst=True)
rain_sum = precipitacao.groupby('Data').sum().reset_index()[['Data', 'Precipitacao_2']]

rain_sum['Ano'] = rain_sum['Data'].dt.year
rain_sum['Mes'] = rain_sum['Data'].dt.month

list_of_years = list(range(2011, 2020, 1))
year_options = [{'label': str(y), 'value': y} for y in list_of_years]
year_options_slider = {y: str(y) for y in list_of_years}

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
app = dash.Dash(
    __name__,
    title='Sistema Inteligente de Previsão de Alagamentos',
    external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'],
    url_base_pathname='/',
    assets_external_path='/assets/',
    assets_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets/')
)

# Callbacks ----------------------------------------


@app.callback(
    [Output('mes-hist-dados', 'options'), Output('mes-hist-dados', 'value')],
    [Input('ano-hist-dados', 'value')],
    state=[State('mes-hist-dados', 'value')]
)
def update_month_options(ano, mes):
  if int(ano) == 2019:
    year_options = [{'label': k, 'value': int(v)} for k, v in dict_months.items() if int(v) < 10]
    if int(mes) >= 10:
      mes = str(9)
    return year_options, mes
  else:
    return [{'label': k, 'value': int(v)} for k, v in dict_months.items()], mes


@app.callback(
    Output('plots-hist-dados', 'figure'),
    [Input('btn-update-hist-dados', 'n_clicks')],
    state=[
        State('metrica-hist-dados', 'value'),
        State('estacao-hist-dados', 'value'),
        State('ano-hist-dados', 'value'),
        State('mes-hist-dados', 'value'),
    ]
)
def update_graphs(n_clicks, col, est, year, month):
  return make_data_repair_plots(col, est, year, month)


@app.callback(
    [Output('kpi-ordem-servico', 'children'),
     Output('mapa-alagamentos', 'figure'),
     Output('ordem-servico-subplots', 'figure'),
     Output('ordem-servico-subplots-month', 'figure'),
     Output('kpi-precipitacao-media', 'children')],
    [Input('slider-ordem-servico', 'value')],
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
  ordem_servico_figure_month = make_rain_ordem_servico_plot_grouped_by(gb_label_plot, rain_sum_plot)
  count = label_plot.shape[0]

  rain_sum_plot['Ano'] = rain_sum_plot['Data'].dt.year
  media_anual = int(rain_sum_plot.groupby(['Ano']).sum()[['Precipitacao_2']].mean().item())

  return count, mapa, ordem_servico_figure, ordem_servico_figure_month, f'{media_anual} mm/ano'


@app.callback(
    [Output('cptec-mapa', 'figure'), Output('cptec-mapa-warning', 'children')],
    [Input('rb-cptec-mapa', 'value')],
)
def update_polygon_mapa(time):
  fig, text = make_cptec_polygon(time)
  return fig, text.lower().capitalize()


@app.callback(
    Output('cptec-graphs', 'figure'),
    [Input('rb-cptec-graphs', 'value')],
)
def update_cptec_predictions(model):
  return make_cptec_prediction(model)


@app.callback(
    Output('prop-alagamento-graph', 'figure'),
    [Input('rb-prop-alagamento-graph', 'value')],
)
def update_prob_graph(model):
  return make_prob_graph(model)


# Startup figures -----------------------------------
data_plots_fig = make_subplots(2, 1, shared_xaxes=True)
data_plots_fig.update_layout(margin=dict(l=50, r=30, t=40, b=30),
                             template='plotly_dark',
                             paper_bgcolor=BG_DARK, plot_bgcolor=BG_DARK)
mapa = go.Figure()
ordemservico_fig = make_subplots(2, 1, shared_xaxes=True)
ordemservico_fig_month = make_subplots(2, 1, shared_xaxes=True)

cptec_fig = make_subplots(2, 2, shared_xaxes=True)
cptec_poly_fig = make_cptec_polygon('Hoje')
prob_fig = go.Figure()

# Single Components ---------------------------------

map_config = {'modeBarButtonsToRemove': ['select2d', 'lasso2d', 'toggleHover', 'toImage'],
              'displaylogo': False}

graph_config = {'modeBarButtonsToRemove': ['select2d', 'lasso2d', 'toggleHover', 'toImage',
                                           'zoomIn2d', 'zoomOut2d', 'toggleHover', 'resetViews', 'toggleSpikelines',
                                           'hoverClosestCartesian', 'hoverCompareCartesian', 'autoScale2d'],
                'displaylogo': False}

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
    id='metrica-hist-dados',
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
    id='estacao-hist-dados',
    clearable=False
)
year_dropdown = dcc.Dropdown(
    options=year_options,
    value='2019',
    multi=False,
    id='ano-hist-dados',
    clearable=False
)
mes_dropdown = dcc.Dropdown(
    options=months_options[0:10],
    value='9',
    multi=False,
    id='mes-hist-dados',
    clearable=False
)
atualizar_button = html.Button(
    'Atualizar',
    id='btn-update-hist-dados',
    n_clicks=0,
    className='button'
)
data_subplots = dcc.Graph(
    id='plots-hist-dados',
    figure=data_plots_fig,
    config=graph_config,
    style={'width': '100%', 'boxShadow': '0px 1px 5px 0px rgba(255,255,255,.2)', 'borderRadius': '6px'}
)

# Tab 2 components
map_figure = dcc.Graph(
    id='mapa-alagamentos',
    figure=mapa,
    config=map_config,
)
year_slider = dcc.RangeSlider(
    min=list_of_years[0],
    max=list_of_years[-1],
    step=None,
    marks=year_options_slider,
    value=[list_of_years[0], list_of_years[-1]],
    id='slider-ordem-servico',
    className='slider'
)
ordemservico_figure = dcc.Graph(
    id='ordem-servico-subplots',
    figure=ordemservico_fig,
    config=graph_config,
    style={'marginBottom': '2.5em', 'boxShadow': '0px 1px 5px 0px rgba(255,255,255,.2)', 'borderRadius': '6px'}
)

ordemservico_figure_month = dcc.Graph(
    id='ordem-servico-subplots-month',
    figure=ordemservico_fig_month,
    config=graph_config,
    style={'boxShadow': '0px 1px 5px 0px rgba(255,255,255,.2)', 'borderRadius': '6px'}
)

# Tab 3 components
radio_button_model_tab3 = dcc.RadioItems(
    options=[
        {'label': 'WRF 05x05 km', 'value': 'wrf'},
        {'label': 'BAM 20x20 km', 'value': 'bam'},
    ],
    id='rb-cptec-graphs',
    value='wrf',
    labelStyle={'display': 'flex', 'alignItems': 'center', 'marginLeft': '1em'},
    className='model-selection-items'
)
cptec_figure = dcc.Graph(
    id='cptec-graphs',
    figure=cptec_fig,
    config=graph_config,
    style={'width': '100%', 'marginTop': '1.5em',
           'boxShadow': '0px 1px 5px 0px rgba(255,255,255,.2)', 'borderRadius': '6px'}
)
cptec_poly_figure = dcc.Graph(
    id='cptec-mapa',
    figure=cptec_poly_fig,
    config=map_config,
)
radio_button_poly = dcc.RadioItems(
    options=[
        {'label': 'Hoje', 'value': 'Hoje'},
        {'label': '48 horas', 'value': '48 horas'},
        {'label': '72 horas', 'value': '72 horas'}
    ],
    id='rb-cptec-mapa',
    value='Hoje',
    labelStyle={'display': 'flex', 'alignItems': 'center', 'marginLeft': '1em'},
    className='model-selection-items'
)

# Tab 4 components
radio_button_model_tab4 = dcc.RadioItems(
    options=[
        {'label': 'WRF 05x05 km', 'value': 'wrf'},
        {'label': 'BAM 20x20 km', 'value': 'bam'},
    ],
    id='rb-prop-alagamento-graph',
    value='wrf',
    labelStyle={'display': 'flex', 'alignItems': 'center', 'marginLeft': '1em'},
    className='model-selection-items'
)
prediction_prob_figure = dcc.Graph(
    id='prop-alagamento-graph',
    figure=prob_fig,
    config=graph_config,
    style={'width': '100%', 'marginTop': '1.5em',
           'boxShadow': '0px 1px 5px 0px rgba(255,255,255,.2)', 'borderRadius': '6px'}
)

# App layout ----------------------------------------
root_layout = html.Div(className='root', children=[
    html.Div(className='header', children=[
        html.H1('Sistema Inteligente de Previsão de Alagamentos', className='header-title'),
        html.A(href='https://maua.br', target='_blank', className='header-logo-maua', children=[
            html.Img(src='/assets/maua-logo-branco.png'),
        ]),
        html.A(href='https://www2.santoandre.sp.gov.br', target='_blank', className='header-logo-sa', children=[
            html.Img(src='/assets/prefeitura-sa-logo.png'),
        ]),
    ]),
    dcc.Tabs([
        # Tab 1
        dcc.Tab(label='Dados Históricos', className='tab1', children=[
            html.Div(className='tab1-container', children=[
                html.Div(className='tab1-header', children=[
                    html.Div(className='tab1-form', children=[
                      html.H5('Filtros', className='tab1-form-title'),
                      html.Div(className='tab1-dropdown-wrapper', children=[
                          html.Div(className='tab1-dropdown-group', children=[
                              html.Label(u'Dado Meteorológico', className='dropdown-labels'),
                              metricas_dropdown,
                              html.Label(u'Estação Meteorológica', className='dropdown-labels'),
                              estacao_dropdown,
                          ]),
                          html.Div(className='tab1-dropdown-group', children=[
                              html.Label(u'Ano', className='dropdown-labels'),
                              year_dropdown,
                              html.Label(u'Mês', className='dropdown-labels'),
                              mes_dropdown,
                          ]),
                      ]),
                        atualizar_button
                    ]),
                    html.Div(className='info', children=[
                        html.H3('Tratamento dos dados', className='info-title'),
                        html.P(
                            'Nesta página é possível visualizar os dados das 5 estações meteorológicas espalhadas pelo município de Santo André. Por terem sido gerados a partir de medições de sensores ao longo de anos, fez-se necessário realizar o tratamento dos mesmos antes de usá-los nos modelos de previsão de alagamentos.',
                            className='info-content'
                        ),
                        html.P(
                            'Para corrigí-los foram utilizadas técnicas matemáticas e de aprendizagem de máquina capazes de explicar de forma bastante acurada o comportamento real dos dados, conforme ilustram os gráficos abaixo.',
                            className='info-content'
                        ),
                    ]),
                ]),
                html.Div(className='tab1-graphs-wrapper', children=[
                    data_subplots,
                ]),
            ])
        ]),

        # Tab 2
        dcc.Tab(label='Histórico de Alagamentos', className='tab2', children=[
            html.Div(className='tab2-container', children=[
                html.Div(className='tab2-side-wrapper', children=[
                    html.Div(className='tab2-cards-wrapper', children=[
                        html.Div(className='card', children=[
                            html.Div(className='card-wrapper', children=[
                                html.Span(id='kpi-ordem-servico', className='card-value'),
                            ]),
                            html.H5('Nº de ordens de serviço', className='card-title'),
                        ]),
                        html.Div(className='card', children=[
                            html.Div(className='card-wrapper', children=[
                                html.Span('', id='kpi-precipitacao-media', className='card-value'),
                            ]),
                            html.H5('Precipitação média no período', className='card-title'),
                        ]),
                    ]),
                    html.Div(className='tab2-slider-wrapper', children=[
                        html.H5('Período', className='slider-title'),
                        year_slider,
                    ]),
                    html.Div(className='tab2-map-wrapper', children=[
                        html.H5(className='tab2-map-title', children='Regiões de Alagamento'),
                        html.Div(className='tab2-map-filter', children=[
                            map_figure,
                        ]),
                    ]),
                ]),
                html.Div(className='tab2-graphs-wrapper', children=[
                    ordemservico_figure,
                    ordemservico_figure_month,
                ]),
            ]),
        ]),

        # Tab 3
        dcc.Tab(label='Previsão do Tempo', className='tab3', children=[
            html.Div(className='tab3-container', children=[
                html.Div(className='tab3-graphs-wrapper', children=[
                    html.Div(className='model-selection', children=[
                        html.Label('Selecione o modelo de previsão:'),
                        radio_button_model_tab3,
                    ]),
                    cptec_figure,
                ]),
                html.Div(className='model-selection', children=[
                    html.Label('Selecione o período de previsão:'),
                    radio_button_poly,
                ]),
                html.Div(className='tab3-wrapper', children=[
                    html.Div(className='tab3-info-wrapper', children=[
                        html.Div(className='card', children=[
                            html.Div(className='card-wrapper', children=[
                                html.Span(id='cptec-mapa-warning', className='card-value'),
                                html.A(href='http://tempo.cptec.inpe.br/avisos', target='_blank',
                                       className='card-tooltip')
                            ]),
                            html.H5('Aviso meteorológico para Santo André', className='card-title'),
                        ]),
                        html.Div(className='info', children=[
                            html.H3('Previsões numéricas', className='info-title'),
                            html.P(className='info-content', children=[
                                'Os dados de previsão numérica utilizados nesta página pertencem ao ',
                                html.A('CPTEC/INPE', href='https://www.cptec.inpe.br/sp/santo-andre',
                                       target='_blank', className='link'),
                                ' e são referentes apenas ao município de Santo André. Os números que seguem os modelos atmosféricos WRF e BAM dizem respeito a resolução espacial de cada previsão. Quanto menor essa resolução, mais precisa é a previsão de tempo.',
                            ]),
                            html.P(className='info-content', children=[
                                'O ',
                                html.A('Centro de Previsão de Tempo e Estudos Climáticos (CPTEC)',
                                       href='https://www.cptec.inpe.br', target='_blank', className='link'),
                                ' do ',
                                html.A('Instituto Nacional de Pesquisas Espaciais (INPE)',
                                       href='http://www.inpe.br', target='_blank', className='link'),
                                ' é o centro mais avançado de previsão numérica de tempo e clima da América Latina, fornecendo previsões de tempo de curto e médio prazos e climáticas de alta precisão desde o início de 1995, além de dominar técnicas de modelagem numérica altamente complexas da atmosfera e dos oceanos a fim de prever condições futuras.',
                            ]),
                        ]),
                        html.Div(className='info', children=[
                            html.H3('Modelos atmosféricos', className='info-title'),
                            html.P(className='info-content', children=[
                                'O modelo de ',
                                html.A('Pesquisa e Previsão do Tempo (WRF - Weather Research and Forecasting)',
                                       href='https://en.wikipedia.org/wiki/Weather_Research_and_Forecasting_Model', target='_blank', className='link'),
                                ', desenvolvido por agências e universidades norte-americanas, é um sistema de previsão numérica do tempo projetado para atender às necessidades de pesquisa atmosférica e de previsão operacional. Sistemas de previsão numérica referem-se à simulações e previsões da atmosfera por modelos computacionais com o intuito de prever o tempo com vários dias de antecedência. O WRF possui dois núcleos dinâmicos, um sistema de assimilação de dados e uma arquitetura de software que permitem a computação paralela e extensibilidade do sistema, além de atender a uma ampla gama de aplicações meteorológicas em escalas que variam de metros a milhares de quilômetros.',
                            ]),
                            html.P(className='info-content', children=[
                                'O ',
                                html.A('Modelo Atmosférico Brasileiro (BAM - Brazilian Atmospheric Model)',
                                       href='http://www.cntu.org.br/new/component/content/article?id=3807', target='_blank', className='link'),
                                ', desenvolvido pelo CPTEC, é um modelo global de previsão de tempo capaz de gerar as condições iniciais para a execução de modelos regionais, além de ser utilizado na geração das previsões climáticas sazionais e de cenários climáticos de mais longo prazo. O BAM apresenta-se como solução para deficiências de modelos utilizados anteriormente, possuindo vantagens como: a melhora na previsão para o sudeste do Brasil em decorrência de uma maior resolução horizontal, modelando melhor eventos com orografia complexa; e o aumento da resolução espacial com a qual as previsões de tempo e clima são processadas.',
                            ]),
                        ]),
                    ]),
                    html.Div(className='tab3-map-wrapper', children=[
                        html.H5(className='tab3-map-title', children='Avisos Meteorológicos'),
                        html.Div(className='tab3-map-filter', children=[
                            cptec_poly_figure,
                        ]),
                        html.Div(className='tab3-map-labels', children=[
                            html.Div(className='tab3-map-label-wrapper', children=[
                                html.Span('Aviso de observação', style={
                                          'backgroundColor': 'rgba(223, 209, 126, .25)'}, className='tab3-map-label'),
                            ]),
                            html.Div(className='tab3-map-label-wrapper', children=[
                                html.Span('Aviso de atenção', style={
                                          'backgroundColor': 'rgba(234, 176, 9, .25)'}, className='tab3-map-label'),
                            ]),
                            html.Div(className='tab3-map-label-wrapper', children=[
                                html.Span('Aviso especial', style={
                                          'backgroundColor': 'rgba(253, 63, 110, .25)'}, className='tab3-map-label'),
                            ]),
                            html.Div(className='tab3-map-label-wrapper', children=[
                                html.Span('Aviso extraordinário de risco iminente', style={
                                          'backgroundColor': 'rgba(44, 80, 237, .25)'}, className='tab3-map-label'),
                            ]),
                            html.Div(className='tab3-map-label-wrapper', children=[
                                html.Span('Aviso cessado', style={
                                          'backgroundColor': 'rgba(195, 195, 195, .25)'}, className='tab3-map-label'),
                            ]),
                        ]),
                    ]),
                ]),
            ]),
        ]),

        #Tab 4
        dcc.Tab(label='Previsão de Alagamento', className='tab4', children=[
            html.Div(className='tab4-container', children=[
                html.Div(className='tab4-graphs-wrapper', children=[
                    html.Div(className='model-selection', children=[
                        html.Label('Selecione o modelo de previsão:'),
                        radio_button_model_tab4,
                    ]),
                    prediction_prob_figure,
                ]),
            ])
        ]),
    ]),
    html.Div(className='copyright', children=[
        'Desenvolvido por:',
        html.A('Felipe Andrade', href='https://github.com/Kaisen-san', target='_blank', className='link'),
        html.A('Felipe Ippolito', href='https://github.com/feippolito', target='_blank', className='link'),
        html.A('Vinícius Pereira', href='https://github.com/VinPer', target='_blank', className='link'),
    ]),
])


app.layout = root_layout
server = app.server


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8080, debug=True, use_reloader=True)
