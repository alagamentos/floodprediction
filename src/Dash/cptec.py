import urllib.request
import re
import json
from datetime import datetime, timedelta
from requests import get
from shapely.geometry import Polygon
from urllib.request import urlopen


def get_prediction(model):

  # Obtendo dia para buscar previsão
  if len(str(datetime.now().day)) == 1:
      dia = '0' + str(datetime.now().day)
  else:
      dia = str(datetime.now().day)
  if len(str(datetime.now().month)) == 1:
      mes = '0' + str(datetime.now().month)
  else:
      mes = str(datetime.now().month)
  ano = datetime.now().year

  if model == 'wrf7':
    try:
      res_json = get(
          f'http://ftp.cptec.inpe.br/modelos/tempo/WRF/ams_07km/recortes/grh/json/{ano}/{mes}/{dia}/00/4704.json').json()
    except:
      res_json = get(
        f'http://ftp.cptec.inpe.br/modelos/tempo/WRF/ams_07km/recortes/grh/json/{ano}/{mes}/{int(dia)-1}/00/4704.json').json()
  elif model == 'wrf':
    try:
      res_json = get(
          f'http://ftp.cptec.inpe.br/modelos/tempo/WRF/ams_05km/recortes/grh/json/{ano}/{mes}/{dia}/00/4704.json').json()
    except:
      res_json = get(
        f'http://ftp.cptec.inpe.br/modelos/tempo/WRF/ams_05km/recortes/grh/json/{ano}/{mes}/{int(dia)-1}/00/4704.json').json()
  elif model == 'bam':
    try:
      res_json = get(
          f'http://ftp.cptec.inpe.br/modelos/tempo/BAM/TQ0666L064/recortes/grh/json/{ano}/{mes}/{dia}/00/4704.json').json()
    except:
      res_json = get(
        f'http://ftp.cptec.inpe.br/modelos/tempo/BAM/TQ0666L064/recortes/grh/json/{ano}/{mes}/{int(dia)-1}/00/4704.json').json()

  #Dados puros
  raw_data = res_json['datasets'][0]['data']

  # Obtenção de dados necessários
  x_data_t = {}
  y_data = {}

  # Obtenção de datetime corrigido
  initial_date = datetime.fromisoformat(raw_data[0]['date']) # Data inicial devido a divergência entre modelos
  x_data_t["precipitacao"] = [initial_date + timedelta(hours = i['fcst']) for i in raw_data]
  x_data_t["precipitacao_acc"] = [initial_date + timedelta(hours = i['fcst']) for i in raw_data]
  x_data_t["temperatura"] = [initial_date + timedelta(hours = i['fcst']) for i in raw_data]
  x_data_t["temperatura_aparente"] = [initial_date + timedelta(hours = i['fcst']) for i in raw_data]
  x_data_t["pressao"] = [initial_date + timedelta(hours = i['fcst']) for i in raw_data]
  x_data_t["umidade_relativa"] = [initial_date + timedelta(hours = i['fcst']) for i in raw_data]

  # Obtenção de dados meteorológicos
  y_data["precipitacao"] = [i['prec'] for i in raw_data]
  y_data["temperatura"] = [i['temp'] for i in raw_data]
  y_data["temperatura_aparente"] = [i['heat_index'] for i in raw_data]
  y_data["pressao"] = [i['press'] for i in raw_data]
  y_data['umidade_relativa'] = [i['ur'] for i in raw_data]
  acc = 0
  for i in raw_data:
    # soma a precipitação atual com a acumulada anterior
    precipitacao_acc = i['prec']+acc
    # atualiza o valor da precipitação acumulada
    acc = precipitacao_acc
    y_data['precipitacao_acc'] = (precipitacao_acc)

  return x_data_t, y_data


def extract_data(source_string: str):
    res = json.loads(source_string)
    x_data = [point['x']for point in res]
    x_data_t = [datetime.fromtimestamp(t//1000) for t in x_data]
    y_data = [point['y']for point in res]

    return x_data, x_data_t, y_data


def get_SantoAndre_polygon():

  path = 'https://raw.githubusercontent.com/tbrugz/geodata-br/master/geojson/geojs-35-mun.json'

  with urlopen(path) as response:
      counties = json.load(response)
  SA = [i for i in counties['features'] if i['properties']['name'] == 'Santo André'][0]

  return SA


def verify_title_string(t):
  if 'observação' in t.lower() or 'observacao' in t.lower():
    text = 'Aviso de Observação'
  elif 'atenção' in t.lower() or 'atencao' in t.lower():
    text = 'Aviso de Atenção'
  elif 'especial' in t.lower():
    text = 'Aviso Especial'
  elif 'extraordinário' in t.lower() or 'extraordinario' in t.lower() or 'risco' in t.lower():
    text = 'Aviso Extraordinário de Risco Iminente'
  elif 'cessado' in t.lower():
    text = 'Aviso Cessado'
  else:
    text = 'Sem Aviso'

  return text


def get_polygon():

  value_dict = {'Aviso de Observação': 1,
                'Aviso de Atenção': 2,
                'Aviso Especial': 3,
                'Aviso Extraordinário de Risco Iminente': 4,
                'Aviso Cessado': 5
                }

  inverse_value_dict = {
      0: 'Sem Aviso',
      1: 'Aviso de Observação',
      2: 'Aviso de Atenção',
      3: 'Aviso Especial',
      4: 'Aviso Extraordinário de Risco Iminente',
      5: 'Aviso Cessado'}

  intersection = {}
  output_dict = {}

  SA = get_SantoAndre_polygon()
  SA_polygon = Polygon(SA['geometry']['coordinates'][0])
  SA_layer = dict(sourcetype='geojson',
                  source=SA,
                  below='',
                  type='fill',
                  opacity=0.25,
                  color='#1c1e2f')

  with urllib.request.urlopen('http://tempo.cptec.inpe.br/avisos/') as response:
    html_source = str(response.read())

  htmlnow = re.search(r'^(.+?)\/\/ 48 horas', html_source).group(1)
  html48 = re.search(r'\/\/ 48 horas(.*?)\/\/ 72 horas', html_source).group(1)
  html72 = re.search(r'\/\/ 72 horas(.*)', html_source).group(1)

  for text, html in zip(['Hoje', '48 horas', '72 horas'], [htmlnow, html48, html72]):

    intersection[text] = 0

    output_dict[text] = {'geom':  [],
                         'title': []}

    poly_func_string_list = re.findall(r'google.maps.Polygon(.*?)\)', html)
    poly_func_string = re.search(r'new google.maps.Polygon\((.*?)\}\)', html)

    if poly_func_string is None:
      continue

    poly_func_string = poly_func_string.group(1)

    for poly_func_string in poly_func_string_list:

      polygon_string = re.search(r'paths\: (.*?),\\n', poly_func_string).group(1)

      polygon_string = polygon_string.replace('lat', '"lat"').replace('lng', '"lng"')
      polygon_dict = json.loads(polygon_string)
      polygon_points = [(p['lng'], p['lat']) for p in polygon_dict]

      title_string = re.search(r'title\:\"(.*?)"', poly_func_string).group(1)
      title_string = title_string.replace('\\xc3\\xa7', 'ç').replace('\\xc3\\xa3', 'ã').replace('\\xc3\\xa1', 'á')

      title_string = verify_title_string(title_string)

      # Populate Dict
      output_dict[text]['geom'].append(polygon_points)
      output_dict[text]['title'].append(title_string)

      polygon = Polygon(polygon_points)

      if SA_polygon.intersects(polygon):
        if value_dict[title_string] > intersection[text]:
          intersection[text] = value_dict[title_string]

  for k in intersection.keys():
    output_dict[k]['aviso'] = inverse_value_dict[intersection[k]]

  return output_dict, SA_polygon, SA_layer
