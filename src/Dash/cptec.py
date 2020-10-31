import urllib.request
import re
import json
from datetime import datetime
from shapely.geometry import Polygon
from urllib.request import urlopen

def get_prediction(model):
  if model == 'bam':
    with urllib.request.urlopen('https://previsaonumerica.cptec.inpe.br/novo/meteograma/bam/sp/santo-andre') as response:
        html = str(response.read())
  elif model == 'wrf':
    with urllib.request.urlopen('https://previsaonumerica.cptec.inpe.br/novo/meteograma/wrf/sp/santo-andre') as response:
      html = str(response.read())

  raw_string = {}
  raw_string['precipitacao'] = re.search(r'"ident":\"precipitacao\",\"data\"\:(.*?),"uni', html).group(1)
  raw_string['precipitacao_acc'] = re.search(r'\"precipitacao\-acumulada\",\"data\"\:(.*?),"uni', html).group(1)
  raw_string['temperatura'] = re.search(r'\"ident\":\"temperatura\",\"data\":(.*?),"uni', html).group(1)
  raw_string['temperatura_aparente'] = re.search(r'\"ident\"\:\"temperatura-aparente\",\"data\":(.*?),"uni', html).group(1)
  raw_string['umidade_relativa'] = re.search(r'\"umidade\-relativa\",\"data\"\:(.*?),"uni', html).group(1)
  raw_string['pressao'] = re.search(r'\"ident\":\"pressao\-ao\-nivel\-do\-mar\",\"data\":(.*?),"uni', html).group(1)

  keys = list(raw_string.keys())
  x_data, x_data_t, y_data = {}, {}, {}
  for k in keys:
      x_data[k], x_data_t[k], y_data[k] = extract_data(raw_string[k])

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
  SA = [ i for i in counties['features'] if i['properties']['name'] == 'Santo André' ][0]


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
    text = 'Sem Alerta'

  return text

def get_polygon():

  value_dict = {'Aviso de Observação':1,
              'Aviso de Atenção': 2,
              'Aviso Especial': 3,
              'Aviso Extraordinário de Risco Iminente':4,
              'Aviso Cessado': 5
              }

  inverse_value_dict = {
              0:'Sem Alerta',
              1:'Aviso de Observação',
              2:'Aviso de Atenção',
              3:'Aviso Especial',
              4:'Aviso Extraordinário de Risco Iminente',
              5:'Aviso Cessado'}

  intersection = {}
  output_dict = {}

  SA = get_SantoAndre_polygon()
  SA_polygon = Polygon(SA['geometry']['coordinates'][0])
  SA_layer = dict(sourcetype = 'geojson',
              source = SA ,
              below='',
              type = 'fill',
              opacity=0.25,
              color = 'white')

  with urllib.request.urlopen('http://tempo.cptec.inpe.br/avisos/') as response:
    html_source = str(response.read())

  htmlnow = re.search(r'^(.+?)\/\/ 48 horas', html_source).group(1)
  html48 = re.search(r'\/\/ 48 horas(.*?)\/\/ 72 horas', html_source).group(1)
  html72 = re.search(r'\/\/ 72 horas(.*)', html_source).group(1)

  for text, html in zip(['Hoje', '48 horas', '72 horas'],[htmlnow, html48, html72]):

    intersection[text] = 0

    output_dict[text] = {'geom':  [],
                          'title': []}

    poly_func_string_list = re.findall(r'google.maps.Polygon(.*?)\)', html)
    poly_func_string = re.search(r'new google.maps.Polygon\((.*?)\}\)', html)

    if poly_func_string is None: continue

    poly_func_string = poly_func_string.group(1)

    for poly_func_string in poly_func_string_list:

      polygon_string = re.search(r'paths\: (.*?),\\n', poly_func_string).group(1)

      polygon_string = polygon_string.replace('lat', '"lat"').replace('lng', '"lng"')
      polygon_dict = json.loads(polygon_string)
      polygon_points = [(p['lng'], p['lat'] ) for p in polygon_dict]

      title_string = re.search(r'title\:\"(.*?)"', poly_func_string).group(1)
      title_string = title_string.replace('\\xc3\\xa7','ç').replace('\\xc3\\xa3','ã').replace('\\xc3\\xa1','á')

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
