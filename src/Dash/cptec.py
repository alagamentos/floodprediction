import urllib.request
import re
import json
from datetime import datetime

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

def get_polygon():
  with urllib.request.urlopen('http://tempo.cptec.inpe.br/avisos/') as response:
    html_source = str(response.read())

  htmlnow = re.search(r'^(.+?)\/\/ 48 horas', html_source).group(1)
  html48 = re.search(r'\/\/ 48 horas(.*?)\/\/ 72 horas', html_source).group(1)
  html72 = re.search(r'\/\/ 72 horas[^:]*:(.*)', html_source).group(1)

  output_dict = {}
  for text, html in zip(['Hoje', '48 horas', '72 horas'],[htmlnow, html48, html72]):

      output_dict[text] = {'geom':  [] ,
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

          # Populate Dict
          output_dict[text]['geom'].append(polygon_points)
          output_dict[text]['title'].append(title_string)

  return output_dict
