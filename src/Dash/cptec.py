import urllib.request
import re
import json
from datetime import datetime

def extract_data(source_string: str):
    res = json.loads(source_string)
    x_data = [point['x']for point in res]
    x_data_t = [datetime.fromtimestamp(t//1000) for t in x_data]
    y_data = [point['y']for point in res]

    return x_data, x_data_t, y_data

def get_prediction():
  with urllib.request.urlopen('https://previsaonumerica.cptec.inpe.br/novo/meteograma/bam/sp/santo-andre') as response:
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
