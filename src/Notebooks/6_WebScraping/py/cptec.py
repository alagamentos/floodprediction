#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import urllib.request
with urllib.request.urlopen('https://previsaonumerica.cptec.inpe.br/novo/meteograma/bam/sp/santo-andre') as response:
    html = str(response.read())


# In[ ]:


str(html)


# In[ ]:


import re

raw_string = {}
raw_string['precipitacao'] = re.search(r'"ident":\"precipitacao\",\"data\"\:(.*?),"uni', html).group(1)
raw_string['precipitacao_acc'] = re.search(r'\"precipitacao\-acumulada\",\"data\"\:(.*?),"uni', html).group(1)
raw_string['temperatura'] = re.search(r'\"ident\":\"temperatura\",\"data\":(.*?),"uni', html).group(1)
raw_string['temperatura_aparente'] = re.search(r'\"ident\"\:\"temperatura-aparente\",\"data\":(.*?),"uni', html).group(1)
raw_string['umidade_relativa'] = re.search(r'\"umidade\-relativa\",\"data\"\:(.*?),"uni', html).group(1)
raw_string['pressao'] = re.search(r'\"ident\":\"pressao\-ao\-nivel\-do\-mar\",\"data\":(.*?),"uni', html).group(1)


# In[ ]:


import json
from datetime import datetime 

def extract_data(source_string: str):
    res = json.loads(source_string)
    x_data = [point['x']for point in res]
    x_data_t = [datetime.fromtimestamp(t//1000) for t in x_data]
    y_data = [point['y']for point in res]
    
    return x_data, x_data_t, y_data


# In[ ]:


keys = list(raw_string.keys())

x_data, x_data_t, y_data = {}, {}, {}

for k in keys:
    print(k)
    x_data[k], x_data_t[k], y_data[k] = extract_data(raw_string[k])


# In[ ]:


import matplotlib.pyplot as plt

plt.plot(x_data_t['precipitacao_acc'], y_data['precipitacao_acc'])
plt.bar(x_data_t['precipitacao'], y_data['precipitacao'])
plt.show()


# In[ ]:


for k in keys:
    plt.figure()
    plt.title(k)
    plt.plot(x_data_t[k], y_data[k])

