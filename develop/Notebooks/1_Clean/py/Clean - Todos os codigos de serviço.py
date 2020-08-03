#!/usr/bin/env python
# coding: utf-8

# ## Import data

# In[1]:


import pandas as pd


# In[3]:


# Create dir

from os import mkdir
from os import path

path_ = "../../../data/cleandata"
if not path.exists(path_):
    mkdir(path_)
    
path_ = "../../../data/cleandata/Ordens de serviço"
if not path.exists(path_):
    mkdir(path_)


# In[4]:


os_cod = pd.read_excel('../../../data/rawdata/Ordens de serviço/Todos os codigos de serviço - 01012010 a 30092019.xlsx')


# ## Separate date from data

# In[5]:


cod_null = os_cod.iloc[:,1:5].isnull().copy(deep = True) # Columns 1:5 are null if (tipo or data)

datas_cod = os_cod[cod_null.all(axis='columns')].copy(deep=True)
datas_cod = datas_cod[:-2] # Remove last 2 rows

dados_cod = os_cod[~cod_null.all(axis='columns')].copy(deep=True)
dados_cod.drop(index=[0], inplace=True)


# In[6]:


tipo_cod = datas_cod[datas_cod['Unnamed: 0'].str.contains("-")]
datas_cod = datas_cod[~datas_cod['Unnamed: 0'].str.contains("-")]


# In[7]:


for i in range(len(tipo_cod.index)):
    start = tipo_cod.index[i]
    try:
        end = tipo_cod.index[i+1]
        dados_cod.loc[start:end,'Tipo'] = tipo_cod.iloc[i,0]
    except:
        end = dados_cod.index[-1]
        dados_cod.loc[start:end,'Tipo'] = tipo_cod.iloc[i,0]


# In[8]:


for i in range(len(datas_cod.index)):
    start = datas_cod.index[i]
    try:
        end = datas_cod.index[i+1]
        dados_cod.loc[start:end,'datas'] = datas_cod.iloc[i,0]
    except:
        end = dados_cod.index[-1]
        dados_cod.loc[start:end,'datas'] = datas_cod.iloc[i,0]


# In[9]:


dados_cod.columns = ['ID?','OS','Hora','Endereco1','Endereco2','?','Comentario1','Comentario2',
                     'Status', 'Tipo','Data']


# # Codigos de Serviço

# In[10]:


dados_cod.to_csv(
    r'../../../data/cleandata/Ordens de serviço/Todos os codigos de serviço - 01012010 a 30092019.csv',
    sep=';',
    index=False)

