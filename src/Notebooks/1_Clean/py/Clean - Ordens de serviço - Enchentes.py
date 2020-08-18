#!/usr/bin/env python
# coding: utf-8

# ## Import data

# In[ ]:


import pandas as pd


# In[ ]:


# Create dir

from os import mkdir
from os import path

if not path.exists("./cleandata"):
    mkdir('./cleandata')
    
if not path.exists("./cleandata/Ordens de serviço"):
    mkdir('./cleandata/Ordens de serviço')


# ## Import data

# In[ ]:


os_raw = pd.read_excel('../../../data/rawdata/Ordens de serviço/Enchentes - 01012010 a 30092019.xlsx')


# ## Separate date from data

# In[ ]:


os_null = os_raw.iloc[:,1:5].isnull().copy(deep = True) # Check if null columns 1:5

datas = os_raw[os_null.all(axis='columns')].copy(deep=True)
datas = datas.loc[datas['Unnamed: 0'] != '809 - DDC - Enchente / Inundação / Alagamento', :]
datas.drop(index=[1947, 1948], inplace=True)

dados = os_raw[~os_null.all(axis='columns')].copy(deep=True)
dados.drop(index=[0], inplace=True)


# ## Append date on data

# In[ ]:


for i in range(len(datas.index)):
    try:
        end = datas.index[i+1]
        start = datas.index[i]
        dados.loc[start:end,'Data'] = datas.iloc[i,0]     
    except IndexError:
        end = dados.index[-1]
        start = datas.index[i]
        dados.loc[start:end,'Data'] = datas.iloc[i,0] 


# In[ ]:


dados.tail(5)


# In[ ]:


dados.columns = ['ID1',
                 'ID2',
                 'Hora',
                 'Endereco1',
                 'Endereco2',
                 'Comentario1',
                 'Comentario2',
                 'Status',
                 'Data']


# In[ ]:


dados.head()


# In[ ]:


dados = dados.replace({';': ','}, regex=True)


# # Save data

# In[ ]:


dados.to_csv(
    r'../../../data/cleandata/Ordens de serviço/Enchentes - 01012010 a 30092019.csv',
    sep=';',
    index=False)

