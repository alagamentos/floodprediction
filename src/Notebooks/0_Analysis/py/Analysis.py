#!/usr/bin/env python
# coding: utf-8

# In[1]:


import datetime
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

import pandas as pd
import seaborn as sns


try:
    from jupyterthemes import jtplot
    jtplot.style()
except:
    pass


# In[2]:


est0 = pd.read_csv('../../../data/cleandata/Info pluviometricas/Concatenated Data/Camilopolis/Camilopolis.csv', sep=';')
est1 = pd.read_csv('../../../data/cleandata/Info pluviometricas/Concatenated Data/Erasmo Assunção/Erasmo Assunção.csv', sep=';')
est2 = pd.read_csv('../../../data/cleandata/Info pluviometricas/Concatenated Data/Paraiso/Paraiso.csv', sep=';')
est3 = pd.read_csv('../../../data/cleandata/Info pluviometricas/Concatenated Data/RM 9/RM 9.csv', sep=';')
est4 = pd.read_csv('../../../data/cleandata/Info pluviometricas/Concatenated Data/Vitória/Vitória.csv', sep=';')

est = [est0, est1, est2, est3, est4]


# In[3]:


ose = pd.read_csv('../../../data/cleandata/Ordens de serviço/Enchentes - 01012010 a 30092019.csv',
                 sep=';') # Ordem de serviço enchentes


# In[4]:


min_ = []
max_ = []
for e in est:
    min_.append(min(e['Data']))
    max_.append(max(e['Data']))
print(min(min_), max(max_))


# In[5]:


merge1 = est0.merge(est1, on = 'Data_Hora', how = 'outer', suffixes = ('_0', '_1'))


# In[6]:


merge2 = est2.merge(est3, on = 'Data_Hora', how = 'outer', suffixes = ('_2', '_3'))


# In[7]:


new_cols = []
for col in est4.columns:
    if col != 'Data_Hora':
        col = col + '_4'
    new_cols.append(col)
    
est4.columns = new_cols


# In[8]:


merge3 = merge1.merge(merge2, on = 'Data_Hora', how = 'outer')


# In[9]:


final = merge3.merge(est4, on = 'Data_Hora', how = 'outer')


# In[10]:


final.columns


# In[12]:


plt.figure(figsize = (15,15))
plt.plot(final['Precipitacao_0'])
plt.plot(final['Precipitacao_0'])
plt.plot(final['Precipitacao_0'])
plt.plot(final['Precipitacao_0'])
plt.show()

