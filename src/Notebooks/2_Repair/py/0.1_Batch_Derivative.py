#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
sys.path.insert(1, '../../Pipeline')

import imp
import utils
imp.reload(utils)
from utils import *


# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

try:
    from jupyterthemes import jtplot
    jtplot.style()
except:
    pass

ip = pd.read_csv('../../../data/cleandata/Info pluviometricas/Merged Data/merged.csv',
                 sep = ';',
                 dtype = {'Local_0': object, 'Local_1':object,
                          'Local_2':object,  'Local_3':object})

print(ip.columns)
ip.head()


# In[ ]:


ip['Data_Hora'] = pd.to_datetime(ip['Data_Hora'])
ip = ip.sort_values(by = 'Data_Hora')
ip.tail()


# In[ ]:


"""
UmidadeRelativa
PressaoAtmosferica
TemperaturaDoAr
TemperaturaInterna
PontoDeOrvalho
SensacaoTermica
RadiacaoSolar
DirecaoDoVento
Precipitacao
""";


# In[ ]:


"""VelocidadeDoVento"""


# #### Umidade Relativa 

# In[ ]:


cols_um = [i for i in ip.columns if 'UmidadeRelativa' in i]
um = ip[cols_um].fillna(np.nan)
start, stop = 87234, 87234 + 500
for col in um.columns:
    peaks = derivative_threshold(um[col], 12, False, start, stop, lw = 2, figsize = (11, 15))
    zeros = derivative_zero(um[col], 3, False, False, start, stop, lw = 2, figsize = (11, 15))
    nans = um[col].isna()
    error = [zeros[i] or peaks[i] or nans[i] for i in range(len(peaks))]
    error_reg = list_2_regions(error)
    error_reg = increase_margins(3, error_reg, len(peaks))
    error = regions_2_list(error_reg, len(peaks))
    plot_regions(um[col].fillna(0), error_reg, start, stop, plt_type = 'lines', 
                 title = 'Umidade Relativa', figsize = (11, 8), lw = 3)


# #### PressaoAtmosferica

# In[ ]:


cols_um = [i for i in ip.columns if 'PressaoAtmosferica' in i]
um = ip[cols_um].fillna(np.nan)
um.head()

for col in um.columns:
    peaks = derivative_threshold(um[col], 50, False, 1000, 1500, ylim = [900, 1000])
    zeros = derivative_zero(um[col], 7, False, False, 1000, 1500, ylim = [900, 1000])
    nans = um[col].isna()
    error = [zeros[i] or peaks[i] or nans[i] for i in range(len(peaks))]
    error_reg = list_2_regions(error)
    error_reg = increase_margins(5, error_reg, len(peaks))
    plot_regions(um[col], error_reg, 11100, 11500, plt_type = 'lines', ylim=[900, 950])


# #### TemperaturaDoAr

# In[ ]:


cols_um = [i for i in ip.columns if 'TemperaturaDoAr' in i]
um = ip[cols_um].fillna(np.nan)
um.head()

for col in um.columns:
    peaks = derivative_threshold(um[col], 6, False, 4000, 4500)
    zeros = derivative_zero(um[col], 4,False, False, 1000, 1500)
    nans = um[col].isna()
    error = [zeros[i] or peaks[i] or nans[i] for i in range(len(peaks))]
    error_reg = list_2_regions(error)
    error_reg = increase_margins(3, error_reg, len(peaks))
    plot_regions(um[col], error_reg, 1000, 1500, plt_type = 'lines',
                 title = 'TemperaturaDoAr', figsize = (11, 8), lw = 3)


# #### TemperaturaInterna

# In[ ]:


cols_um = [i for i in ip.columns if 'TemperaturaInterna' in i]
um = ip[cols_um].fillna(np.nan)
um.head()

for col in um.columns:
    peaks = derivative_threshold(um[col], 6, False, 4000, 4500)
    zeros = derivative_zero(um[col], 4, False, False, 1000, 1500)
    nans = um[col].isna()
    error = [zeros[i] or peaks[i] or nans[i] for i in range(len(peaks))]
    error_reg = list_2_regions(error)
    error_reg = increase_margins(3, error_reg, len(peaks))
    plot_regions(um[col], error_reg, 1000, 1500, plt_type = 'lines')


# #### PontoDeOrvalho

# In[ ]:


cols_um = [i for i in ip.columns if 'PontoDeOrvalho' in i]
um = ip[cols_um].fillna(np.nan)
um.head()

for col in um.columns:
    peaks = derivative_threshold(um[col], 3.5, False, 4000, 4500)
    zeros = derivative_zero(um[col], 4, False, False, 1000, 1500)
    nans = um[col].isna()
    error = [zeros[i] or peaks[i] or nans[i] for i in range(len(peaks))]
    error_reg = list_2_regions(error)
    error_reg = increase_margins(3, error_reg, len(peaks))
    plot_regions(um[col], error_reg, 1000, 1500, plt_type = 'lines', ylim=[0, 20], title = 'Ponto de Orvalho')


# #### SensacaoTermica

# In[ ]:


cols_um = [i for i in ip.columns if 'SensacaoTermica' in i]
um = ip[cols_um].fillna(np.nan)
um.head()
start, stop = 200000, 200500
for col in um.columns:
    peaks = derivative_threshold(um[col].fillna(0), 4, False, start, stop)
    zeros = derivative_zero(um[col].fillna(0), 10, False, False, start, stop)
    nans = um[col].isna()
    error = [zeros[i] or peaks[i] or nans[i] for i in range(len(peaks))]
    error_reg = list_2_regions(error)
    error_reg = increase_margins(3, error_reg, len(peaks))
    plot_regions(um[col].fillna(0), error_reg, start, stop, plt_type = 'lines')


# #### RadiacaoSolar

# In[ ]:


cols_um = [i for i in ip.columns if 'RadiacaoSolar' in i]
um = ip[cols_um].fillna(np.nan)
um.head()
start, stop = 0, 1000
for col in um.columns:
    peaks = derivative_threshold(um[col].fillna(0), 750, False, start, stop)
    zeros = derivative_zero(um[col].fillna(0), 50, False, plot = False, plt_start = start, plt_stop = stop)
    const_not_null = derivative_zero(um[col].fillna(0), 3, True, False, start, stop)
    nans = um[col].isna()
    error = [zeros[i] or const_not_null[i] or peaks[i] for i in range(len(zeros))]
    # error = nans
    error_reg = list_2_regions(error)
    error_reg = increase_margins(5, error_reg, len(zeros))
    error = regions_2_list(error_reg, len(peaks))
    error = [nans[i] or error[i] for i in range(len(error))]
    error_reg = list_2_regions(error)
    plot_regions(um[col].fillna(0), error_reg, start, stop, plt_type = 'lines')


# #### DirecaoDoVento

# In[ ]:


cols_um = [i for i in ip.columns if 'DirecaoDoVento' in i]
um = ip[cols_um]
um.head()
start, stop = 0, 0 + 500
for col in um.columns[:-1]:
    #peaks = derivative_threshold(um[col].fillna(0), 120, True, start, stop)
    zeros = derivative_zero(um[col].fillna(0), 3, False, start, stop)
    nans = um[col].isna()
    error = [zeros[i] or nans[i] for i in range(len(zeros))]
    error_reg = list_2_regions(error)
    error_reg = increase_margins(3, error_reg, len(zeros))
    plot_regions(um[col], error_reg, start, stop, plt_type = 'lines')


# #### Velocidade do Vento

# In[ ]:


cols_um = [i for i in ip.columns if 'VelocidadeDoVento' in i]
um = ip[cols_um].fillna(np.nan)
um.head()
start, stop = 0, 1000
for col in um.columns:
    peaks = derivative_threshold(um[col], 8, False, start, stop)
    zeros = derivative_zero(um[col].fillna(0), 5, False, start, stop)
    nans = um[col].isna()
    error = [zeros[i] or peaks[i] or nans[i] for i in range(len(peaks))]
    error_reg = list_2_regions(error)
    error_reg = increase_margins(3, error_reg, len(peaks))
    plot_regions(um[col].fillna(0), error_reg, start, stop, plt_type = 'lines')


# #### Precipitacao

# In[ ]:


cols_um = [i for i in ip.columns if 'Precipitacao' in i]
um = ip[cols_um].fillna(np.nan)
um.head()
start, stop = 15000, 20000
max_ = 0
for col in um.columns[:-1]:
    print(um[col].nlargest(5))

    #zeros = derivative_zero(um[col].fillna(0), 10, False, start, stop)
    #nans = um[col].isna()
    #error = [zeros[i] or peaks[i] or nans[i] for i in range(len(peaks))]
    #error_reg = list_2_regions(error)
    #error_reg = increase_margins(3, error_reg, len(peaks))
    #plot_regions(um[col].fillna(0), error_reg, len(um[col]), start, stop, plt_type = 'lines')

