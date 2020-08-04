#!/usr/bin/env python
# coding: utf-8

# Esse *notebook* é a primeira tentative de recontruir os dados com as regiões de erro já identificadas,

# In[1]:


import sys
sys.path.insert(1, '../../Pipeline')

import imp
import utils
imp.reload(utils)
from utils import *


# In[2]:


# Python imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Pandas Config
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

# Notebook Theme
# try:
#     from jupyterthemes import jtplot
#     jtplot.style()
# except:
#     pass

# ML imports
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import xgboost

ip = pd.read_csv('../../../data/cleandata/Info pluviometricas/Merged Data/merged_wRegions.csv',
                 sep = ';',
                 dtype = {'Local_0': object, 'Local_1':object,
                          'Local_2':object,  'Local_3':object})

print(ip.columns)
ip.head()


# ## Validate Regions

# In[3]:


select = 'RadiacaoSolar_4'
start, stop = 150000, 150500

## ===================================
error = ip[select + '_error'].to_list()
error_reg = list_2_regions(error)
plot_regions(ip[select].fillna(0), error_reg, start, stop, plt_type = 'lines')

select = 'RadiacaoSolar_2'

## ===================================
error = ip[select + '_error'].to_list()
error_reg = list_2_regions(error)
plot_regions(ip[select].fillna(0), error_reg, start, stop, plt_type = 'lines')


# ## Interpolation

# In[4]:


cols_rad = [i for i in ip.columns if 'RadiacaoSolar' in i]
rad = ip[cols_rad].copy(deep = True)
ip[cols_rad].head()


# In[5]:


print(ip['RadiacaoSolar_0_error'].sum())
print(ip['RadiacaoSolar_1_error'].sum())
print(ip['RadiacaoSolar_2_error'].sum())
print(ip['RadiacaoSolar_3_error'].sum())
print(ip['RadiacaoSolar_4_error'].sum())


# In[6]:


error = ip['RadiacaoSolar_0_error']
error_reg = list_2_regions(error)


# In[7]:


reg_size = [i[1] - i[0] for i in error_reg]
max_size = 5

interpol_reg = [error_reg[i] for i in range(len(reg_size)) if reg_size[i] <= max_size]
ip['RadiacaoSolar_0_interpol']  = regions_2_list(interpol_reg, len(ip))
ip.loc[ip['RadiacaoSolar_0_interpol'],'RadiacaoSolar_0_error'] = False
#ip['RadiacaoSolar_0_error'] = ip['RadiacaoSolar_0_error'] & ~ip['RadiacaoSolar_0_interpol']


# In[8]:


s = ip['RadiacaoSolar_0']
s = s.fillna(-1)
s[ip['RadiacaoSolar_0_interpol']] = np.nan
s = s.interpolate(method = 'linear', limit = max_size)
s[s == -1] = np.nan
ip['RadiacaoSolar_0'] = s


# In[9]:


start, stop = 150000, 150500
plt.figure(figsize = (12,7))
plt.plot(s[start:stop], lw = 2)
plt.plot(s[start:stop].where(ip['RadiacaoSolar_0_interpol'][start:stop]), c = 'red', lw =2)
plt.plot(s[start:stop].where(ip['RadiacaoSolar_0_error'][start:stop]), c = 'black', lw = 2)
plt.show()


# ## Regressor RadiacaoSolar_0

# In[10]:


label = 'RadiacaoSolar_0'
features = rad.drop(columns=label).columns.to_list()


# ## History pivot - *shift*

# - https://support.ptc.com/help/thingworx_hc/thingworx_analytics_8/index.html#page/thingworx_analytics_8/twxa-time-series.html
# 
# > Time series data also differs somewhat from non-time series data in the training of predictive models. A lookbackSize parameter is required for training time series models. The **lookbackSize** defines the number of recent data points to be used when predicting each future value in the time series. Any value greater than 1 is acceptable but generally, a power of 2 is used (2, 4, 8, 16). Larger values affect performance because more records are used for predictions. When a value of 0 is specified, auto-windowing will take place and ThingWorx Analytics will try a set of lookback sizes (2, 4, 8, and 16) in order to select the size that produces the most accurate results.
# >
# > Training a time series predictive model also requires a **lookahead** parameter that indicates the number of time steps ahead to predict. In most cases, the lookahead defaults to 1 (it defaults to 0 if goal history is not in use). A lookahead of 1 means the model can be used to predict one time step ahead. To predict outcomes further ahead, enter any value greater than 1.
# >
# > Because basic machine learning algorithms are not time-aware, ThingWorx Analytics uses **history pivoting** to transform time series data into non-time series data that can be trained using the same basic algorithms as non-time series data. During this transformation, the data is grouped and sorted, by entity and time, and any necessary interpolations take place to produce Analytics-ready data. The table below shows the history-pivoted data from both sets of time series predictions shown above.
# 
# > Aplication Example:
# >
# > Creating a Virtual Sensor – A very expensive sensor is added to a pump to measure its efficiency in a controlled environment. The sensor captures running conditions on the pump and captures readings for the pump's efficiency. That pump also has several inexpensive sensors that are also collecting data. A model can be trained to emulate this expensive sensor, as a virtual sensor, by predicting its value from the inexpensive sensor values. From there, many pumps could be deployed with only the inexpensive sensors. By using the model created on the first pump, each pump could have a virtual version of the expensive sensor without the cost of it being deployed with each pump

# Shift do Label (RadiacaoSolar_0) e também do erro do label.
# 
# O shift do erro do label permite eliminar as amostras com o valor atrasado errado após o shift. A eliminação dos erros a posteriori garante que a amostra apenas terá dados contínuos.
# 
# Para o treinamento é importante utilizar apenas dados considerados corretos. Isso é necessário não somente para o _Label_ mas também para os _Features_ de treinamento - dados das outras estações da amostra atual e os dados atrasados do label. 
# 
# > Shift = atrasar valores por uma amostra

# In[11]:


horas = 24
print('Amostras necessárias:', horas*60/ 15)


# In[12]:


rad_d = rad.copy(deep = True)
lookbackSize = 30

for i in range(1,lookbackSize + 1):
    rad_d['delay_' + str(i)] = ip['RadiacaoSolar_0'].shift(i)
for i in range(1,lookbackSize + 1):
    rad_d['delay_error_' + str(i)] = ip['RadiacaoSolar_0_error'].shift(i)
    
cols_delay = [i for i in rad_d.columns if 'delay' in i]
print('Dataframe completo: ', len(rad_d), 'amostras')
rad_d.head(5)


# ## Eliminar dados errados

# In[13]:


error_cols_all_features = [i for i in rad_d.columns if 'error' in i ]
i = 0 
for col in error_cols_all_features:
    if i == 0:
        has_error_all_features = rad_d[col]
        i+=1
    else:
        has_error_all_features = has_error_all_features | rad_d[col].fillna(value = True)
        
rad_e_all_features = rad_d[has_error_all_features == False].drop(columns = error_cols_all_features)

print('Dados com erro: ', has_error_all_features.sum()/len(rad_d)*100, '%')
print('Dados com erro :', has_error_all_features.sum(), 'amostras')
print('Dados corretos:', len(rad_e_all_features), 'amostras')


# Os dados com erro representam 89.9% de todo o dataset, isso é, considerando o erro das outras estações e o erro do label (**RadiacaoSolar_0**) para as colunas de atraso (*delay*).

# In[14]:


error_cols = [i for i in rad_d.columns if 'delay_error' in i or i == 'RadiacaoSolar_0_error']
i = 0 
for col in error_cols:
    if i == 0:
        has_error = rad_d[col]
        i+=1
    else:
        has_error = has_error | rad_d[col].fillna(value = True)
        
rad_e = rad_d[has_error == False].drop(columns = error_cols)

print('Dados com erro: ', has_error.sum()/len(rad_d)*100, '%')
print('Dados com erro :', has_error.sum(), 'amostras')
print('Dados corretos:', len(rad_e), 'amostras')


# Quando não consideramos os dados das outras estações com erro, os dados errados caiem de 89,9% para 22.9%. Portanto a seguir utilizaremos a segunda opção e para lidar com os dados errados das outras estações será incluido no treinamento os features de erro *'RadiacaoSolar_x_error'* .

# In[15]:


# Achar Continuidade

rad_0 = rad_e[label].dropna()
regions, sizes = [],[]
init = rad_0.index[0]
for i in range(1,len(rad_0.index ) -1):
    if rad_0.index[i] + 1 != rad_0.index[i+1] :
        final = rad_0.index[i]
        regions.append([init, final])
        sizes.append(final-init)
        init = rad_0.index[i + 1]
print(len(regions))


# In[16]:


rad_e.columns
plot_cols = [i for i in rad_e.columns if 'delay' not in i and 'error' not in i]
print(plot_cols)


# In[17]:


# figsize = (12,7 * len(plot_cols))
# fig, ax = plt.subplots(len(plot_cols), 1,figsize = figsize)
# i = 0
# for col in ['RadiacaoSolar_0']:
#     for reg in regions:
#         start, stop = reg
#         ax[i].plot(rad_e[[col]].loc[start:stop].index, rad_e[[col]].loc[start:stop])
#         ax[i].set_title(col)
#     i+=1
# plt.show()


# In[18]:


start, stop = 0, 150500

plt.figure(figsize= (12,12))
plt.plot(ip['RadiacaoSolar_0'][start:stop].fillna(0))
plt.plot(ip['RadiacaoSolar_0'][start:stop].where(ip['RadiacaoSolar_0_error'][start:stop]),
          c = 'red')
plt.show()


# ## TimeSeries Split - Cross Validation

# Documentation:
# - https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
# 
# Example
# - https://medium.com/keita-starts-data-science/time-series-split-with-scikit-learn-74f5be38489e
# 
# Why use K-1 Cross Validation instead of K_FOLD CV?
# 
# >Time-series (or other intrinsically ordered data) can be problematic for cross-validation. If some pattern emerges in year 3 and stays for years 4-6, then your model can pick up on it, even though it wasn't part of years 1 & 2.
# >An approach that's sometimes more principled for time series is forward chaining, where your procedure would be something like this:
# >fold 1 : training [1], test [2]
# >fold 2 : training [1 2], test [3]
# >fold 3 : training [1 2 3], test [4]
# >fold 4 : training [1 2 3 4], test [5]
# >fold 5 : training [1 2 3 4 5], test [6]
# >That more accurately models the situation you'll see at prediction time, where you'll model on past data and predict on forward-looking data. It also will give you a sense of the dependence of your modeling on data size.
# 
# - https://stackoverflow.com/questions/26991997/multiple-line-quote-in-markdown?rq=1
# 
# Why every statistician should know about cross-validation
# - https://robjhyndman.com/hyndsight/crossvalidation/
# 
# > Cross-validation is primarily a way of measuring the predictive performance of a statistical model. Every statistician knows that the model fit statistics are not a good guide to how well a model will predict: high 
# $R^2$ does not necessarily mean a good model. It is easy to over-fit the data by including too many degrees of freedom and so inflate $R^2$ and other fit statistics.
# 
# > Minimizing a CV statistic is a useful way to do model selection such as choosing variables in a regression or choosing the degrees of freedom of a nonparametric smoother. It is certainly far better than procedures based on statistical tests and provides a nearly unbiased measure of the true MSE on new observations.
# 
# > **Cross-validation for time series**
# >
# >When the data are not independent cross-validation becomes more difficult as leaving out an observation does not remove all the associated information due to the correlations with other observations. For time series forecasting, a cross-validation statistic is obtained as follows
# >
# >    1. Fit the model to the data $y_1, ..., y_t$ and let $ŷ_{t+1}$ denote the forecast of the next observation. Then compute the error $e^*_{t+1} = y_{t+1} - ŷ_{t+1}$ for the forecast observation.
# >
# >    2. Repeat step 1 for $t = m,...,n-1$ where $m$ is the minimum number of observations needed for fitting the model.
# >
# >    3. Compute the MSE from $e^*_{m+1}, ..., e^*_n$.
# 
# ###### *Os dados estão organizados de forma que uma amostra k não depende de outra amostra k+n; k-1 CV é realmente necessário?*
# 
# Outros links:
# - https://www.kaggle.com/c/ieee-fraud-detection/discussion/103065
# - https://www.kaggle.com/kashnitsky/correct-time-aware-cross-validation-scheme

# In[19]:


n_splits = 3

features = list(rad_e.drop(columns=[label]).columns)

label_i = rad_e.columns.get_loc(label)
features_i = [rad_e.columns.get_loc(c) for c in features if c in rad]

X_train, X_test = [], []
y_train, y_test = [], []

tscv = TimeSeriesSplit(n_splits = n_splits)

for train_index, test_index in tscv.split(rad_e[[label]].dropna()):
    y_train.append(rad_e[label].iloc[train_index].values)
    y_test.append(rad_e[label].iloc[test_index].values)
    
    X_train.append(rad_e[features].iloc[train_index])
    X_test.append(rad_e[features].iloc[test_index])


# In[20]:


# Check CV regions
fig, ax = plt.subplots(n_splits, 1, figsize = (12,7*n_splits), sharex=True)
for n in range(n_splits):
    #ax[n].plot(rad_e[label], c='Green', alpha = 0.2)
    ax[n].plot(X_train[n].index, y_train[n], c='Blue')
    ax[n].plot(X_test[n].index,  y_test[n],c='Red') 


# ## Create model

# In[21]:


xgb = []
y_predict = []

for n in range(n_splits):
    xgb.append(xgboost.XGBRegressor(objective='reg:squarederror'))
    xgb[n].fit(X_train[n], y_train[n])
    
    y_predict.append(xgb[n].predict(X_test[n]))


# ##  Score

# In[22]:


# ================ #
# Calculate Score  #
# ================ #


# ## Predict tests samples

# In[23]:


start, stop = 0, 1000

fig, ax = plt.subplots(n_splits, 1, figsize = (12,7*n_splits))

for n in range(n_splits):
    aux = X_test[n].index[0]
    len_ = stop - start
    x = np.linspace(aux,aux+len_ + 1,len_)
    d_start,d_stop = aux+start, aux+stop
    ax[n].plot(x, y_test[n][start:stop], c = 'blue', label = 'Original')
    ax[n].plot(x, y_predict[n][start:stop],'--', c = 'red', label = 'Predicted')
    ax[n].legend()    

plt.show()


# ## Predict on faulty data

# In[24]:


drop_cols = [i for i in rad_d.columns if 'delay_error' in i]
drop_cols.append(label)
drop_cols.append(label+'_error')


# In[25]:


val_predict = []
for n in range(n_splits):    
    val_predict.append(xgb[n].predict(rad_d.drop(columns = drop_cols)))


# In[26]:


start, stop = 0, 1000

fig, ax = plt.subplots(n_splits, 1, figsize = (12,7*n_splits))

for n in range(n_splits):
    ax[n].plot(rad_d[start:stop].index, rad_d[label][start:stop], c = 'blue', label = 'Original')
    ax[n].plot(rad_d[start:stop].index, val_predict[n][start:stop],'--', c = 'red', label = 'Predicted')
    ax[n].legend()
plt.show()


# In[27]:


start, stop = 337750, 338200

fig, ax = plt.subplots(n_splits, 1, figsize = (12,7*n_splits))

for n in range(n_splits):
    ax[n].plot(rad_d[start:stop].index, rad_d[label][start:stop], c = 'blue', label = 'Original')
    ax[n].plot(rad_d[start:stop].index, val_predict[n][start:stop],'--', c = 'red', label = 'Predicted')
    ax[n].legend()
plt.show()


# **Como que samples iguais (e.g. 345000 - 350000) produzem saídas diferentes?**

# Essa previsão é feita *de uma vez só*. Proximo passo é testar com a previsão recorrente, isto é, prever amostra k, atualoziar os features de k+1, prever k+1.

# ## Recurrent prediction

# In[28]:


rad_d.head(2)


# In[29]:


rad_d[start+30:stop].head(10)


# In[30]:


recurrent_y = np.zeros(len(rad_d), dtype=np.float64)
rad_recurrent = rad_d.drop(columns = drop_cols).copy(deep = True)

start, stop = 337750, 338500
for i in range(start, stop, 1):
    x = rad_recurrent.loc[[i]]
    pred_ = xgb[2].predict(x)
    recurrent_y[i] = pred_
    for l in range(1, lookbackSize+1):
        rad_recurrent.loc[i + l, 'delay_' + str(l) ] = pred_


# In[31]:


plt.figure(figsize = (12,7))
plt.plot(rad_d['RadiacaoSolar_0'][start:stop], label =  'Original')
plt.plot(rad_d['RadiacaoSolar_0'][start:stop].index,
         recurrent_y[start:stop],'--', label = 'Previsao')
plt.legend()
plt.show()


# In[32]:


len(recurrent_y) * lookbackSize


# Melhorou bastante!
# 
# - Problema:
#     - Previsão muito lenta para todo o dataset n (487977 amostras)
#     - Atualizar m vezes o delay ( lookbackSize = 30)
#     - O(n*m) =  14639310 atualizações
# 
# 
# 
# - Solução:
#     - Prever apenas para os dados considerados falhos
# 
#     
# - Próximos passos:
#     - Previsão automática nos dados falhos
#         - Identificar regiões de dados falhos e prever somente nessas regiões de forma recorrente
#         - Para os dados redundantes (5 estações mesmo dados) - Achar a estação com menor número de dados falhos, corrigir essa estação, proxíma estação.     
#     - Incluir **Hora** e mês(?) no treinamento 
#     - Otimizar hiper-parametros: **Hyperopt**
#     - Incluir Rede LSTM com XGBoost 
# > https://towardsdatascience.com/power-of-xgboost-lstm-in-forecasting-natural-gas-price-f426fada80f0
