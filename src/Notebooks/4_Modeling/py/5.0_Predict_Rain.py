#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# In[ ]:


original_df = pd.read_csv('../../../data/cleandata/Info pluviometricas/Merged Data/repaired.csv', sep = ';')
original_df['Data_Hora'] = pd.to_datetime(original_df['Data_Hora'])
original_df['Date'] = original_df['Data_Hora'].dt.date


# In[ ]:


interest_cols = list({c.split('_')[0] for c in original_df.columns if '_error' in c})
interest_cols.remove('TemperaturaInterna')
interest_cols.remove('SensacaoTermica')


# # Group Stations - Mean 

# In[ ]:


for c in interest_cols:
    original_df[c] = (original_df[c+'_0'] + original_df[c+'_1'] + original_df[c+'_2'] + original_df[c+'_3'] + original_df[c+'_4'])/5 


# ## Plot data

# In[ ]:


df_plot = original_df[original_df.Data_Hora.dt.year == 2015]

fig = go.Figure(layout=dict(template = 'plotly_dark'))

for col in ['PontoDeOrvalho', 'Precipitacao', 'UmidadeRelativa', 'TemperaturaDoAr']:    
    fig.add_trace(go.Scatter(
        x = df_plot['Data_Hora'],
        y = df_plot[col],
        name = col,
                            )
                 )
fig.show()


# # Feature Engineering

# In[ ]:


interest_cols += ['Diff_Temp_POrvalho']
original_df['Diff_Temp_POrvalho'] = original_df['TemperaturaDoAr'] -  original_df['PontoDeOrvalho']


# ## Has Rain

# In[ ]:



has_rain_treshold = 10
precipitacao_sum = original_df.loc[:, ['Date', 'Precipitacao']].groupby('Date').sum()
precipitacao_sum.loc[:, 'Rain_Today'] = precipitacao_sum['Precipitacao'] > has_rain_treshold
precipitacao_sum.loc[:, 'Rain_Next_Day'] = precipitacao_sum.loc[:, 'Rain_Today'].shift(-1)
precipitacao_sum = precipitacao_sum.dropna()

precipitacao_sum.index = pd.to_datetime(precipitacao_sum.index, yearfirst=True)
precipitacao_sum.head()


# # Create Datewise DataFrame 

# In[ ]:


df = original_df[interest_cols + ['Date' , 'Data_Hora'] ]
df = df.set_index('Data_Hora')


# In[ ]:


unique_dates = df.index.round('D').unique()
df_date = pd.DataFrame(precipitacao_sum.index, columns = ['Date'])


# In[ ]:


df_date = df_date.merge(precipitacao_sum.loc[:, ['Rain_Today','Rain_Next_Day']], on = 'Date')
df_date = df_date.set_index('Date')


# ## Simple Metrics

# In[ ]:



sum_date = df[interest_cols + ['Date']].groupby('Date').sum()
sum_date.columns = [c + '_sum' for c in sum_date.columns]

median_date = df[interest_cols + ['Date']].groupby('Date').median()
median_date.columns = [c + '_median' for c in median_date.columns]

min_date = df[interest_cols + ['Date']].groupby('Date').min()
min_date.columns = [c + '_min' for c in min_date.columns]

max_date = df[interest_cols + ['Date']].groupby('Date').max()
max_date.columns = [c + '_max' for c in max_date.columns]


# In[ ]:


df_date = pd.concat([df_date, sum_date, median_date, min_date, max_date], axis = 1)
df_date.head(2)


# ## Time Metrics

# In[ ]:


hours = [3, 9, 15, 21 ]
for selected_hour in hours:

    selected_df = df.loc[(df.index.hour == selected_hour ) & (df.index.minute == 0 ), interest_cols ]
    selected_df.index = selected_df.index.round('D')
    selected_df.columns = [f'{c}_{selected_hour}H' for c in selected_df.columns]
    df_date = pd.concat([df_date, selected_df], axis = 1)


df_date = df_date.dropna(axis = 0)


# In[ ]:


df_date['Rain_Next_Day'] = df_date['Rain_Next_Day'].astype(int)
df_date['Rain_Today'] = df_date['Rain_Today'].astype(int)


# In[ ]:


df_date.head()


# # Reference Model

# In[ ]:


import xgboost as xgb
import catboost as cb
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import sys
sys.path.append('../../Pipeline')

from ml_utils import *


# In[ ]:


X, y = df_date.drop(columns = ['Rain_Next_Day']), df_date.Rain_Next_Day.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[ ]:


clf = xgb.XGBClassifier()#tree_method = 'gpu_hist')

eval_set = [(X_train, y_train), (X_test, y_test)]

clf.fit(X_train, y_train,  eval_metric=["logloss","error", "auc", "map"], eval_set=eval_set, verbose=False);

keys = clf.evals_result()['validation_0'].keys()

fig, ax = plt.subplots( 1, len(keys) ,figsize = (7*len(keys),7))
ax = ax.ravel()
for i, key in enumerate(keys):
    ax[i].set_title(key)
    ax[i].plot(clf.evals_result()['validation_0'][key], lw = 3)
    ax[i].plot(clf.evals_result()['validation_1'][key], lw = 3)
plt.show()


# In[ ]:


X_test.shape, X_train.shape


# In[ ]:


y_pred = clf.predict(X_test)
plot_confusion_matrix(y_pred, y_test, ['0', '1'])


# In[ ]:


f1_score(y_pred, y_test)


# In[ ]:


plt.figure(figsize = (15,30))

features_imp = dict(zip(X_train.columns, clf.feature_importances_))
features_imp = {k: v for k, v in sorted(features_imp.items(), key=lambda item: item[1])}

plt.barh(list(features_imp.keys()), features_imp.values())
plt.show()

