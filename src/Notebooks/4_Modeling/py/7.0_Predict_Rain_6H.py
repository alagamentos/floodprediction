#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')


# In[ ]:


get_ipython().run_line_magic('autoreload', '')

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import xgboost as xgb
import catboost as cb
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score,                            recall_score, fbeta_score, precision_recall_curve

import numpy as np
import matplotlib.pyplot as plt

from hyperopt import hp
import hyperopt.pyll
from hyperopt.pyll import scope
from hyperopt import STATUS_OK
from hyperopt import fmin, tpe, Trials

from datetime import datetime

import sys
sys.path.append('../../Pipeline')

from ml_utils import plot_confusion_matrix, plot_precision_recall, arg_nearest
from utils import moving_average, reverse_mod


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
#     original_df[c] = (original_df[c+'_0'] + original_df[c+'_1'] +
#                       original_df[c+'_2'] + original_df[c+'_3'] + original_df[c+'_4'])/5 
    original_df[c] = original_df[c+'_0']


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


# # Create 6H-wise DataFrame 

# In[ ]:


hours = 3

unique_values = np.arange(0, len(original_df), 1)
n_group = hours * 4

h_id = np.repeat(unique_values, n_group)
original_df['h_id'] = h_id[:len(original_df)]


# In[ ]:


drop_rows = original_df.groupby('h_id').count().iloc[:,0] < n_group
drop_rows = drop_rows[drop_rows].index.to_list()


# In[ ]:


drop_rows = original_df.groupby('h_id').count().iloc[:,0] < n_group
drop_rows = drop_rows[drop_rows].index.to_list()

for row in drop_rows:
    original_df = original_df[original_df.h_id != row]
    


# In[ ]:


map_ = original_df.loc[::n_group, ['Data_Hora', 'h_id']].set_index('h_id').to_dict()['Data_Hora']


# ## Has Rain

# In[ ]:


precipitacao_sum = original_df.loc[:, ['h_id', 'Precipitacao']].groupby('h_id').sum()


# In[ ]:


y = precipitacao_sum.loc[precipitacao_sum['Precipitacao'] != 0, 'Precipitacao'].values

fig = go.Figure()

fig.add_trace(go.Box(
    name = f'Precipitacao Acc {hours}H',
    y = y,
                    ))
fig.show()


# In[ ]:


has_rain_treshold = 0.01
shift = 2

precipitacao_sum.loc[:, 'Rain_Now'] = (precipitacao_sum['Precipitacao'] > has_rain_treshold ).astype(int)
precipitacao_sum.loc[:, f'Rain_Next_{hours}H'] = precipitacao_sum.loc[:, 'Rain_Now'].shift(-shift)
precipitacao_sum = precipitacao_sum.dropna()

# Remove last index (h_id) from original_df
for i in range(shift):
    last_index = original_df.groupby('h_id').count().iloc[:,0].index[-1]
    original_df = original_df[original_df.h_id != last_index]

precipitacao_sum.head(10)


# In[ ]:


df = precipitacao_sum.loc[:, ['Rain_Now', f'Rain_Next_{hours}H']]


# ## Simple Metrics

# In[ ]:


interest_cols.remove('DirecaoDoVento')

sum_df = original_df[interest_cols + ['h_id']].groupby('h_id').sum()
sum_df.columns = [c + '_sum' for c in sum_df.columns]
sum_df = sum_df.loc[:,'Precipitacao_sum']

# median_df = original_df[interest_cols + ['h_id']].groupby('h_id').median()
# median_df.columns = [c + '_median' for c in median_df.columns]

mean_df = original_df[interest_cols + ['h_id']].groupby('h_id').mean()
mean_df.columns = [c + '_mean' for c in mean_df.columns]

min_df = original_df[interest_cols + ['h_id']].groupby('h_id').min()
min_df.columns = [c + '_min' for c in min_df.columns]

max_df = original_df[interest_cols + ['h_id']].groupby('h_id').max()
max_df.columns = [c + '_max' for c in max_df.columns]


# In[ ]:


# df = pd.concat([df, mean_df], axis = 1)
df = pd.concat([df, sum_df, mean_df, min_df, max_df], axis = 1)
df.index = df.index.map(map_)
df = df.dropna()


# In[ ]:


df.head()


# ## Time Metrics

# In[ ]:



diff_cols = interest_cols.copy()
diff_cols.remove('VelocidadeDoVento')

for col in diff_cols:
    diff = original_df.loc[n_group-1::n_group, col].values -           original_df.loc[0::n_group, col].values
    df[f'{col}_diff'] = diff
    


# In[ ]:


df.head()


# ## Seasonal Metrics

# In[ ]:



# def get_season(Row):
    
#     doy = Row.name.timetuple().tm_yday
    
#     fall_start = datetime.strptime('2020-03-20', '%Y-%m-%d' ).timetuple().tm_yday
#     summer_start = datetime.strptime('2020-06-20', '%Y-%m-%d' ).timetuple().tm_yday
#     spring_start = datetime.strptime('2020-09-22', '%Y-%m-%d' ).timetuple().tm_yday
#     spring_end = datetime.strptime('2020-12-21', '%Y-%m-%d' ).timetuple().tm_yday
    
#     fall = range(fall_start, summer_start)
#     summer = range(summer_start, spring_start)
#     spring = range(spring_start, spring_end)
    
#     if doy in fall:
#         season = 1#'fall'
#     elif doy in summer:
#         season = 2#'winter'
#     elif doy in spring:
#         season = 3#'spring'
#     else:
#         season = 0#'summer' 
    
#     return season

# df_date['season'] =  df_date.apply(get_season, axis = 1)


# In[ ]:


# seasonal_means = ['Precipitacao_mean']#, 'RadiacaoSolar_mean', 'TemperaturaDoAr_mean']

# for s in seasonal_means:
#     map_ = dict(df_date.groupby('season').mean()['Precipitacao_mean'])
#     df_date[f'seasonalMean_{s}'] =  df_date['season'].map(map_)

# df_date = df_date.drop(columns = ['season'])


# In[ ]:


# df_date


# In[ ]:


# df_date.corr()['Rain_Next_Day'].abs().sort_values(ascending = False).to_dict()


# In[ ]:


# plt.bar(df_date.corr()['Rain_Next_Day'].index,df_date.corr()['Rain_Next_Day'].values)


# # Reference Model

# In[ ]:


label = f'Rain_Next_{hours}H'


# In[ ]:


X, y = df.drop(columns = [label]), df[label].astype(int).values

X = X.drop(columns = 'Rain_Now')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_test.shape, X_train.shape


# In[ ]:


clf = xgb.XGBClassifier(tree_method = 'gpu_hist')

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

y_pred = clf.predict(X_test)
plot_confusion_matrix(y_test, y_pred, ['0', '1'])


# In[ ]:


df[label].value_counts()/df.shape[0]


# In[ ]:


f1_score(y_pred, y_test)


# # Feature Selection

# In[ ]:


import shap

# load JS visualization code to notebook
shap.initjs()


# In[ ]:


explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test)
# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:])


# In[ ]:


shap.force_plot(explainer.expected_value, shap_values[217,:], X_test.iloc[217,:])


# In[ ]:


shap.force_plot(explainer.expected_value, shap_values[0:500, :], X_test.iloc[0:500, :])


# In[ ]:


plt.figure(figsize = (8,14))

features_imp = dict(zip(X_train.columns, clf.feature_importances_))
features_imp = {k: v for k, v in sorted(features_imp.items(), key=lambda item: item[1])}

plt.barh(list(features_imp.keys()), features_imp.values())
plt.show()


# In[ ]:



plt.imshow(X_train.corr())
plt.colorbar()
plt.show()

colorscale=[[0.0, "rgb(240, 0, 0)"],
            [0.3, "rgb(240, 240, 239)"],
            [1.0, 'rgb(240, 240, 240)']]

fig = go.Figure()

fig.add_trace(go.Heatmap(z = X_train.corr(),
                         x = X_train.columns,
                         y = X_train.columns, 
                         colorscale = colorscale))
fig.update_layout(width = 700, height = 700)
fig.show()


# In[ ]:


def remove_high_correlation(df, threshold, label_corr):
    
    dataset = df.copy()
    
    remove_columns = []
    
    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (np.abs(corr_matrix.iloc[i, j]) >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname_i = corr_matrix.columns[i] # getting the name of column
                colname_j = corr_matrix.columns[j] # getting the name of column
                if label_corr[colname_i] > label_corr[colname_j]:
                    col_corr.add(colname_j)
                    colname = colname_j
                else:
                    col_corr.add(colname_i)
                    colname = colname_i
                if colname in dataset.columns:
                    remove_columns.append(colname) # deleting the column from the dataset
                    
    return remove_columns


# In[ ]:


label_corr = pd.concat([X_train.reset_index(drop = True), pd.Series(y_train, name = 'label')], axis = 1).corr()['label'].abs().to_dict()

remove_columns = remove_high_correlation(X_train, 0.7, label_corr)
remove_columns += ['Precipitacao_min']


# In[ ]:


X_test_sel = X_test.drop(columns = remove_columns)
X_train_sel = X_train.drop(columns = remove_columns)


# In[ ]:


fig = go.Figure()
fig.add_trace(go.Heatmap( z = X_train_sel.corr(),
                         x = X_train_sel.columns,
                         y = X_train_sel.columns) )
fig.update_layout(template = 'plotly_dark',width = 700, height = 700)
fig.show()


# In[ ]:


clf = xgb.XGBClassifier()#tree_method = 'gpu_hist')

eval_set = [(X_train_sel, y_train), (X_test_sel, y_test)]

clf.fit(X_train_sel, y_train,  eval_metric=["logloss","error", "auc", "map"],
        eval_set=eval_set, verbose=False, early_stopping_rounds=10,);

keys = clf.evals_result()['validation_0'].keys()

fig, ax = plt.subplots( 1, len(keys) ,figsize = (7*len(keys),7))
ax = ax.ravel()
for i, key in enumerate(keys):
    ax[i].set_title(key)
    ax[i].plot(clf.evals_result()['validation_0'][key], lw = 3)
    ax[i].plot(clf.evals_result()['validation_1'][key], lw = 3)
plt.show()

y_pred = clf.predict(X_test_sel)
plot_confusion_matrix(y_test, y_pred, ['0', '1'])


# In[ ]:


plt.figure(figsize = (12,9))

features_imp = dict(zip(X_train_sel.columns, clf.feature_importances_))
features_imp = {k: v for k, v in sorted(features_imp.items(), key=lambda item: item[1])}

plt.barh(list(features_imp.keys()), features_imp.values())
plt.show()


# # Model Optimization

# In[ ]:



param_hyperopt = {
    'max_depth':scope.int(hp.quniform('max_depth', 5, 30, 1)),
    'n_estimators':scope.int(hp.quniform('n_estimators', 5, 1000, 1)),
    'min_child_weight':  scope.int(hp.quniform('min_child_weight', 1, 8, 1)),
    'reg_lambda':hp.uniform('reg_lambda', 0.01, 500.0),
    'reg_alpha':hp.uniform('reg_alpha', 0.01, 500.0),
    'colsample_bytree':hp.uniform('colsample_bytree', 0.3, 1.0),
    'early_stopping_rounds':  scope.int(hp.quniform('early_stopping_rounds', 1, 20, 1)),
                 }

def cost_function(params):
    
    fit_parameters = {}
    fit_parameters['early_stopping_rounds'] = params.pop('early_stopping_rounds')

    clf = xgb.XGBClassifier(**params,
                            objective="binary:logistic",
                            random_state=42)

    clf.fit(X_train_sel, y_train, eval_set = eval_set, eval_metric=["logloss"], verbose = False, **fit_parameters)
    y_pred = clf.predict(X_test_sel)

    return {'loss':-fbeta_score(y_test, y_pred, beta=2),'status': STATUS_OK}

num_eval = 250
eval_set = [(X_train_sel, y_train), (X_test_sel, y_test)]

trials = Trials()
best_param = fmin(cost_function,
                     param_hyperopt,
                     algo=tpe.suggest,
                     max_evals=num_eval,
                     trials=trials,
                     rstate=np.random.RandomState(1))


# In[ ]:


best_param['min_child_weight'] = int(best_param['min_child_weight'])
best_param['n_estimators'] = int(best_param['n_estimators'])
best_param['max_depth'] = int(best_param['max_depth'])
best_param['early_stopping_rounds'] = int(best_param['early_stopping_rounds'])
best_param


# In[ ]:


params = best_param.copy()

fit_parameters = {}
fit_parameters['early_stopping_rounds'] = params.pop('early_stopping_rounds')

clf = xgb.XGBClassifier(**params,
                        objective="binary:logistic",
                        random_state=42)

clf.fit(X_train_sel, y_train, eval_set = eval_set, eval_metric=["logloss"],
        verbose = False,**fit_parameters)
y_pred = clf.predict(X_test_sel)
y_pred_prob = clf.predict_proba(X_test_sel)

plot_confusion_matrix(y_test, y_pred, ['0','1'])
evaluate = (y_test, y_pred)
print('f1_score: ', f1_score(*evaluate))
print('Accuracy: ', accuracy_score(*evaluate))
print('Precision: ', precision_score(*evaluate))
print('Recall: ', recall_score(*evaluate))


# In[ ]:


fig = plot_precision_recall(y_test, y_pred_prob[:,1])
fig.update_layout(template = 'plotly_dark')
fig.show()


# In[ ]:


desired_recall = 0.8

precision, recall, threshold = precision_recall_curve(y_test, y_pred_prob[:,1])
y_pred_threshold = (y_pred_prob[:,1] > threshold[arg_nearest(recall, desired_recall)]).astype(int)

plot_confusion_matrix(y_test, y_pred_threshold, ['0','1'])
evaluate = (y_test, y_pred_threshold)
print('f1_score: ', f1_score(*evaluate))
print('Accuracy: ', accuracy_score(*evaluate))
print('Precision: ', precision_score(*evaluate))
print('Recall: ', recall_score(*evaluate))

