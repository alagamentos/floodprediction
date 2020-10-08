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

from hyperopt import hp
import hyperopt.pyll
from hyperopt.pyll import scope
from hyperopt import STATUS_OK
from hyperopt import fmin, tpe, Trials

from datetime import datetime

import sys
sys.path.append('../../Pipeline')

from ml_utils import *
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
    original_df[c] = (original_df[c+'_0'] + original_df[c+'_1'] +
                      original_df[c+'_2'] + original_df[c+'_3'] + original_df[c+'_4'])/5 


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

mean_date = df[interest_cols + ['Date']].groupby('Date').mean()
mean_date.columns = [c + '_mean' for c in mean_date.columns]

min_date = df[interest_cols + ['Date']].groupby('Date').min()
min_date.columns = [c + '_min' for c in min_date.columns]

max_date = df[interest_cols + ['Date']].groupby('Date').max()
max_date.columns = [c + '_max' for c in max_date.columns]


# In[ ]:


df_date = pd.concat([df_date, sum_date, mean_date, median_date, min_date, max_date], axis = 1)
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


# ## Seasonal Metrics

# In[ ]:



def get_season(Row):
    
    doy = Row.name.timetuple().tm_yday
    
    fall_start = datetime.strptime('2020-03-20', '%Y-%m-%d' ).timetuple().tm_yday
    summer_start = datetime.strptime('2020-06-20', '%Y-%m-%d' ).timetuple().tm_yday
    spring_start = datetime.strptime('2020-09-22', '%Y-%m-%d' ).timetuple().tm_yday
    spring_end = datetime.strptime('2020-12-21', '%Y-%m-%d' ).timetuple().tm_yday
    
    fall = range(fall_start, summer_start)
    summer = range(summer_start, spring_start)
    spring = range(spring_start, spring_end)
    
    if doy in fall:
        season = 1#'fall'
    elif doy in summer:
        season = 2#'winter'
    elif doy in spring:
        season = 3#'spring'
    else:
        season = 0#'summer' 
    
    return season

df_date['season'] =  df_date.apply(get_season, axis = 1)


# In[ ]:


seasonal_means = ['Precipitacao_mean']#, 'RadiacaoSolar_mean', 'TemperaturaDoAr_mean']

for s in seasonal_means:
    map_ = dict(df_date.groupby('season').mean()['Precipitacao_mean'])
    df_date[f'seasonalMean_{s}'] =  df_date['season'].map(map_)

df_date = df_date.drop(columns = ['season'])


# In[ ]:


df_date


# In[ ]:


df_date.corr()['Rain_Next_Day'].abs().sort_values(ascending = False).to_dict()


# In[ ]:


plt.bar(df_date.corr()['Rain_Next_Day'].index,df_date.corr()['Rain_Next_Day'].values)


# # Reference Model

# In[ ]:


X, y = df_date.drop(columns = ['Rain_Next_Day']), df_date.Rain_Next_Day.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_test.shape, X_train.shape


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

y_pred = clf.predict(X_test)
plot_confusion_matrix(y_test, y_pred, ['0', '1'])


# In[ ]:


df_date.Rain_Next_Day.value_counts()/df_date.shape[0]


# In[ ]:


f1_score(y_pred, y_test)


# # Feature Selection

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

fig.add_trace(go.Heatmap( z = X_train.corr(),
                         x = X_train.columns,
                         y = X_train.columns, 
                         colorscale = colorscale))
fig.update_layout(width = 700, height = 700)
fig.show()


# In[ ]:


def remove_high_correlation(df, threshold):
    
    dataset = df.copy()
    
    remove_columns = []
    
    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (np.abs(corr_matrix.iloc[i, j]) >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
                if colname in dataset.columns:
                    remove_columns.append(colname) # deleting the column from the dataset
                    
                    
    return remove_columns


# In[ ]:


remove_columns = remove_high_correlation(X_train, 0.7)
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

    clf.fit(X_train_sel, y_train, eval_set = eval_set, eval_metric=["logloss"], verbose = False,**fit_parameters)
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


# # Include Wind Direction

# In[ ]:


errors = pd.read_csv('../../../data/cleandata/Info pluviometricas/Merged Data/error_regions.csv', sep = ';')
errors[[c for c in errors.columns if 'Direcao' in c]].sum()


# In[ ]:


selected_wind = 2


# In[ ]:


rv_ = reverse_mod(original_df[f'DirecaoDoVento_{selected_wind}'].values)
v_ = moving_average(rv_, window = 15)
v_ = np.mod(v_, 360)


# In[ ]:


fig = go.Figure(layout=dict(template = 'plotly_dark'))

fig.add_trace(go.Scatter(
x= list(range(len(rv_))),
y = rv_))

fig.show()


# In[ ]:


fig = go.Figure(layout=dict(template = 'plotly_dark'))

fig.add_trace(go.Scatter(
    x = original_df['Data_Hora'],
    y = original_df[f'DirecaoDoVento_{selected_wind}'],
    name = f'DirecaoDoVento_{selected_wind}',
    line = dict(color = 'gray')),
             )

fig.add_trace(go.Scatter(
    x = original_df['Data_Hora'],
    y = v_,
    name = 'Avg'),
             )
    
fig.show()


# In[ ]:


wind_direction = np.empty(v_.shape, dtype = np.int)
wind_direction[(v_ > 315 ) | (v_ <=  45 )] = 0 # North
wind_direction[(v_ >  45 ) & (v_ <= 135 )] = 1 # East
wind_direction[(v_ > 135 ) & (v_ <= 225 )] = 2 # South
wind_direction[(v_ > 225 ) & (v_ <= 315 )] = 3 # West

original_df['DirecaoDoVento_cat'] = wind_direction
original_df['Date'] = pd.to_datetime(original_df['Date'], yearfirst=True)

# Mode
wind_direction_mode = original_df.loc[ original_df['Data_Hora'].dt.hour > 20, ['DirecaoDoVento_cat','Date']].groupby('Date')                        .apply(pd.DataFrame.mode).set_index('Date', drop = True)


# In[ ]:


X = X.merge(wind_direction_mode, right_index=True, left_index=True)
dummies = pd.get_dummies(X['DirecaoDoVento_cat'], drop_first=True)
dummies.columns = [f'DirecaoDoVento_{i}' for i in range(len(dummies.columns))]
X = pd.concat([X, dummies], axis = 1 ).drop(columns = ['DirecaoDoVento_cat'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

remove_columns = remove_high_correlation(X_train, 0.7)
remove_columns += ['Precipitacao_min']
X_test_sel = X_test.drop(columns = remove_columns)
X_train_sel = X_train.drop(columns = remove_columns)


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

    clf.fit(X_train_sel, y_train, eval_set = eval_set, eval_metric=["logloss"], verbose = False,**fit_parameters)
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

