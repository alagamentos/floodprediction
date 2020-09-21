#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


df = pd.read_csv('../../../data/cleandata/OpenWeather/history_forecast_bulk.csv')


# In[ ]:


df.isna().sum()


# In[ ]:


df.shape


# In[ ]:


df[df['rain'] != df['accumulated']]


# In[ ]:


df_p = pd.read_csv('../../../data/cleandata/Info pluviometricas/Merged Data/prepped_data.csv', sep=';')
df_p['Data_Hora'] = pd.to_datetime(df_p['Data_Hora'], yearfirst=True)


# In[ ]:


df['Data_Forecast'] = pd.to_datetime(df['forecast dt iso'].str[:-10])
df['Data_Slice'] = pd.to_datetime(df['slice dt iso'].str[:-10])
df['rain'] = df['rain'].fillna(0)


# In[ ]:


df['Data_Hora'].min()


# In[ ]:


df_owm = df[(df['Data_Slice'] - df['Data_Forecast']).astype('timedelta64[h]') <= 3]


# In[ ]:


df_m = df_p.merge(df_owm[['Data_Hora', 'rain']], on='Data_Hora', how='left')


# In[ ]:


#df_m[(~df_m['rain'].isna()) & (df_m['rain'] > 0)]
df_m['Diferença'] = np.sqrt((df_m['Precipitacao'] - df_m['rain']) ** 2)


# In[ ]:


from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

df_slice = df_m[(~df_m['rain'].isna()) & (df_m['Local'] == 4)]

print(sqrt(mean_squared_error(df_slice['Precipitacao'], df_slice['rain'])))
print(r2_score(df_slice['Precipitacao'], df_slice['rain']))


# In[ ]:


import plotly.express as px
import plotly.graph_objects as go

# fig = px.line(df_slice, x="Data_Hora", y="Precipitacao")
# fig.show()

df_slice = df_m[(~df_m['rain'].isna())]

fig = go.Figure()
fig.add_trace(go.Scatter(x=df_slice.loc[df_slice['Local'] == 1, "Data_Hora"], y=df_slice.loc[df_slice['Local'] == 1, "Precipitacao"],
                    mode='lines',
                    name='Prec 1'))
fig.add_trace(go.Scatter(x=df_slice.loc[df_slice['Local'] == 2, "Data_Hora"], y=df_slice.loc[df_slice['Local'] == 2, "Precipitacao"],
                    mode='lines',
                    name='Prec 2'))
fig.add_trace(go.Scatter(x=df_slice.loc[df_slice['Local'] == 3, "Data_Hora"], y=df_slice.loc[df_slice['Local'] == 3, "Precipitacao"],
                    mode='lines',
                    name='Prec 3'))
fig.add_trace(go.Scatter(x=df_slice.loc[df_slice['Local'] == 4, "Data_Hora"], y=df_slice.loc[df_slice['Local'] == 4, "Precipitacao"],
                    mode='lines',
                    name='Prec 4'))
fig.add_trace(go.Scatter(x=df_slice.loc[df_slice['Local'] == 5, "Data_Hora"], y=df_slice.loc[df_slice['Local'] == 5, "Precipitacao"],
                    mode='lines',
                    name='Prec 5'))
fig.add_trace(go.Scatter(x=df_slice.loc[df_slice['Local'] == 5, "Data_Hora"], y=df_slice.loc[df_slice['Local'] == 5, "rain"],
                    mode='lines',
                    name='rain'))

fig.show()


# In[ ]:


from datetime import datetime, timedelta


# In[ ]:


df_slice['Data'] = df_slice['Data_Hora'].dt.strftime('%Y-%m-%d')
df_prec_sum = df_slice.groupby(['Data', 'Local']).sum().reset_index()[['Data', 'Local', 'rain']]
df_prec_sum.columns = ['Data', 'Local', 'rain_sum']
df_slice = df_slice.merge(df_prec_sum, on=['Data', 'Local'])


# In[ ]:


df_owm = 


# In[ ]:


df_slice_2 = df_slice.groupby(['Data', 'Local']).max().reset_index()

fig = go.Figure()
fig.add_trace(go.Scatter(x=df_slice_2.loc[df_slice_2['Local'] == 1, "Data_Hora"], y=df_slice_2.loc[df_slice_2['Local'] == 1, "PrecSum"],
                    mode='lines',
                    name='Prec 1'))
fig.add_trace(go.Scatter(x=df_slice_2.loc[df_slice_2['Local'] == 2, "Data_Hora"], y=df_slice_2.loc[df_slice_2['Local'] == 2, "PrecSum"],
                    mode='lines',
                    name='Prec 2'))
fig.add_trace(go.Scatter(x=df_slice_2.loc[df_slice_2['Local'] == 3, "Data_Hora"], y=df_slice_2.loc[df_slice_2['Local'] == 3, "PrecSum"],
                    mode='lines',
                    name='Prec 3'))
fig.add_trace(go.Scatter(x=df_slice_2.loc[df_slice_2['Local'] == 4, "Data_Hora"], y=df_slice_2.loc[df_slice_2['Local'] == 4, "PrecSum"],
                    mode='lines',
                    name='Prec 4'))
fig.add_trace(go.Scatter(x=df_slice_2.loc[df_slice_2['Local'] == 5, "Data_Hora"], y=df_slice_2.loc[df_slice_2['Local'] == 5, "PrecSum"],
                    mode='lines',
                    name='Prec 5'))
fig.add_trace(go.Scatter(x=df_slice_2.loc[df_slice_2['Local'] == 5, "Data_Hora"], y=df_slice_2.loc[df_slice_2['Local'] == 5, "rain_sum"],
                    mode='lines',
                    name='rain'))
fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)

fig.show()


# In[ ]:


df.columns


# # ...

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from plotly import graph_objects as go
import plotly as py

from datetime import datetime
from datetime import timedelta

import xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, accuracy_score, f1_score, confusion_matrix, recall_score, precision_score

from sklearn.utils import resample


# In[ ]:


def upsampleData(X, label):
    # Separate true and false
    false_label = X[X[label]==0].copy()
    true_label = X[X[label]==1].copy()
    
    # Upsample true values
    label_upsampled = resample(true_label,
                            replace=True, # sample with replacement
                            n_samples=len(false_label), # match number in majority class
                            random_state=378) # reproducible results
    upsampled = pd.concat([false_label, label_upsampled])
    
    # Separate x and y
    x = upsampled[[c for c in X.columns if label not in c]]
    y = upsampled[label]
    
    return x, y


# In[ ]:


def trainXGB(df, cols_rem, label, verbose=True):
    xgb = xgboost.XGBClassifier()

    # Separate x and y and remove unnecessary columns
    x = df[[c for c in df.columns if c not in cols_rem]]
    y = df[label]
    
    # Split training and test data
    x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3, random_state = 378, stratify=y)
    
    # Upsample true values
    X = pd.concat([x_treino, y_treino], axis=1)
    x_treino, y_treino = upsampleData(X, label)

    # XGBClassifier parameters
    param = {'max_depth':50, 'eta':1, 'objective':'binary:logistic', 'min_child_weight': 1, 'lambda': 1, 'alpha': 0, 'gamma': 0}

    # Generate DMatrices with training and test data
    df_train = xgboost.DMatrix(data=x_treino, label=y_treino)
    df_test = xgboost.DMatrix(data=x_teste, label=y_teste)

    # Train model and predict on training and test data
    bst = xgboost.train(param, df_train, 2, feval=f1_score)
    y_teste_pred = bst.predict(xgboost.DMatrix(data=x_teste, label=y_teste))
    y_teste_pred = [1 if i>0.5 else 0 for i in y_teste_pred]
    y_treino_pred = bst.predict(xgboost.DMatrix(data=x_treino, label=y_treino))
    y_treino_pred = [1 if i>0.5 else 0 for i in y_treino_pred]
    
    # Print results if verbose is true
    if verbose:
        print(f"Treino: {accuracy_score(y_treino, y_treino_pred)}")
        print(f"Teste: {accuracy_score(y_teste, y_teste_pred)}")
        print(f"Precisão: {precision_score(y_teste, y_teste_pred)}")
        print(f"Recall: {recall_score(y_teste, y_teste_pred)}")
        print(f"F1: {f1_score(y_teste, y_teste_pred)}")
        display(confusion_matrix(y_teste, y_teste_pred, normalize='true'))
        display(confusion_matrix(y_teste, y_teste_pred,))
        
    # Store results in a dict
    results = {
        'Features': list(x.columns),
        'Train_Acc': accuracy_score(y_treino, y_treino_pred),
        'Test_Acc': accuracy_score(y_teste, y_teste_pred),
        'Precision': precision_score(y_teste, y_teste_pred),
        'Recall': recall_score(y_teste, y_teste_pred),
        'F1': f1_score(y_teste, y_teste_pred),
        'Ver_Pos': confusion_matrix(y_teste, y_teste_pred, normalize='true')[1,1]
    }
    
    return bst, results, y_treino_pred, y_teste_pred


# In[ ]:


df_p = pd.read_csv('../../../data/cleandata/Info pluviometricas/Merged Data/prepped_data.csv', sep=';')
df_p.groupby('Label').count()


# In[ ]:


df_p = df_p.sort_values(['Data_Hora', 'Local'])
#df_p['Label'] = df_p['Label'].shift(-5*6, fill_value = 0)


# In[ ]:


# Parameters
label = 'Label'
cols_rem = ['LocalMax', 'Label', 'Label_Old', 'Cluster', 'Data', 'Hora', 'Data_Hora', 'Ordens', 'Minuto'] + [c for c in df_p.columns if 'Hora_' in c]
# Result set
prepped_models = {}

for l in range(6):
    if l != 0:
        df_train = df_p[df_p['Local'] == l]
    else:
        df_train = df_p.copy()
        
    print(f'----- LOCAL {l} -----')
    model, training_res, y_treino_pred, y_teste_pred = trainXGB(df_train, cols_rem, label)
    
    prepped_models[l] = {
        'model': model,
        'results': training_res,
        'y_treino': y_treino_pred,
        'y_teste': y_teste_pred
    }


# In[ ]:


df_owm['Data'] = df_owm['Data_Slice'].dt.strftime('%Y-%m-%d')
df_owm_grouped = df_owm.groupby('Data').sum().reset_index()[['Data', 'rain']]


# In[ ]:


df_p['Data_Hora'] = pd.to_datetime(df_p['Data_Hora'], yearfirst=True)
df_p['Data'] = df_p['Data_Hora'].dt.strftime('%Y-%m-%d')


# In[ ]:


df = df_p.merge(df_owm_grouped, on='Data').rename(columns={'PrecSum': 'PrecSumOld', 'rain': 'PrecSum'})


# In[ ]:


prepped_models[0]['results']['Features']


# In[ ]:


label_pred = prepped_models[0]['model'].predict(xgboost.DMatrix(data=df[prepped_models[0]['results']['Features']]))
df['Label_pred'] = [1 if i>0.5 else 0 for i in label_pred]


# In[ ]:


df.columns


# In[ ]:


df.loc[df['Label'] == 1, ['Local', 'Data_Hora', 'Precipitacao', 'PrecSum', 'PrecSumOld', 'Label', 'Label_pred']].sort_values(by=['Local', 'Data_Hora']
).to_csv('../../../data/analysis/labels_prediction_owm_forecast.csv', index=False, sep=';', decimal=',')


# In[ ]:


print(df[(df['Label'] == df['Label_pred']) & (df['Label'] == 1)].shape)
print(df[df['Label'] == 1].shape)

