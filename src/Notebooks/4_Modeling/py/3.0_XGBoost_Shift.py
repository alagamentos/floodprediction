#!/usr/bin/env python
# coding: utf-8

# # Inicialização

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


# # Funções

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


# # Prepped Data

# In[ ]:


df_p = pd.read_csv('../../../data/cleandata/Info pluviometricas/Merged Data/prepped_data.csv', sep=';')
df_p.groupby('Label').count()


# In[ ]:


df_p = df_p.sort_values(['Data_Hora', 'Local'])
df_p['Label'] = df_p['Label'].shift(-5*6, fill_value = 0)


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


prepped_models[0]['results']


# # Full Data

# In[ ]:


df_f = pd.read_csv('../../../data/cleandata/Info pluviometricas/Merged Data/full_data.csv', sep=';')
display(df_f.head())
df_f.shape


# In[ ]:


df_f['Data_Hora'] = pd.to_datetime(df_f['Data_Hora'], yearfirst=True)
df_f = df_f[df_f['Data_Hora'].dt.minute == 0]
df_f = df_f.drop(columns = ['LocalMax_d_ow', 'LocalMax_h_All', 'LocalMax_h', 'LocalMax_h_ow', 'LocalMax_d'] + [c for c in df_f.columns if 'Local_' in c])
df_f = df_f.rename(columns = {'LocalMax_d_All': 'Label'})
df_f['Dia'] = df_f['Data_Hora'].dt.day
df_f['Mes'] = df_f['Data_Hora'].dt.month
df_f['Data'] = df_f['Data_Hora'].dt.strftime('%Y-%m-%d')
df_f['Local'] = df_f['Local'].replace({'Camilopolis': 1, 'Erasmo': 2, 'Paraiso': 3, 'RM': 4, 'Vitoria': 5})


# In[ ]:


df_prec_sum = df_f.groupby(['Data', 'Local']).sum().reset_index()[['Data', 'Local', 'Precipitacao']]
df_prec_sum.columns = ['Data', 'Local', 'PrecSum']
df_f = df_f.merge(df_prec_sum, on=['Data', 'Local'])
df_f.loc[(df_f['Label'] == 1) & (df_f['PrecSum'] <= 10), 'Label'] = 0


# In[ ]:


cols_dummies = ['Local', 'Mes',]# 'Dia']

df_f_ohe = df_f.copy()

for c in cols_dummies:
    df_f_ohe = pd.concat([df_f_ohe, pd.get_dummies(df_f[c], prefix=c)], axis=1)
    
df_f_ohe = df_f_ohe.sort_values(['Data', 'Local'])

df_f_ohe['Label_Old'] = df_f_ohe['Label']
df_f_ohe['Label'] = df_f_ohe['Label'].shift(-5*6, fill_value = 0)


# In[ ]:


df_f_ohe.columns


# In[ ]:


test_cases = [
    [],
    ['DirecaoDoVento', 'VelocidadeDoVento'],
    ['DirecaoDoVento', 'VelocidadeDoVento', 'TemperaturaInterna'],
    ['DirecaoDoVento', 'VelocidadeDoVento', 'TemperaturaInterna', 'PontoDeOrvalho'],
    ['DirecaoDoVento', 'VelocidadeDoVento', 'TemperaturaInterna', 'PontoDeOrvalho', 'UmidadeRelativa'],
    ['DirecaoDoVento', 'VelocidadeDoVento', 'TemperaturaInterna', 'PontoDeOrvalho', 'UmidadeRelativa', 'PressaoAtmosferica'],
    ['DirecaoDoVento', 'VelocidadeDoVento', 'TemperaturaInterna', 'PontoDeOrvalho', 'UmidadeRelativa', 'PressaoAtmosferica', 'RadiacaoSolar'],
    ['DirecaoDoVento', 'VelocidadeDoVento', 'TemperaturaInterna', 'PontoDeOrvalho', 'UmidadeRelativa', 'PressaoAtmosferica', 'RadiacaoSolar', 'TemperaturaDoAr'],
    ['DirecaoDoVento', 'VelocidadeDoVento', 'TemperaturaInterna', 'PontoDeOrvalho', 'UmidadeRelativa', 'PressaoAtmosferica', 'RadiacaoSolar', 'TemperaturaDoAr', 'Local'],
    ['DirecaoDoVento', 'VelocidadeDoVento', 'TemperaturaInterna', 'PontoDeOrvalho', 'UmidadeRelativa', 'PressaoAtmosferica', 'RadiacaoSolar', 'TemperaturaDoAr', 'PrecSum'],
    ['DirecaoDoVento', 'VelocidadeDoVento', 'TemperaturaInterna', 'PontoDeOrvalho', 'UmidadeRelativa', 'PressaoAtmosferica', 'RadiacaoSolar', 'TemperaturaDoAr', 'Precipitacao']
]

df_training_result = pd.DataFrame(columns = ['Removed_Cols', 'Local', 'Features', 'Train_Acc', 'Test_Acc', 'Precision', 'Recall', 'F1', 'Ver_Pos'])
label = 'Label'

for case in test_cases:
    print(f'---------- CASE ----------')
    print(case)
    print(f'--------------------------')
    for l in range(6):
        if l != 0:
            df_train = df_f_ohe[df_f_ohe['Local'] == l].drop(columns = [c for c in df_f_ohe.columns if 'Local' in c])
        else:
            df_train = df_f_ohe.copy()

        cols_rem = ['Label', 'Label_Old', 'Data', 'Data_Hora'] + cols_dummies
        cols_rem = cols_rem + case
            
        print(f'----- LOCAL {l} -----')
        model, training_res, y_treino_pred, y_teste_pred = trainXGB(df_train, cols_rem, label)
        
        df_training_result = df_training_result.append(
            {**{'Model': model, 'Removed_Cols': case, 'Local': l}, **training_res},
            ignore_index=True
        )


# In[ ]:


#df_training_result.to_csv('../../../data/analysis/training_test_shift.csv', index=False, sep=';', decimal=',')
df_training_result


# In[ ]:


df_best_local = pd.DataFrame(columns = df_training_result.columns)

for l in range(6):
    df_best_local = df_best_local.append(df_training_result[(df_training_result['Local'] == l)].sort_values('F1', ascending=False).reset_index(drop=True).loc[0])
    
df_best_local


# In[ ]:


#df_best_local.to_csv('../../../data/analysis/best_shift_local.csv', index=False, sep=';', decimal=',')


# # OWM

# In[ ]:


df_owm = pd.read_csv('../../../data/cleandata/Info pluviometricas/Merged Data/OpenWeatherMapSantoAndre.csv')


# In[ ]:


df_owm['Data_Hora'] = pd.to_datetime(df_owm['dt_iso'].str[:-10])
df_owm['Data_Hora'] = df_owm.apply(lambda x: x['Data_Hora'] + pd.Timedelta(hours = x['timezone'] / 3600), axis = 1)
df_owm = df_owm[(datetime.strptime('2019-08-30', '%Y-%m-%d') >= df_owm['Data_Hora']) & (df_owm['Data_Hora'] >= datetime.strptime('2010-01-01', '%Y-%m-%d'))]
df_owm = df_owm.drop(columns = ['sea_level', 'grnd_level', 'rain_3h', 'snow_1h', 'snow_3h'])
df_owm = df_owm.fillna(0)
df_owm = df_owm.drop_duplicates(subset='Data_Hora')


# In[ ]:


df_loc = pd.read_csv('../../../data/cleandata/Ordens de serviço/labels_day.csv', sep=';')
df_loc['Data'] = pd.to_datetime(df_loc['Data'], yearfirst=True)
df_loc = df_loc[['Data', 'LocalMax']]
df_loc.columns = ['Data', 'Label']
df_loc.head()


# In[ ]:


df_owm['Data'] = pd.to_datetime(df_owm['Data_Hora'].dt.strftime('%Y-%m-%d'), yearfirst=True)
df = df_owm.merge(df_loc, on='Data', how='left')
df = df.fillna(0)


# In[ ]:


df_g = df.groupby('Data').sum().reset_index()[['Data', 'rain_1h']]
df_g.columns = ['Data', 'rain_sum']
df = df.merge(df_g, on='Data')


# In[ ]:


df['Mes'] = df['Data_Hora'].dt.month
df['Dia'] = df['Data_Hora'].dt.day
# df = df.drop(columns = ['dt', 'dt_iso', 'timezone', 'city_name', 'lat', 'lon', 'weather_icon', 'weather_id', 'weather_main',
#                         'Vitoria', 'Erasmo', 'Paraiso', 'RM', 'Null', 'Camilopolis'])
df = df.drop(columns = ['dt', 'dt_iso', 'timezone', 'city_name', 'lat', 'lon', 'weather_icon', 'weather_id', 'weather_main'])
# df = df.drop(columns = ['dt', 'dt_iso', 'timezone', 'city_name', 'lat', 'lon', 'weather_icon', 'weather_id', 'weather_main',
#                         'Data_Hora', 'Data'])
# df['weather_description'] = df['weather_description'].rank(method='dense', ascending=False).astype(int)


# In[ ]:


cols_dummies = ['Mes', 'weather_description', 'Dia']

df_ohe = df.copy()

for c in cols_dummies:
    df_ohe = pd.concat([df_ohe, pd.get_dummies(df_ohe[c], prefix=c)], axis=1)
    
df_ohe = df_ohe.sort_values(['Data'])

df_ohe['Label_Old'] = df_ohe['Label']
df_ohe['Label'] = df_ohe['Label'].shift(-1*6, fill_value = 0)


# In[ ]:


df_ohe.columns


# In[ ]:


test_cases = [
    [],
    ['feels_like'],
    ['feels_like', 'temp_min', 'temp_max'],
    ['feels_like', 'temp_min', 'temp_max', 'pressure'],
    ['feels_like', 'temp_min', 'temp_max', 'pressure', 'humidity'],
    ['feels_like', 'temp_min', 'temp_max', 'pressure', 'humidity', 'wind_speed', 'wind_deg'],
    ['feels_like', 'temp_min', 'temp_max', 'pressure', 'humidity', 'wind_speed', 'wind_deg', 'clouds_all'],
    ['feels_like', 'temp_min', 'temp_max', 'pressure', 'humidity', 'wind_speed', 'wind_deg', 'clouds_all'] + [c for c in df_ohe.columns if 'weather_description' in c],
    ['temp_min', 'temp_max', 'pressure', 'humidity', 'wind_speed', 'wind_deg', 'clouds_all'] + [c for c in df_ohe.columns if 'weather_description' in c],
]

df_training_result_owm = pd.DataFrame(columns = ['Removed_Cols', 'Features', 'Train_Acc', 'Test_Acc', 'Precision', 'Recall', 'F1', 'Ver_Pos'])
label = 'Label'

for case in test_cases:
    print(f'---------- CASE ----------')
    print(case)
    print(f'--------------------------')
    
    df_train = df_ohe.copy()

    cols_rem = ['Label', 'Label_Old', 'Data', 'Data_Hora'] + cols_dummies
    cols_rem = cols_rem + case

    model, training_res, y_treino_pred, y_teste_pred = trainXGB(df_train, cols_rem, label)

    df_training_result_owm = df_training_result_owm.append(
        {**{'Removed_Cols': case}, **training_res},
        ignore_index=True
    )


# In[ ]:


df_training_result_owm


# # Teste "real"

# In[ ]:


df_best_local = df_best_local.reset_index(drop=True)
df_best_local


# In[ ]:


df_m = df_f_ohe[(df_f_ohe['Label'] == 1) | (df_f_ohe['Label_Old'] == 1)].copy()
df_m['Data'] = df_m['Data_Hora'].dt.strftime('%Y-%m-%d')


# In[ ]:


def getPrecMomento(row):
    prec_momento = df_m.loc[(df_m['Data_Hora'] <= row['Data_Hora']) & (df_m['Local'] == row['Local']) & (df_m['Data'] == row['Data']), 'Precipitacao'].sum()
    return prec_momento

df_m['PrecMomento'] = df_m.apply(getPrecMomento, axis=1)

df_m = df_m.rename(columns = {'PrecSum': 'PrecSumOld', 'PrecMomento': 'PrecSum'})


# In[ ]:


df_m_2 = df_m.copy()
df_m_2['Label_Pred'] = 0

for l in range(6):
    label_pred = df_best_local.loc[l,'Model'].predict(xgboost.DMatrix(data=df_m_2.loc[df_m_2['Local'] == l, df_best_local.loc[l, 'Features']]))
    df_m_2.loc[df_m_2['Local'] == l, 'Label_Pred'] = [1 if i>0.5 else 0 for i in label_pred]


# In[ ]:


print(df_m_2[df_m_2['Label_Pred'] == 1].shape)
print(df_m_2.shape)


# In[ ]:


df_m_2[['Local', 'Data_Hora', 'Precipitacao', 'PrecSum', 'PrecSumOld', 'Label', 'Label_Pred']].sort_values(by=['Local', 'Data_Hora']
).to_csv('../../../data/analysis/labels_prediction_shift.csv', index=False, sep=';', decimal=',')


# In[ ]:


pd.set_option("display.max_rows", 200)
df_f_ohe[(df_f_ohe['Data_Hora'] >= datetime(2018,12,23)) & (df_f_ohe['Data_Hora'] <= datetime(2018,12,30)) & (df_f_ohe['Local'] == 4)][['Data_Hora', 'Precipitacao', 'PrecSum']]


# In[ ]:


df_f_ohe['Local']


# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




