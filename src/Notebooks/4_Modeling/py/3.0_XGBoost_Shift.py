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
    # Separar verdadeiro e falso
    false_label = X[X[label]==0].copy()
    true_label = X[X[label]==1].copy()
    
    # Realizar upsample para os valores verdadeiros
    label_upsampled = resample(true_label,
                            replace=True, # sample with replacement
                            n_samples=len(false_label), # match number in majority class
                            random_state=378) # reproducible results
    upsampled = pd.concat([false_label, label_upsampled])
    
    # Separar x e y
    x = upsampled[[c for c in X.columns if label not in c]]
    y = upsampled[label]
    
    return x, y


# In[ ]:


def trainXGB(df, cols_rem, label, verbose=True):
    xgb = xgboost.XGBClassifier()

    # Separar x e y e remover colunas desnecessárias
    x = df[[c for c in df.columns if c not in cols_rem]]
    y = df[label]
    
    # Separar dados de treinamento e teste
    x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3, random_state = 378, stratify=y)
    
    # Upsample
    X = pd.concat([x_treino, y_treino], axis=1)
    x_treino, y_treino = upsampleData(X, label)

    # Parâmetros do XGBClassifier
    param = {'max_depth':50, 'eta':1, 'objective':'binary:logistic', 'min_child_weight': 1, 'lambda': 1, 'alpha': 0, 'gamma': 0}

    # Gerar DMatrix com dados de treinamento e teste
    df_train = xgboost.DMatrix(data=x_treino, label=y_treino)
    df_test = xgboost.DMatrix(data=x_teste, label=y_teste)

    # Treinar modelo e predizer em cima dos dados de treinamento e teste
    bst = xgboost.train(param, df_train, 2, feval=f1_score)
    y_teste_pred = bst.predict(xgboost.DMatrix(data=x_teste, label=y_teste))
    y_teste_pred = [1 if i>0.5 else 0 for i in y_teste_pred]
    y_treino_pred = bst.predict(xgboost.DMatrix(data=x_treino, label=y_treino))
    y_treino_pred = [1 if i>0.5 else 0 for i in y_treino_pred]
    
    # Mostrar resultados se verbose é verdadeiro
    if verbose:
        print(f"Treino: {accuracy_score(y_treino, y_treino_pred)}")
        print(f"Teste: {accuracy_score(y_teste, y_teste_pred)}")
        print(f"Precisão: {precision_score(y_teste, y_teste_pred)}")
        print(f"Recall: {recall_score(y_teste, y_teste_pred)}")
        print(f"F1: {f1_score(y_teste, y_teste_pred)}")
        display(confusion_matrix(y_teste, y_teste_pred, normalize='true'))
        display(confusion_matrix(y_teste, y_teste_pred,))
        
    # Salvar resultados em um dict
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

# ## Carregar dados

# In[ ]:


df_p = pd.read_csv('../../../data/cleandata/Info pluviometricas/Merged Data/prepped_data.csv', sep=';')
df_p.groupby('Label').count()


# In[ ]:


df_p = df_p.sort_values(['Data_Hora', 'Local'])
# Shiftar label 6 horas para frente
df_p['Label'] = df_p['Label'].shift(-5*6, fill_value = 0)


# ## Treinar modelo

# In[ ]:


# Parâmetros
label = 'Label'
cols_rem = ['LocalMax', 'Label', 'Label_Old', 'Cluster', 'Data', 'Hora', 'Data_Hora', 'Ordens', 'Minuto'] + [c for c in df_p.columns if 'Hora_' in c]
# Conjunto de resultados
prepped_models = {}

# Gerar um modelo para cada local e um modelo geral
for l in range(6):
    if l != 0:
        df_train = df_p[df_p['Local'] == l]
    else:
        df_train = df_p.copy()
        
    print(f'----- LOCAL {l} -----')
    model, training_res, y_treino_pred, y_teste_pred = trainXGB(df_train, cols_rem, label)
    
    # Salvar modelo e resultados
    prepped_models[l] = {
        'model': model,
        'results': training_res,
        'y_treino': y_treino_pred,
        'y_teste': y_teste_pred
    }


# In[ ]:


prepped_models[0]['results']


# # Full Data

# ## Carregar dados

# In[ ]:


df_f = pd.read_csv('../../../data/cleandata/Info pluviometricas/Merged Data/full_data.csv', sep=';')
display(df_f.head())
df_f.shape


# ## Preparar dados

# In[ ]:


# Filtrar por hora -> minuto 0
df_f['Data_Hora'] = pd.to_datetime(df_f['Data_Hora'], yearfirst=True)
df_f = df_f[df_f['Data_Hora'].dt.minute == 0]
# Remover colunas desnecessárias
df_f = df_f.drop(columns = ['LocalMax_d_ow', 'LocalMax_h_All', 'LocalMax_h', 'LocalMax_h_ow', 'LocalMax_d'] + [c for c in df_f.columns if 'Local_' in c])
df_f = df_f.rename(columns = {'LocalMax_d_All': 'Label'})
# Colunas de data
df_f['Dia'] = df_f['Data_Hora'].dt.day
df_f['Mes'] = df_f['Data_Hora'].dt.month
df_f['Data'] = df_f['Data_Hora'].dt.strftime('%Y-%m-%d')
# Codificar local
df_f['Local'] = df_f['Local'].replace({'Camilopolis': 1, 'Erasmo': 2, 'Paraiso': 3, 'RM': 4, 'Vitoria': 5})


# In[ ]:


# Soma da precipitação do dia
df_prec_sum = df_f.groupby(['Data', 'Local']).sum().reset_index()[['Data', 'Local', 'Precipitacao']]
df_prec_sum.columns = ['Data', 'Local', 'PrecSum']
df_f = df_f.merge(df_prec_sum, on=['Data', 'Local'])
df_f.loc[(df_f['Label'] == 1) & (df_f['PrecSum'] <= 10), 'Label'] = 0


# In[ ]:


# Realizar OHE para colunas categóricas
cols_dummies = ['Local', 'Mes',]# 'Dia']

df_f_ohe = df_f.copy()

for c in cols_dummies:
    df_f_ohe = pd.concat([df_f_ohe, pd.get_dummies(df_f[c], prefix=c)], axis=1)
    
df_f_ohe = df_f_ohe.sort_values(['Data', 'Local'])

# Shiftar label 6 horas para frente
df_f_ohe['Label_Old'] = df_f_ohe['Label']
df_f_ohe['Label'] = df_f_ohe['Label'].shift(-5*6, fill_value = 0)


# In[ ]:


df_f_ohe.columns


# ## Treinar modelo

# In[ ]:


# Testar continuamente removendo colunas
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

# Salvar todos os resultados em um dataframe
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


# ### Melhores modelos para cada local

# In[ ]:


# Identificar melhores modelos
df_best_local = pd.DataFrame(columns = df_training_result.columns)

for l in range(6):
    df_best_local = df_best_local.append(df_training_result[(df_training_result['Local'] == l)].sort_values('F1', ascending=False).reset_index(drop=True).loc[0])
    
df_best_local


# In[ ]:


#df_best_local.to_csv('../../../data/analysis/best_shift_local.csv', index=False, sep=';', decimal=',')


# # OWM

# ## Carregar dados

# In[ ]:


df_owm = pd.read_csv('../../../data/cleandata/Info pluviometricas/Merged Data/OpenWeatherMapSantoAndre.csv')


# ## Preparar dados

# In[ ]:


# Preparar data hora
df_owm['Data_Hora'] = pd.to_datetime(df_owm['dt_iso'].str[:-10])
df_owm['Data_Hora'] = df_owm.apply(lambda x: x['Data_Hora'] + pd.Timedelta(hours = x['timezone'] / 3600), axis = 1)
# Filtrar nas datas que temos labels
df_owm = df_owm[(datetime.strptime('2019-08-30', '%Y-%m-%d') >= df_owm['Data_Hora']) & (df_owm['Data_Hora'] >= datetime.strptime('2010-01-01', '%Y-%m-%d'))]
# Remover colunas e valores nulos
df_owm = df_owm.drop(columns = ['sea_level', 'grnd_level', 'rain_3h', 'snow_1h', 'snow_3h'])
df_owm = df_owm.fillna(0)
df_owm = df_owm.drop_duplicates(subset='Data_Hora')


# In[ ]:


# Carregar labels e selecionar LocalMax
df_loc = pd.read_csv('../../../data/cleandata/Ordens de serviço/labels_day.csv', sep=';')
df_loc['Data'] = pd.to_datetime(df_loc['Data'], yearfirst=True)
df_loc = df_loc[['Data', 'LocalMax']]
df_loc.columns = ['Data', 'Label']
df_loc.head()


# In[ ]:


# Juntar dados com labels
df_owm['Data'] = pd.to_datetime(df_owm['Data_Hora'].dt.strftime('%Y-%m-%d'), yearfirst=True)
df = df_owm.merge(df_loc, on='Data', how='left')
df = df.fillna(0)


# In[ ]:


# Soma da precipitação do dia
df_g = df.groupby('Data').sum().reset_index()[['Data', 'rain_1h']]
df_g.columns = ['Data', 'rain_sum']
df = df.merge(df_g, on='Data')


# In[ ]:


# Colunas de data
df['Mes'] = df['Data_Hora'].dt.month
df['Dia'] = df['Data_Hora'].dt.day
# df = df.drop(columns = ['dt', 'dt_iso', 'timezone', 'city_name', 'lat', 'lon', 'weather_icon', 'weather_id', 'weather_main',
#                         'Vitoria', 'Erasmo', 'Paraiso', 'RM', 'Null', 'Camilopolis'])
# Remover colunas desnecessárias
df = df.drop(columns = ['dt', 'dt_iso', 'timezone', 'city_name', 'lat', 'lon', 'weather_icon', 'weather_id', 'weather_main'])
# df = df.drop(columns = ['dt', 'dt_iso', 'timezone', 'city_name', 'lat', 'lon', 'weather_icon', 'weather_id', 'weather_main',
#                         'Data_Hora', 'Data'])
# df['weather_description'] = df['weather_description'].rank(method='dense', ascending=False).astype(int)


# In[ ]:


# Realizar OHE para colunas categóricas
cols_dummies = ['Mes', 'weather_description', 'Dia']

df_ohe = df.copy()

for c in cols_dummies:
    df_ohe = pd.concat([df_ohe, pd.get_dummies(df_ohe[c], prefix=c)], axis=1)
    
df_ohe = df_ohe.sort_values(['Data'])

# Shiftar label 6 horas para frente
df_ohe['Label_Old'] = df_ohe['Label']
df_ohe['Label'] = df_ohe['Label'].shift(-1*6, fill_value = 0)


# In[ ]:


df_ohe.columns


# ## Treinar modelo

# In[ ]:


# Remover colunas de passo em passo
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

# Salvar todos os resultados em um dataframe
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


# # Usar modelo 1 para gerar novas labels

# ## Treinar modelo

# In[ ]:


df_p = pd.read_csv('../../../data/cleandata/Info pluviometricas/Merged Data/prepped_data.csv', sep=';')
df_p['Data_Hora'] = pd.to_datetime(df_p['Data_Hora'], yearfirst=True)
df_p.groupby('Label').count()


# In[ ]:


df_p = df_p.sort_values(['Data_Hora', 'Local'])


# In[ ]:


# Parâmetros
label = 'Label'
cols_rem = ['LocalMax', 'Label', 'Label_Old', 'Cluster', 'Data', 'Hora', 'Data_Hora', 'Ordens', 'Minuto'] + [c for c in df_p.columns if 'Hora_' in c]
# Conjunto de resultados
prepped_models = {}

# Gerar um modelo para cada local e um modelo geral
for l in range(6):
    if l != 0:
        df_train = df_p[df_p['Local'] == l]
    else:
        df_train = df_p.copy()
        
    print(f'----- LOCAL {l} -----')
    model, training_res, y_treino_pred, y_teste_pred = trainXGB(df_train, cols_rem, label)
    
    # Salvar modelo e resultados
    prepped_models[l] = {
        'model': model,
        'results': training_res,
        'y_treino': y_treino_pred,
        'y_teste': y_teste_pred
    }


# In[ ]:


prepped_models[0]['results']


# In[ ]:


prepped_models[0]['model']


# ## Obter precipitação até aquele momento do dia

# In[ ]:


df_m = df_p[df_p['Label'] == 1].copy()
df_m['Data'] = df_m['Data_Hora'].dt.strftime("%Y-%m-%d")

def getPrecMomento(row):
    prec_momento = df_m.loc[(df_m['Data_Hora'] <= row['Data_Hora']) & (df_m['Local'] == row['Local']) & (df_m['Data'] == row['Data']), 'Precipitacao'].sum()
    return prec_momento

df_m['PrecMomento'] = df_m.apply(getPrecMomento, axis=1)

df_m = df_m.rename(columns = {'PrecSum': 'PrecSumOld', 'PrecMomento': 'PrecSum'})


# ## Prever com acumulo do dia

# In[ ]:


label_pred = prepped_models[0]['model'].predict(xgboost.DMatrix(data=df_m[prepped_models[0]['results']['Features']]))
df_m['Label_Pred'] = [1 if i>0.5 else 0 for i in label_pred]


# In[ ]:


df_m['Label_Pred']


# Teste: se, com essa estratégia, o modelo consegue prever em pelo menos um momento do dia com label que vai ocorrer alagamento

# In[ ]:


df_g = df_m.groupby(['Data', 'Local']).max()
print(df_g[df_g['Label'] == df_g['Label_Pred']].shape)
print(df_g.shape)


# Próximo passo: obter o primeiro momento em que o modelo prevê um alagamento

# In[ ]:


df_m.columns


# In[ ]:


df_g = df_m.groupby(['Data', 'Local', 'Label_Pred']).min().reset_index()
#df_g[df_g['Label_Pred'] == 1]

df_g = df_g.loc[df_g['Label_Pred'] == 1, ['Data', 'Local', 'Data_Hora']].rename(columns={'Data_Hora':'Min_Hora'})
df_g['Min_Hora'] = df_g['Min_Hora'].dt.hour


# ## Treinar modelo com todos os dias

# In[ ]:


df_p_new = df_p.copy()
df_p_new['Data'] = df_p_new['Data_Hora'].dt.strftime('%Y-%m-%d')
df_p_new = df_p_new.merge(df_g, on=['Local', 'Data'], how='left').fillna(24)


# In[ ]:


df_p_new['Label_New'] = 0
df_p_new.loc[(df_p_new['Label'] == 1) & (df_p_new['Data_Hora'].dt.hour >= df_p_new['Min_Hora']), 'Label_New'] = 1
df_p_new = df_p_new.rename(columns = {'Label': 'Label_Old', 'Label_New': 'Label'})


# In[ ]:


df_p_new['Label'].value_counts()


# In[ ]:


df_p_new['Label'] = df_p_new['Label'].shift(-5*6, fill_value = 0)


# ...

# In[ ]:


df_p_new[['Data_Hora', 'Local', 'Label']]


# In[ ]:


# Parâmetros
label = 'Label'
cols_rem = ['LocalMax', 'Label', 'Label_Old', 'Cluster', 'Data', 'Hora', 'Data_Hora', 'Ordens', 'Minuto', 'Min_Hora'] + [c for c in df_p.columns if 'Hora_' in c]
# Conjunto de resultados
prepped_models_new = {}

# Gerar um modelo para cada local e um modelo geral
for l in range(6):
    if l != 0:
        df_train = df_p_new[df_p_new['Local'] == l]
    else:
        df_train = df_p_new.copy()
        
    print(f'----- LOCAL {l} -----')
    model, training_res, y_treino_pred, y_teste_pred = trainXGB(df_train, cols_rem, label)
    
    # Salvar modelo e resultados
    prepped_models_new[l] = {
        'model': model,
        'results': training_res,
        'y_treino': y_treino_pred,
        'y_teste': y_teste_pred
    }


# ### Treinar modelo com mais features

# In[ ]:


df_f_ohe.drop(columns='Label').merge(df_p_new[['Data_Hora', 'Local', 'Label']], on=['Data_Hora', 'Local'])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




