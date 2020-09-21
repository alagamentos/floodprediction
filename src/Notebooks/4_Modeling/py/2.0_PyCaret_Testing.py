#!/usr/bin/env python
# coding: utf-8

# # Inicialização

# In[ ]:


from pycaret.classification import *
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from sklearn.utils import resample


# # Prepped Data

# In[ ]:


df = pd.read_csv('../../../data/cleandata/Info pluviometricas/Merged Data/prepped_data.csv', sep=';')
df.groupby('Label').count()


# ## Upsampling

# In[ ]:


label_name = 'Label'

not_label = df[df[label_name]==0].copy()
label = df[df[label_name]==1].copy()

# upsample minority
upsampled = resample(label,
                     replace=True, # sample with replacement
                     n_samples=len(not_label), # match number in majority class
                     random_state=378) # reproducible results

# combine majority and upsampled minority
df_upsampled = pd.concat([not_label, upsampled])
df_upsampled.groupby('Label').count()


# In[ ]:


#df_upsampled = df.copy()
df_upsampled = df_upsampled.drop(columns = 'Data_Hora')
df_upsampled['Label'] = df_upsampled['Label'].astype(int)


# ## PyCaret: Setup

# In[ ]:


clf = setup(df_upsampled, target = 'Label', session_id=42, log_experiment=True, experiment_name='clf', fix_imbalance = True)


# ## Obter melhor modelo

# In[ ]:


best_model = compare_models()


# In[ ]:


best_model


# ## Testar com DecisionTreeClassifier

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, accuracy_score, f1_score, confusion_matrix, recall_score, precision_score


# In[ ]:


cols_rem = ['Label', 'Data_Hora']
label_name = 'Label'

x = df[[c for c in df.columns if c not in cols_rem]]
y = df[label_name]

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3, random_state = 378)

X = pd.concat([x_treino, y_treino], axis=1)

# separate minority and majority classes
not_ordem = X[X[label_name]==0].copy()
ordem = X[X[label_name]==1].copy()

# upsample minority
ordem_upsampled = resample(ordem,
                        replace=True, # sample with replacement
                        n_samples=len(not_ordem), # match number in majority class
                        random_state=378) # reproducible results

# combine majority and upsampled minority
upsampled = pd.concat([not_ordem, ordem_upsampled])

x_treino = upsampled[[c for c in df.columns if c not in cols_rem]]
y_treino = upsampled[label_name]

display(y_treino.value_counts())


# In[ ]:


model = DecisionTreeClassifier(random_state=42)

model.fit(x_treino, y_treino)


# In[ ]:


model = DecisionTreeClassifier(random_state=42)

model.fit(x_treino, y_treino)
y_teste_pred = model.predict(x_teste)
y_treino_pred = model.predict(x_treino)

print(f"Treino: {accuracy_score(y_treino, y_treino_pred)}")
print(f"Teste: {accuracy_score(y_teste, y_teste_pred)}")
print(f"Precisão: {precision_score(y_teste, y_teste_pred)}")
print(f"Recall: {recall_score(y_teste, y_teste_pred)}")
print(f"F1: {f1_score(y_teste, y_teste_pred)}")
display(confusion_matrix(y_teste, y_teste_pred, normalize='true'))
display(confusion_matrix(y_teste, y_teste_pred,))


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


plot_model(model, plot = 'boundary')


# In[ ]:


model = create_model('xgboost')


# In[ ]:


model = create_model('dt')


# In[ ]:


model


# In[ ]:


x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3, random_state = 378)


# # Full Data

# In[ ]:


df_f = pd.read_csv('../../../data/cleandata/Info pluviometricas/Merged Data/full_data.csv', sep=';')
display(df_f.head())
print(df_f.shape)
print(df_f.columns)


# In[ ]:


df_f = df_f.drop(columns = ['LocalMax_d_All', 'LocalMax_d_ow', 'Local_d_Null', 'LocalMax_h_ow', 'Local_h_Null', 'LocalMax_d', 'Local_d', 'Local_h'])
df_f['Data_Hora'] = pd.to_datetime(df_f['Data_Hora'], yearfirst=True)
df_f['Data'] = df_f['Data_Hora'].dt.strftime('%Y-%m-%d')
df_f['Mes'] = df_f['Data_Hora'].dt.month
df_f['Dia'] = df_f['Data_Hora'].dt.day
df_f = df_f[df_f['Data_Hora'].dt.minute == 0]


# ## PrecSum

# In[ ]:


df_prec_sum = df_f.groupby(['Data', 'Local']).sum().reset_index()[['Data', 'Local', 'Precipitacao']]
df_prec_sum.columns = ['Data', 'Local', 'PrecSum']
df_f = df_f.merge(df_prec_sum, on=['Data', 'Local'])
df_f.head(10)


# In[ ]:


label_name = 'LocalMax_h_All'

df_f.loc[(df_f[label_name] == 1) & (df_f['PrecSum'] <= 10), label_name] = 0
df_f.groupby(label_name).count()


# ## Upsampling

# In[ ]:


x = df_f[[c for c in df_f.columns if label_name not in c]]
y = df_f[label_name]

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3, random_state = 378)

X = pd.concat([x_treino, y_treino], axis=1)

# separate minority and majority classes
not_label = X[X[label_name]==0].copy()
label = X[X[label_name]==1].copy()

# upsample minority
upsampled = resample(label,
                        replace=True, # sample with replacement
                        n_samples=len(not_label), # match number in majority class
                        random_state=378) # reproducible results

# combine majority and upsampled minority
df_f_upsampled = pd.concat([not_label, upsampled])

x_treino = df_f_upsampled[[c for c in df_f.columns if label_name not in c]]
y_treino = df_f_upsampled[label_name]

df_f_upsampled = pd.concat([x_treino, y_treino], axis=1)

display(y_treino.value_counts())


# ## PyCaret: Setup

# In[ ]:


df_f_upsampled = df_f.copy()


# In[ ]:


df_f_upsampled = df_f_upsampled.drop(columns = ['Data_Hora', 'Data', 'LocalMax_h'])


# In[ ]:


df_f_upsampled[label_name] = df_f_upsampled[label_name].astype(int)


# In[ ]:


clf_f = setup(df_f_upsampled, target = label_name, session_id=42, log_experiment=True, experiment_name='clf_f', fix_imbalance = True)


# ## Obter melhor modelo

# In[ ]:


best_model_f = compare_models()


# In[ ]:


best_model_f


# In[ ]:


model = create_model('xgboost')


# In[ ]:


plot_model(model, 'feature')


# In[ ]:


plot_model(model, 'boundary')


# ## Testar com ExtraTreesClassifier

# In[ ]:


from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, accuracy_score, f1_score, confusion_matrix, recall_score, precision_score


# In[ ]:


df_f_train = df_f.copy()
df_f_train['Local'] = df_f_train['Local'].replace({'Camilopolis': 1, 'Erasmo': 2, 'Paraiso': 3, 'RM': 4, 'Vitoria': 5})
df_f_train = df_f_train.merge(pd.get_dummies(df_f_train[['Mes', 'Local']], columns = ['Mes', 'Local'], prefix = ['Mes', 'Local']), left_index=True, right_index=True)


# In[ ]:


cols_rem = ['LocalMax_h', 'LocalMax_h_All', 'Data_Hora', 'Data'] + ['Mes', 'Local']
label_name = 'LocalMax_h'

x = df_f_train[[c for c in df_f_train.columns if c not in cols_rem]]
y = df_f_train[label_name]

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3, random_state = 378)

X = pd.concat([x_treino, y_treino], axis=1)

# separate minority and majority classes
not_ordem = X[X[label_name]==0].copy()
ordem = X[X[label_name]==1].copy()

# upsample minority
ordem_upsampled = resample(ordem,
                        replace=True, # sample with replacement
                        n_samples=len(not_ordem), # match number in majority class
                        random_state=378) # reproducible results

# combine majority and upsampled minority
upsampled = pd.concat([not_ordem, ordem_upsampled])

x_treino = upsampled[[c for c in df_f_train.columns if c not in cols_rem]]
y_treino = upsampled[label_name]

display(y_treino.value_counts())


# In[ ]:


model_f = ExtraTreesClassifier(random_state=42)

model_f.fit(x_treino, y_treino)


# In[ ]:


y_teste_pred = model_f.predict(x_teste)
y_treino_pred = model_f.predict(x_treino)

print(f"Treino: {accuracy_score(y_treino, y_treino_pred)}")
print(f"Teste: {accuracy_score(y_teste, y_teste_pred)}")
print(f"Precisão: {precision_score(y_teste, y_teste_pred)}")
print(f"Recall: {recall_score(y_teste, y_teste_pred)}")
print(f"F1: {f1_score(y_teste, y_teste_pred)}")
display(confusion_matrix(y_teste, y_teste_pred, normalize='true'))
display(confusion_matrix(y_teste, y_teste_pred,))


# In[ ]:


best_model_f.


# In[ ]:


x_treino


# In[ ]:


df_f_upsampled


# In[ ]:


df_f[(df_f['PressaoAtmosferica'] == 0)]

