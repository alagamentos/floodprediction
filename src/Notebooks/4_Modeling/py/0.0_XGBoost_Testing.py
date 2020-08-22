#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from plotly import graph_objects as go
import plotly as py

from datetime import datetime
from datetime import timedelta

import xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, accuracy_score, f1_score


# In[ ]:


df_cluster = pd.read_csv('../../../data/cleandata/Info pluviometricas/Merged Data/clustered.csv', sep = ';')
df_cluster


# In[ ]:


xgb = xgboost.XGBClassifier()



x = df_cluster[[c for c in df_cluster.columns if 'Ordens' not in c]]
y = df_cluster['Ordens']

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3, random_state = 378)

#xgb.fit(x_treino, y_treino, eval_set = [(x_treino, y_treino), (x_teste, y_teste)], eval_metric=f1_score)
param = {'max_depth':2, 'eta':1, 'objective':'binary:logistic'}

df_train = xgboost.DMatrix(data=x_treino, label=y_treino)

bst = xgboost.train(param, df_train, 2, feval=f1_score)
y_pred = bst.predict(xgboost.DMatrix(data=x_teste, label=y_teste))


# In[ ]:


y_pred = [1 if i>0.5 else 0 for i in y_pred]


# In[ ]:


print(accuracy_score(y_teste, y_pred))
print(f1_score(y_teste, y_pred))


# In[ ]:




