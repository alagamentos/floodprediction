#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


merged = pd.read_csv('../../../data/cleandata/Info pluviometricas/Merged Data/merged.csv', sep = ';')


# In[ ]:


merged['Data_Hora'] = pd.to_datetime(merged['Data_Hora'], yearfirst= True).sort_values(ascending =  True)


# In[ ]:


start, stop = merged['Data_Hora'].iloc[0], merged['Data_Hora'].iloc[-1]


# In[ ]:


def days_hours_minutes(td):
    return int(td.days), td.seconds//3600, (td.seconds//60)%60

days_hours_minutes(stop - start)


# In[ ]:


from datetime import date, timedelta
# Criar Vetor de data (15 em 15 minutos )

d,h,m = days_hours_minutes(stop - start)
total_days = d + h/24 + m/24/60 + (1 / 24 / 4)

date_vec= [start + timedelta(x) for x in 
          np.arange(0, total_days, 1 / 24 / 4)]

# remover do vetor de 15 em 15 as amostras existentes em merged
missing = list(set(date_vec) - set(merged['Data_Hora']))
print('Amostras Faltantes:', len(missing),
      '\nTotal de amostras (info pluviometrico):', len(date_vec),
      '\nDeveria ser:',len(merged['Data_Hora']))


# In[ ]:


new_df = pd.DataFrame(date_vec, columns=['Data_Hora'])
new_df['Data_Hora'] = pd.to_datetime(new_df['Data_Hora'], yearfirst=True)


# In[ ]:


merged = new_df.merge(merged, how = 'left', on = 'Data_Hora')


# In[ ]:


list(merged[merged['Local_0'].isna()].Data_Hora.unique())


# In[ ]:


local_cols = [col for col in merged.columns if 'Local' in col]
for col in local_cols:
    merged.loc[:,col] = merged[col].dropna().unique()

