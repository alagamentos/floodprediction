#!/usr/bin/env python
# coding: utf-8

# http://www3.santoandre.sp.gov.br/defesacivil/wp-content/uploads/2019/10/Setembro-2019-Cemaden.pdf
# http://www3.santoandre.sp.gov.br/defesacivil/wp-content/uploads/2019/09/Agosto-2019-Cemaden-2.pdf

# In[ ]:


import pandas as pd

ip = pd.read_csv('../../../data/cleandata/Info pluviometricas/Merged Data/merged.csv',
                 sep = ';',
                 dtype = {'Local_0': object, 'Local_1':object,
                          'Local_2':object,  'Local_3':object},
                parse_dates = ['Data_Hora'])


# In[ ]:


ip['d'], ip['m'], ip['y'], ip['min'] = ip['Data_Hora'].dt.day, ip['Data_Hora'].dt.month,                                       ip['Data_Hora'].dt.year, ip['Data_Hora'].dt.minute

local_cols = [c for c in ip.columns if 'Local' in c]
ip[local_cols].head(2)


# In[ ]:


ip.loc[ (ip['m'] == 8) & (ip['y'] == 2019) & (ip['min'] == 0) ,
       ['Data_Hora', 'Precipitacao_1', 'Precipitacao_2', 'Precipitacao_3', 'Precipitacao_4']
      ].fillna(0).sum()

