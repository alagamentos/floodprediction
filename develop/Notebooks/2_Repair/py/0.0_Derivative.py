#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

try:
    from jupyterthemes import jtplot
    jtplot.style()
except:
    pass


# In[2]:


ip = pd.read_csv('../../../data/cleandata/Info pluviometricas/Merged Data/merged.csv',
                 sep = ';',
                 dtype = {'Local_0': object, 'Local_1':object,
                          'Local_2':object,  'Local_3':object})

print(list(ip.columns))
ip.head()


# #### Umidade Relativa 

# In[3]:


cols_um = [i for i in ip.columns if 'UmidadeRelativa' in i]
um = ip[cols_um].fillna(np.nan)
um.head()


# #### Derivada

# In[4]:


d_um0 = np.gradient(um[cols_um[0]])


# In[5]:


start = 0
stop  = 1500
plt.figure(figsize =  (12,7))
ax1 = plt.subplot(211)
ax1.plot(um[cols_um[0]].index[start:stop],um[cols_um[0]][start:stop])
ax2 = plt.subplot(212)
ax2.plot(um[cols_um[0]].index[start:stop],d_um0[start:stop])
plt.show()


# #### Derivada muito alta

# In[6]:


threshold = 15
high_d = []
for i in range(len(d_um0)):
    if abs(d_um0[i]) > threshold:
        high_d.append(True)
    else:
        high_d.append(False)


# In[7]:


start = 0
stop  = 1500
plt.figure(figsize =  (12,7))
ax1 = plt.subplot(211)
ax1.plot(um[cols_um[0]].index[start:stop],um[cols_um[0]][start:stop])
ax2 = plt.subplot(212)
ax2.plot(um[cols_um[0]].index[start:stop],d_um0[start:stop])
ax2.axhline(threshold , c = 'r')
ax2.axhline(-threshold, c = 'r')
i = 0
for d in high_d: # plot vertical lines  -- cannot plot multiple lines at once
    if d and i > start and i < stop:
        ax1.axvline(i, ymin=-30, ymax=30, c = 'r', alpha = 1)
    i = i + 1
ax2.set_yticks(np.arange(-30,31,10))
ax2.set_ylim(-40,40)
ax1.set_title('Time series')
ax2.set_title('Derivative')
plt.show()


# ### Derivada zero

# In[8]:


n_zeros = 5
is_const = []
for i in range(len(d_um0)):
    aux = True
    for n in range(n_zeros):
        aux = aux and (d_um0[i + n] == 0 or d_um0[i - n] == 0)
    is_const.append(aux)


# In[9]:


start = 1000
stop  = 1500
plt.figure(figsize =  (12,7))
ax1 = plt.subplot(211)

i = 0
for d in is_const: # plot vertical lines  -- cannot plot multiple lines at once
    if d and i > start and i < stop:
        ax1.axvline(i, ymin=-30, ymax=30, c = 'g', alpha = 0.25)
    i = i + 1

ax1.plot(um[cols_um[0]].index[start:stop],um[cols_um[0]][start:stop])
ax2 = plt.subplot(212)
ax2.plot(um[cols_um[0]].index[start:stop],d_um0[start:stop])

ax2.set_yticks(np.arange(-30,31,10))
ax2.set_ylim(-40,40)
ax1.set_title('Time series')
ax2.set_title('Derivative')
plt.show()


# ### Dados com erro

# In[10]:


is_error = [is_const[i] or high_d[i] for i in range(len(high_d))]


# ### Create regions

# In[11]:


regions = []
i = 0
status = False
for bool_ in is_error:
    if bool_ and not status:
        start = i
        status = True
    if not bool_ and status:
        end = i
        status = False
        regions.append([start,end])
    i += 1


# In[12]:


is_error


# In[13]:


regions


# ### Increase Margins

# In[14]:


margin = 2
regions_marg = []
for reg in regions:
    regions_marg.append([reg[0] - margin, reg[1] + margin]) 


# ### Compare with and without margins

# In[15]:


start = 0
stop  = 1000

plt.figure(figsize =  (12,7))
ax1 = plt.subplot(211)
for reg in regions:
    if reg[0] > start and reg[1] < stop:
        ax1.axvspan(reg[0] , reg[1], color = 'red')
ax1.plot(um[cols_um[0]].index[start:stop],um[cols_um[0]][start:stop])
        
ax2 = plt.subplot(212)
for reg in regions_marg:
    if reg[0] > start and reg[1] < stop:
        ax2.axvspan(reg[0], reg[1], color = 'red')
ax2.plot(um[cols_um[0]].index[start:stop],um[cols_um[0]][start:stop])

plt.show()


# In[16]:


start = 0
stop = 1500

plt.figure(figsize =  (12,7))
plt.plot(um[cols_um[0]][start:stop].index,um[cols_um[0]][start:stop])
for reg in regions_marg:
    if reg[0] > start and reg[1] < stop:
        x = list(range(reg[0],reg[1]))
        y = um[cols_um[0]][reg[0]:reg[1]]
        plt.plot(x, y, color = 'red')
plt.show()

