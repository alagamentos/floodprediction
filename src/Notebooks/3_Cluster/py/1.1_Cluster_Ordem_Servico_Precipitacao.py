#!/usr/bin/env python
# coding: utf-8

# In[1]:


path = 'https://raw.githubusercontent.com/tbrugz/geodata-br/master/geojson/geojs-35-mun.json'

from urllib.request import urlopen
import json
with urlopen(path) as response:
    counties = json.load(response)
    
SA = [ i for i in counties['features'] if i['properties']['name'] == 'Santo André' ][0]


# In[2]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from plotly import graph_objects as go
import plotly as py

py.offline.init_notebook_mode()


# In[3]:


df = pd.read_csv('../../../data/cleandata/Ordens de serviço/Enchentes_LatLong.csv',
                 sep = ';')

est = pd.read_csv('../../../data/cleandata/Estacoes/lat_lng_estacoes.csv', sep = ';')



# In[4]:


def Calculate_Dist(lat1, lon1, lat2, lon2):
    r = 6371
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2)**2 + np.cos(phi1) *        np.cos(phi2) *   np.sin(delta_lambda / 2)**2
    res = r * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))
    return np.round(res, 2)

def get_distances(estacoes, ord_serv):
    for index, row in ord_serv.iterrows():
        dist = estacoes.apply(lambda x: 
                           Calculate_Dist(row['lat'], row['lng'],
                                          x['lat'],   x['lng']),
                           axis=1)
        ord_serv.loc[index,'Distance'], arg = dist.min(), dist.argmin()
        ord_serv.loc[index,'Est. Prox'] = estacoes.iloc[arg,0]

    return ord_serv


# In[5]:


ord_serv = get_distances(est, df)
ord_serv.loc[ord_serv['Distance'] > 4.5, 'Est. Prox'] = 'Null'


# In[6]:


fig = go.Figure()

colors = dict(zip(ord_serv['Est. Prox'].unique(),
                  ['black', 'green', 'yellow', 'teal', 'orange', 'blue']) )

fig.add_trace(go.Scatter(x=ord_serv['lng'],
                         y= ord_serv['lat'],
                         marker=dict(
                                    size=7,
                                    color=ord_serv['Est. Prox'].apply(lambda x: colors[x]), #set color equal to a variable
                                    showscale=False
                                ),
                    showlegend = False,
                    mode='markers',
                    name='markers'))

fig.add_trace(go.Scatter(x = est['lng'],
                         y = est['lat'],
                         marker_symbol = 'x',
                         marker=dict(
                                    size=10,
                                    color='red', #set color equal to a variable
                                    showscale=False
                                ),
                    showlegend = False,
                    mode='markers',
                    name='markers'))

fig.show()


# In[7]:


ord_serv = ord_serv[['lat','lng','Data', 'Est. Prox']]
ord_serv.loc[:,'Data'] = pd.to_datetime(ord_serv.loc[:,'Data'])
ord_serv = ord_serv.sort_values('Data')

ord_serv['pos'] = ord_serv['lat'].astype(str).str.rstrip() +                   ord_serv['lng'].astype(str).str.rstrip() 

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(ord_serv['pos'])
ord_serv['pos'] = le.transform(ord_serv['pos'])


# In[27]:


my_index = np.sort(ord_serv['pos'].unique())
my_cols = ord_serv.Data.dt.strftime('%Y-%m-%d').unique()

df = pd.DataFrame(columns=list(my_cols), index = list(my_index))
df.loc[:,:] = 0


# In[28]:


from datetime import datetime
from datetime import timedelta

day_delta = 4
for d in df.columns:

    lim_dates = [datetime.strptime(d, '%Y-%m-%d') + timedelta(days=-day_delta),
                 datetime.strptime(d, '%Y-%m-%d') + timedelta(days=day_delta)]

    selected_dates = ord_serv[(ord_serv['Data'] > lim_dates[0]) &
                        (ord_serv['Data'] <= lim_dates[1])]
        
    selected = selected_dates.pos
    df.loc[df.index.isin(selected),d] = 1


# In[29]:


ord_serv.head(1)


# In[30]:


df.head(1)


# In[45]:


my_map = dict(zip(ord_serv['pos'], ord_serv['Est. Prox']))
df['Estacao'] = df.index.map(my_map)
df = df[~(df['Estacao'] == 'Null')]

df_est = pd.DataFrame(columns=list(my_cols))

for est in ord_serv['Est. Prox'].unique():
    df_est.loc[est,:] =  df[df['Estacao'] == est].drop(columns = ['Estacao']).sum(axis = 0)
    


# In[59]:


df_plot


# In[61]:


import plotly.express as px

df_plot = df_est.T
df_plot['Date'] = df_plot.index
df_plot['Date'] = pd.to_datetime(df_plot['Date'])

fig = px.line(df_plot, x="Date", y=list(df_plot.columns)[:-1],
              title='Ordens de Serviço')
fig.show()


# In[67]:


df_est


# ### Incluir Precipitação para cada dia relativo a cada estação!
# 
# \\/ \\/ \\/ \\/ \\/ \\/ \\/ \\/ \\/ \\/ \\/ \\/ \\/ \\/ \\/ \\/ \\/ \\/ \\/ \\/ \\/ \\/ \\/ \\/ \\/ \\/ \\/ \\/ \\/ \\/ \\/ \\/ \\/ 

# In[ ]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

X = df.values

range_n_clusters = [2, 3, 4, 5, 6]

inertias = []

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)
    
    inertias.append(clusterer.inertia_)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values =             sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

plt.show()


# In[ ]:


plt.plot(range_n_clusters, inertias, '-o', color='black')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(range_n_clusters)
plt.show()


# In[ ]:


plt.plot(range_n_clusters, s_value)


# In[ ]:


ord_serv[ord_serv['cluster'] == 0].shape


# In[ ]:


ord_serv['cluster'] = ord_serv['epos'].map(dict(zip(df.index,cluster_labels)))


# In[ ]:


ord_serv.Data


# In[ ]:


import plotly.express as px

ord_serv['str_Data'] = ord_serv.Data.dt.strftime('%Y-%m-%d')

cluster_ord_serv = ord_serv#[ord_serv['cluster'] == 0]

fig = px.scatter(cluster_ord_serv, x="lng", y="lat", color='cluster', hover_data= ['str_Data']
                )
fig.update_traces(selector={'name':'Europe'}) 
fig.show()


# In[ ]:


fig = go.Figure()

colors = dict(zip(list(range(len(ord_serv.cluster.unique())+ 1 )),
              ['black', 'blue', 'teal', 'green', 'yellow', 'red' ]))

fig.add_trace(go.Scatter(x=ord_serv['lng'],
                         y= ord_serv['lat'],
                         marker=dict(
                                    size=7,
                                    color=ord_serv['cluster'].apply(lambda x: colors[x]), #set color equal to a variable
                                    showscale=False
                                ),
                    showlegend = True,
                    mode='markers',
                    name='markers'))



fig.show()

