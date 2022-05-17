#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import googlemaps
import pandas as pd


# In[ ]:


enchentes = pd.read_csv('../../../data/cleandata/Ordens de serviço/Enchentes - 01012010 a 30092019.csv',
                        sep = ';')

enchentes.head(2)


# In[ ]:


geocoding_API_key = ''
gmaps_geo = googlemaps.Client(key=geocoding_API_key) # Geocoding API


# In[ ]:


def distance(lat2, lon2): 
          
    from math import radians, sin, cos, asin, sqrt
    
    # The math module contains a function named 
    # radians which converts from degrees to radians. 
    lon1 = radians(-46.522956) 
    lon2 = radians(lon2) 
    lat1 = radians(-23.661983,) 
    lat2 = radians(lat2) 
       
    # Haversine formula  
    dlon = lon2 - lon1  
    dlat = lat2 - lat1 
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    
    c = 2 * asin(sqrt(a))  
    # Radius of earth in kilometers. Use 3956 for miles 
    r = 6371       
    # calculate the result 
    return(c * r) 


# In[ ]:


from sys import stdout
for index, row in enchentes.iterrows():
    stdout.write(f"\r{index}/ {len(enchentes)}")
    stdout.flush()
    try:
        adress = row['Endereco1'] + ', ' + row['Endereco2']  + ', Santo André'
        results = gmaps_geo.geocode(adress)
        
        found_SantoAndre = False
        found_SP = False

        # Get the closest to Santo Andre from all the search results
        dist_list = [distance(result['geometry']['location']['lat'],
                              result['geometry']['location']['lng']) for result in results]
        min_indx = dist_list.index(min(dist_list))

        enchentes.loc[index, 'lat'] = results[min_indx]['geometry']['location']['lat']
        enchentes.loc[index, 'lng'] = results[min_indx]['geometry']['location']['lng']
        enchentes.loc[index, 'status'] = 0
    except:
        try:
            adress = row['Endereco1'] + ', Santo André'
            results = gmaps_geo.geocode(adress)

            found_SantoAndre = False
            found_SP = False

            dist_list = [distance(result['geometry']['location']['lat'],
                                  result['geometry']['location']['lng']) for result in results]
            min_indx = dist_list.index(min(dist_list))

            enchentes.loc[index, 'lat'] = results[min_indx]['geometry']['location']['lat']
            enchentes.loc[index, 'lng'] = results[min_indx]['geometry']['location']['lng']
            enchentes.loc[index, 'status'] = 1
        except:
            try:
                adress = row['Endereco2'] + ', Santo André'
                results = gmaps_geo.geocode(adress)

                found_SantoAndre = False
                found_SP = False

                dist_list = [distance(result['geometry']['location']['lat'],
                                      result['geometry']['location']['lng']) for result in results]
                min_indx = dist_list.index(min(dist_list))

                enchentes.loc[index, 'lat'] = results[min_indx]['geometry']['location']['lat']
                enchentes.loc[index, 'lng'] = results[min_indx]['geometry']['location']['lng']
                enchentes.loc[index, 'status'] = 2
            except:
                enchentes.loc[index, 'lat'] = 'Error'
                enchentes.loc[index, 'lng'] = 'Error'
                enchentes.loc[index, 'status'] = 3
                
backup = enchentes.copy(deep = True)


# In[ ]:


for i in range(4):
    len_ = len(backup[backup['status'] == i])
    print(f'status {i}: {len_}')


# # Export

# In[ ]:


backup.to_csv(r'../../../data/cleandata/Ordens de serviço/Enchentes_LatLong.csv', sep=';', index=False)

