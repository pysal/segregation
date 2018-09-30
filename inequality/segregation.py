"""
Segregation Indices
This is a module that calculates several different spatial (and non-spatial) segregation indices
"""

__author__ = "Sergio J. Rey <srey@asu.edu> and Renan X. Cortes <renanc@ucr.edu>"

# Dependencies
import pandas as pd
import pysal as ps
import geopandas as gpd
import scipy
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scipy.ndimage.interpolation import shift


# Function that calculates a specific an overall segregation index
def calculate_segregation(data, group_pop_var, total_pop_var):
    '''
    data: a geopandas DataFrame that contains a geometry column
    group_pop_var: the name of variable that contains the population size of the group of interest
    total_pop_var: the name of variable that contains the total population of the unit
    '''
    
    # Uneveness
    data = data.rename(columns={group_pop_var: 'group_pop_var', total_pop_var: 'total_pop_var'})
    T = data.total_pop_var.sum()
    P = data.group_pop_var.sum() / T
    data['ti'] = data.total_pop_var
    data['pi'] = data.group_pop_var / data.total_pop_var
    D = (((data.total_pop_var * abs(data.pi - P)))/ (2 * T * P * (1 - P))).sum()
    
    # Isolation
    data['xi'] = data.group_pop_var
    X = data['xi'].sum()
    xPx = ((data.xi / X) * (data.xi / data.ti)).sum()
    
    # Clustering
    data['yi'] = data.total_pop_var - data.group_pop_var
    Y = data.yi.sum()
    data['c_lons'] = data.centroid.map(lambda p: p.x)
    data['c_lats'] = data.centroid.map(lambda p: p.y)
    dist = euclidean_distances(data[['c_lons','c_lats']])
    np.fill_diagonal(dist, val = (0.6*data.area)**(1/2))
    c = np.exp(-dist)
    Pxx = ((np.array(data.xi) * c).T * np.array(data.xi)).sum() / X**2
    Pyy = ((np.array(data.yi) * c).T * np.array(data.yi)).sum() / Y**2
    RCL = Pxx / Pyy - 1

    # Concentration
    A = data.area.sum()
    DEL = 1/2 * abs(data.xi / X - data.area / A).sum()
    
    # Centralization
    data['center_lon']  = data.c_lons.mean()
    data['center_lat']  = data.c_lats.mean()
    data['center_dist'] = np.sqrt((data.c_lons - data.center_lon)**2 + (data.c_lats - data.center_lat)**2)
    data_sort_cent = data.sort_values('center_dist')
    
    data_sort_cent['Xi'] = np.cumsum(data_sort_cent.xi) / X
    data_sort_cent['Yi'] = np.cumsum(data_sort_cent.yi) / Y
    data_sort_cent['Ai'] = np.cumsum(data_sort_cent.area) / A
    
    ACE = (shift(data_sort_cent.Xi, 1, cval=np.NaN) * data_sort_cent.Ai).sum() - \
          (data_sort_cent.Xi * shift(data_sort_cent.Ai, 1, cval=np.NaN)).sum()
    
    # Aggregating
    SM = np.mean([D, xPx, RCL, DEL, ACE])
    
    return SM