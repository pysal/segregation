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


# Function that calculates several segregation measures and an overall segregation index
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
    data = data.assign(xi = data.group_pop_var,
                       yi = data.total_pop_var - data.group_pop_var,
                       ti = data.total_pop_var,
                       pi = data.group_pop_var / data.total_pop_var)
    D = (((data.total_pop_var * abs(data.pi - P)))/ (2 * T * P * (1 - P))).sum()
    
    
    num = (np.matmul(np.array(data.ti)[np.newaxis].T, np.array(data.ti)[np.newaxis]) * abs(np.array(data.pi)[np.newaxis].T - np.array(data.pi)[np.newaxis])).sum()
    den = (2 * T**2 * P * (1-P))
    G = num / den
    G
    
    E = P * np.log(1 / P) + (1 - P) * np.log(1 / (1 - P))
    Ei = data.pi * np.log(1 / data.pi) + (1 - data.pi) * np.log(1 / (1 - data.pi))
    H = (data.ti * (E - Ei) / (E * T)).sum()
    H
    
    # Isolation
    X = data.xi.sum()
    Y = data.yi.sum()
    xPx = ((data.xi / X) * (data.xi / data.ti)).sum()
    xPy = ((data.xi / X) * (data.yi / data.ti)).sum()
    yPy = ((data.yi / Y) * (data.yi / data.ti)).sum()
    
    CISO = xPx - X/T
    RI = xPx/yPy
    
    # Clustering
    data = data.assign(c_lons = data.centroid.map(lambda p: p.x),
                       c_lats = data.centroid.map(lambda p: p.y))
    dist = euclidean_distances(data[['c_lons','c_lats']])
    np.fill_diagonal(dist, val = (0.6*data.area)**(1/2))
    c = np.exp(-dist)
    Pxx = ((np.array(data.xi) * c).T * np.array(data.xi)).sum() / X**2
    Pyy = ((np.array(data.yi) * c).T * np.array(data.yi)).sum() / Y**2
    Ptt = ((np.array(data.ti) * c).T * np.array(data.ti)).sum() / T**2
    
    SP = (X*Pxx + Y*Pyy)/(T*Ptt)
    RCL = Pxx / Pyy - 1

    # Concentration
    A = data.area.sum()
    DEL = 1/2 * abs(data.xi / X - data.area / A).sum()
    
    df_mp_sort_area = data
    df_mp_sort_area = df_mp_sort_area.assign(area = df_mp_sort_area.area)
    df_mp_sort_area_asc = df_mp_sort_area.sort_values('area')
    n1 = np.where(((np.cumsum(df_mp_sort_area_asc.ti) / T) < X/T) == False)[0][0]
    
    df_mp_sort_area_des = df_mp_sort_area.sort_values('area', ascending=False)
    n2 = np.where(((np.cumsum(df_mp_sort_area_des.ti) / T) < X/T) == False)[0][0]
    
    n = df_mp_sort_area_asc.shape[0]
    T1 =  df_mp_sort_area_asc.ti[0:(n1+1)].sum()
    T2 =  df_mp_sort_area_asc.ti[n2:n].sum()

    RCO = ((((df_mp_sort_area_asc.xi*df_mp_sort_area_asc.area/X).sum()) / ((df_mp_sort_area_asc.yi*df_mp_sort_area_asc.area/Y).sum())) - 1) / \
          ((((df_mp_sort_area_asc.ti*df_mp_sort_area_asc.area)[0:(n1+1)].sum() / T1) / ((df_mp_sort_area_asc.ti*df_mp_sort_area_asc.area)[n2:n].sum() / T2)) - 1)


    
    # Centralization
    data = data.assign(center_lon = data.c_lons.mean(),
                       center_lat = data.c_lats.mean())
    data['center_dist'] = np.sqrt((data.c_lons - data.center_lon)**2 + (data.c_lats - data.center_lat)**2)
    data_sort_cent = data.sort_values('center_dist')
    
    data_sort_cent = data_sort_cent.assign(Xi = np.cumsum(data_sort_cent.xi) / X,
                                           Yi = np.cumsum(data_sort_cent.yi) / Y,
                                           Ai = np.cumsum(data_sort_cent.area) / A)
    
    ACE = (shift(data_sort_cent.Xi, 1, cval=np.NaN) * data_sort_cent.Ai).sum() - \
          (data_sort_cent.Xi * shift(data_sort_cent.Ai, 1, cval=np.NaN)).sum()
    
    RCE = (shift(data_sort_cent.Xi, 1, cval=np.NaN) * data_sort_cent.Yi).sum() - \
          (data_sort_cent.Xi * shift(data_sort_cent.Yi, 1, cval=np.NaN)).sum()
    
    # Aggregating
    SM = np.mean([D, CISO, RCL, RCO, RCE])
    
    return {'Dissimilarity (D)': D, 
            'Gini (G)': G,
            'Entropy (H)': H,
            
            'Isolation (xPx)': xPx, 
            'Exposure (xPy)': xPy, 
            'Centralized Isolation (CISO)': CISO,
            'Relative Isolation (RI)': RI,
            
            'Spatial Proximity (clustering) (SP)': SP, 
            'Relative Clustering (RCL)': RCL, 
            
            'Delta (concentration) (DEL)': DEL,
            'Relative Concentration (RCO)': RCO,
            
            'Absolute Centralization (ACE)': ACE,
            'Relative Centralization (RCE)': RCE,
            
            'Overall Segregation Measure (SM)': SM
           }
		   
		   
		   
		   
# Function that calculates only an overall segregation measure		   
def calculate_only_segregation(data, group_pop_var, total_pop_var):
    '''
    data: a geopandas DataFrame that contains a geometry column
    group_pop_var: the name of variable that contains the population size of the group of interest
    total_pop_var: the name of variable that contains the total population of the unit
    '''
    
    # Uneveness
    data = data.rename(columns={group_pop_var: 'group_pop_var', total_pop_var: 'total_pop_var'})
    data = data[['geometry', 'group_pop_var', 'total_pop_var']]
    T = data.total_pop_var.sum()
    P = data.group_pop_var.sum() / T
    data = data.assign(xi = data.group_pop_var,
                       yi = data.total_pop_var - data.group_pop_var,
                       ti = data.total_pop_var,
                       pi = data.group_pop_var / data.total_pop_var)
    D = (((data.total_pop_var * abs(data.pi - P)))/ (2 * T * P * (1 - P))).sum()
    
    # Isolation
    X = data['xi'].sum()
    xPx = ((data.xi / X) * (data.xi / data.ti)).sum()
    
    CISO = xPx - X/T
    
    # Clustering
    Y = data.yi.sum()
    data = data.assign(c_lons = data.centroid.map(lambda p: p.x),
                       c_lats = data.centroid.map(lambda p: p.y))
    dist = euclidean_distances(data[['c_lons','c_lats']])
    np.fill_diagonal(dist, val = (0.6*data.area)**(1/2))
    c = np.exp(-dist)
    Pxx = ((np.array(data.xi) * c).T * np.array(data.xi)).sum() / X**2
    Pyy = ((np.array(data.yi) * c).T * np.array(data.yi)).sum() / Y**2

    RCL = Pxx / Pyy - 1

    # Concentration   
    df_mp_sort_area = data
    df_mp_sort_area = df_mp_sort_area.assign(area = df_mp_sort_area.area)
    df_mp_sort_area_asc = df_mp_sort_area.sort_values('area')
    n1 = np.where(((np.cumsum(df_mp_sort_area_asc.ti) / T) < X/T) == False)[0][0]
    
    df_mp_sort_area_des = df_mp_sort_area.sort_values('area', ascending=False)
    n2 = np.where(((np.cumsum(df_mp_sort_area_des.ti) / T) < X/T) == False)[0][0]
    
    n = df_mp_sort_area_asc.shape[0]
    T1 =  df_mp_sort_area_asc.ti[0:(n1+1)].sum()
    T2 =  df_mp_sort_area_asc.ti[n2:n].sum()

    RCO = ((((df_mp_sort_area_asc.xi*df_mp_sort_area_asc.area/X).sum()) / ((df_mp_sort_area_asc.yi*df_mp_sort_area_asc.area/Y).sum())) - 1) / \
          ((((df_mp_sort_area_asc.ti*df_mp_sort_area_asc.area)[0:(n1+1)].sum() / T1) / ((df_mp_sort_area_asc.ti*df_mp_sort_area_asc.area)[n2:n].sum() / T2)) - 1)

    
    # Centralization
    data = data.assign(center_lon = data.c_lons.mean(),
                       center_lat = data.c_lats.mean())
    data['center_dist'] = np.sqrt((data.c_lons - data.center_lon)**2 + (data.c_lats - data.center_lat)**2)
    data_sort_cent = data.sort_values('center_dist')
    
    data_sort_cent = data_sort_cent.assign(Xi = np.cumsum(data_sort_cent.xi) / X,
                                           Yi = np.cumsum(data_sort_cent.yi) / Y)
    
    RCE = (shift(data_sort_cent.Xi, 1, cval=np.NaN) * data_sort_cent.Yi).sum() - \
          (data_sort_cent.Xi * shift(data_sort_cent.Yi, 1, cval=np.NaN)).sum()
    
    # Aggregating
    SM = np.mean([D, CISO, RCL, RCO, RCE])
    
    return SM
	
	
	
	
# Function that calculates the overall segregation measure under the null hyphotesis multiple times
def infer_segregation(data, 
                      group_pop_var, 
                      total_pop_var, 
                      iterations = 1000,
                      null_approach = "eveness"):
    '''
    data: a geopandas DataFrame that contains a geometry column
    group_pop_var: the name of variable that contains the population size of the group of interest
    total_pop_var: the name of variable that contains the total population of the unit
    iterations: number of iterations to compute inference on pseudo p-values
    null_approach: argument that specifies which type of null hyphotesis the inference will iterate. 
        "eveness": establish that each spatial unit would have the same global probability of drawing elements from the minority group of the fixed total unit population. 
        "permutation": randomly allocates the units over space keeping the sample values fixed.
        "even_permutation": randomly allocates the units over space and assuming the same global probability of drawning elements from the minority group in each spatial unit.
    '''
    
    data = data.rename(columns={group_pop_var: 'group_pop_var', total_pop_var: 'total_pop_var'})
    p_null = data.group_pop_var.sum() / data.total_pop_var.sum()
    
    Ds    = []
    Gs    = []
    Hs    = []
    
    xPxs  = []
    xPys  = []
    CISOs = []
    RIs   = []
    
    SPs   = []
    RCLs  = []
    
    DELs  = []
    RCOs  = []
    
    ACEs  = []
    RCEs  = []
    
    SMs   = []
    
    if (null_approach == "eveness"):
        for i in range(iterations):
            
            freq_sim = np.random.binomial(n = np.array([data.total_pop_var.tolist()]), 
                                          p = np.array([[p_null]*data.shape[0]]), 
                                          size = (1, data.shape[0])).tolist()[0]
            data = data.assign(group_pop_var = freq_sim)
            test = calculate_segregation(data, 'group_pop_var', 'total_pop_var')
            Ds.append(list(test.values())[0])
            Gs.append(list(test.values())[1])
            Hs.append(list(test.values())[2])

            xPxs.append(list(test.values())[3])
            xPys.append(list(test.values())[4])
            CISOs.append(list(test.values())[5])
            RIs.append(list(test.values())[6])

            SPs.append(list(test.values())[7])
            RCLs.append(list(test.values())[8])

            DELs.append(list(test.values())[9])
            RCOs.append(list(test.values())[10])

            ACEs.append(list(test.values())[11])
            RCEs.append(list(test.values())[12])

            SMs.append(list(test.values())[13])
            

    if (null_approach == "permutation"):
        for i in range(iterations):

            data = data.assign(geometry = data.geometry[list(np.random.choice(data.shape[0], data.shape[0], replace = False))].reset_index()['geometry'])
            test = calculate_segregation(data, 'group_pop_var', 'total_pop_var')
            Ds.append(list(test.values())[0])
            Gs.append(list(test.values())[1])
            Hs.append(list(test.values())[2])

            xPxs.append(list(test.values())[3])
            xPys.append(list(test.values())[4])
            CISOs.append(list(test.values())[5])
            RIs.append(list(test.values())[6])

            SPs.append(list(test.values())[7])
            RCLs.append(list(test.values())[8])

            DELs.append(list(test.values())[9])
            RCOs.append(list(test.values())[10])

            ACEs.append(list(test.values())[11])
            RCEs.append(list(test.values())[12])

            SMs.append(list(test.values())[13])
            
    if (null_approach == "even_permutation"):
        for i in range(iterations):
            
            freq_sim = np.random.binomial(n = np.array([data.total_pop_var.tolist()]), 
                                          p = np.array([[p_null]*data.shape[0]]), 
                                          size = (1, data.shape[0])).tolist()[0]
            data = data.assign(group_pop_var = freq_sim,
                               geometry = data.geometry[list(np.random.choice(data.shape[0], data.shape[0], replace = False))].reset_index()['geometry'])
            test = calculate_segregation(data, 'group_pop_var', 'total_pop_var')
            Ds.append(list(test.values())[0])
            Gs.append(list(test.values())[1])
            Hs.append(list(test.values())[2])

            xPxs.append(list(test.values())[3])
            xPys.append(list(test.values())[4])
            CISOs.append(list(test.values())[5])
            RIs.append(list(test.values())[6])

            SPs.append(list(test.values())[7])
            RCLs.append(list(test.values())[8])

            DELs.append(list(test.values())[9])
            RCOs.append(list(test.values())[10])

            ACEs.append(list(test.values())[11])
            RCEs.append(list(test.values())[12])

            SMs.append(list(test.values())[13])
    
    
    return Ds, Gs, Hs, xPxs, xPys, CISOs, RIs, SPs, RCLs, DELs, RCOs, ACEs, RCEs, SMs
    