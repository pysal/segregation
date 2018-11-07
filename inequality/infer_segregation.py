"""
Inference Wrappers for Segregation measures
"""

__author__ = "Renan X. Cortes <renanc@ucr.edu> and Sergio J. Rey <sergio.rey@ucr.edu>"


import numpy as np
import pandas as pd
import geopandas as gpd

__all__ = ['Infer_Segregation']

"""
Inference Wrappers for Segregation measures
"""

__author__ = "Renan X. Cortes <renanc@ucr.edu> and Sergio J. Rey <sergio.rey@ucr.edu>"


import numpy as np
import pandas as pd
import geopandas as gpd

__all__ = ['Infer_Segregation']

def _infer_segregation(seg_class, iterations = 500, null_approach = "systematic", two_tailed = True, **kwargs):
    '''
    Perform inference for a single segregation measure

    Parameters
    ----------

    seg_class     : a PySAL segregation object
    
    iterations    : number of iterations under null hyphothesis
    
    null_approach : argument that specifies which type of null hyphotesis the inference will iterate.
    
        "systematic"             : assumes that every group has the same probability with restricted conditional probabilities p_0_j = p_1_j = p_j = n_j/n (multinomial distribution).
        "eveness"                : assumes that each spatial unit has the same global probability of drawing elements from the minority group of the fixed total unit population (binomial distribution).
        
        "permutation"            : randomly allocates the units over space keeping the original values.
        
        "systematic_permutation" : assumes absence os systematic segregation and randomly allocates the units over space.
        "even_permutation"       : assumes the same global probability of drawning elements from the minority group in each spatial unit and randomly allocates the units over space.
    
    two_tailed    : boolean
                    If True, p_value is two-tailed. Otherwise, it is right one-tailed.
    
    **kwargs: customizable parameters to pass to the segregation measures. Usually they need to be the same input that the seg_class was built.
    
    Attributes
    ----------

    p_value     : float
                  Pseudo One or Two-Tailed p-value estimated from the simulations
    
    est_sim     : numpy array
                  Estimates of the segregation measure under the null hypothesis
                  
    statistic   : float
                  The point estimation of the segregation measure that is under test
                
    Notes
    -----
    The one-tailed p_value attribute might not be appropriate for some measures, as the two-tailed. Therefore, it is better to rely on the est_sim attribute.
    
    '''
    
    if (type(two_tailed) is not bool):
        raise TypeError('two_tailed is not a boolean object')
    
    point_estimation = seg_class.statistic
    data             = seg_class.core_data
    
    if (null_approach == "systematic"):
    
        data['other_group_pop'] = data['total_pop_var'] - data['group_pop_var']
        p_j = data['total_pop_var'] / data['total_pop_var'].sum()

        # Group 0: minority group
        p0_i = p_j
        n0 = data['group_pop_var'].sum()
        sim0 = np.random.multinomial(n0, p0_i, size = iterations)

        # Group 1: complement group
        p1_i = p_j
        n1 = data['other_group_pop'].sum()
        sim1 = np.random.multinomial(n1, p1_i, size = iterations)

        Estimates_Stars = np.empty(iterations)
        
        for i in np.array(range(iterations)):
            data_aux = {'simul_group': sim0[i].tolist(), 'simul_tot': (sim0[i] + sim1[i]).tolist()}
            df_aux = pd.DataFrame.from_dict(data_aux)
            
            if (str(type(data)) == '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
                df_aux = gpd.GeoDataFrame(df_aux)
                df_aux['geometry'] = data['geometry']
                
            df_aux = gpd.GeoDataFrame(df_aux)
            df_aux['geometry'] = data['geometry']
            Estimates_Stars[i] = seg_class._function(df_aux, 'simul_group', 'simul_tot', **kwargs)[0]
    
    
    if (null_approach == "eveness"):
        
        p_null = data['group_pop_var'].sum() / data['total_pop_var'].sum()
        
        Estimates_Stars = np.empty(iterations)
        
        for i in np.array(range(iterations)):
            sim = np.random.binomial(n = np.array([data['total_pop_var'].tolist()]), 
                                     p = p_null)
            data_aux = {'simul_group': sim[0], 'simul_tot': data['total_pop_var'].tolist()}
            df_aux = pd.DataFrame.from_dict(data_aux)
            
            if (str(type(data)) == '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
                df_aux = gpd.GeoDataFrame(df_aux)
                df_aux['geometry'] = data['geometry']
            
            Estimates_Stars[i] = seg_class._function(df_aux, 'simul_group', 'simul_tot', **kwargs)[0]
        
        
    if (null_approach == "permutation"):
        
        if (str(type(data)) != '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
            raise TypeError('data is not a GeoDataFrame, therefore, this null approach does not apply.')
        
        Estimates_Stars = np.empty(iterations)
        
        for i in np.array(range(iterations)):
            data = data.assign(geometry = data['geometry'][list(np.random.choice(data.shape[0], data.shape[0], replace = False))].reset_index()['geometry'])
            df_aux = data
            Estimates_Stars[i] = seg_class._function(df_aux, 'group_pop_var', 'total_pop_var', **kwargs)[0]

    
    if (null_approach == "systematic_permutation"):
        
        if (str(type(data)) != '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
            raise TypeError('data is not a GeoDataFrame, therefore, this null approach does not apply.')
    
        data['other_group_pop'] = data['total_pop_var'] - data['group_pop_var']
        p_j = data['total_pop_var'] / data['total_pop_var'].sum()

        # Group 0: minority group
        p0_i = p_j
        n0 = data['group_pop_var'].sum()
        sim0 = np.random.multinomial(n0, p0_i, size = iterations)

        # Group 1: complement group
        p1_i = p_j
        n1 = data['other_group_pop'].sum()
        sim1 = np.random.multinomial(n1, p1_i, size = iterations)

        Estimates_Stars = np.empty(iterations)
        
        for i in np.array(range(iterations)):
            data_aux = {'simul_group': sim0[i].tolist(), 'simul_tot': (sim0[i] + sim1[i]).tolist()}
            df_aux = pd.DataFrame.from_dict(data_aux)
            df_aux = gpd.GeoDataFrame(df_aux)
            df_aux['geometry'] = data['geometry']
            df_aux = df_aux.assign(geometry = df_aux['geometry'][list(np.random.choice(df_aux.shape[0], df_aux.shape[0], replace = False))].reset_index()['geometry'])
            Estimates_Stars[i] = seg_class._function(df_aux, 'simul_group', 'simul_tot', **kwargs)[0]  
    
    
    if (null_approach == "even_permutation"):
        
        if (str(type(data)) != '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
            raise TypeError('data is not a GeoDataFrame, therefore, this null approach does not apply.')
        
        p_null = data['group_pop_var'].sum() / data['total_pop_var'].sum()
        
        Estimates_Stars = np.empty(iterations)
        
        for i in np.array(range(iterations)):
            sim = np.random.binomial(n = np.array([data['total_pop_var'].tolist()]), 
                                     p = p_null)
            data_aux = {'simul_group': sim[0], 'simul_tot': data['total_pop_var'].tolist()}
            df_aux = pd.DataFrame.from_dict(data_aux)
            df_aux = gpd.GeoDataFrame(df_aux)
            df_aux['geometry'] = data['geometry']
            df_aux = df_aux.assign(geometry = df_aux['geometry'][list(np.random.choice(df_aux.shape[0], df_aux.shape[0], replace = False))].reset_index()['geometry'])
            Estimates_Stars[i] = seg_class._function(df_aux, 'simul_group', 'simul_tot', **kwargs)[0]

    
    if not two_tailed:
        p_value = sum(Estimates_Stars > point_estimation) / iterations
    else:
        p_value = (sum(Estimates_Stars > abs(point_estimation)) + sum(Estimates_Stars < -abs(point_estimation))) / iterations
        
    return p_value, Estimates_Stars, point_estimation



class Infer_Segregation:
    '''
    Perform inference for a single segregation measure

    Parameters
    ----------

    seg_class     : a PySAL segregation object
    
    iterations    : number of iterations under null hyphothesis
    
    null_approach : argument that specifies which type of null hyphotesis the inference will iterate.
    
        "systematic"             : assumes that every group has the same probability with restricted conditional probabilities p_0_j = p_1_j = p_j = n_j/n (multinomial distribution).
        "eveness"                : assumes that each spatial unit has the same global probability of drawing elements from the minority group of the fixed total unit population (binomial distribution).
        
        "permutation"            : randomly allocates the units over space keeping the original values.
        
        "systematic_permutation" : assumes absence os systematic segregation and randomly allocates the units over space.
        "even_permutation"       : assumes the same global probability of drawning elements from the minority group in each spatial unit and randomly allocates the units over space.
    
    two_tailed    : boolean
                    If True, p_value is two-tailed. Otherwise, it is right one-tailed.
    
    **kwargs: customizable parameters to pass to the segregation measures. Usually they need to be the same input that the seg_class was built.
    
    Attributes
    ----------

    p_value     : float
                  Pseudo One or Two-Tailed p-value estimated from the simulations
    
    est_sim     : numpy array
                  Estimates of the segregation measure under the null hypothesis
                  
    statistic   : float
                  The point estimation of the segregation measure that is under test
                
    Notes
    -----
    The one-tailed p_value attribute might not be appropriate for some measures, as the two-tailed. Therefore, it is better to rely on the est_sim attribute.
    
    '''

    def __init__(self, seg_class, iterations = 500, null_approach = "systematic", two_tailed = True, **kwargs):
        
        aux = _infer_segregation(seg_class, iterations, null_approach, two_tailed, **kwargs)

        self.p_value   = aux[0]
        self.est_sim   = aux[1]
        self.statistic = aux[2]