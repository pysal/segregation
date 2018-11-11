"""
Inference Wrappers for Segregation measures
"""

__author__ = "Renan X. Cortes <renanc@ucr.edu> and Sergio J. Rey <sergio.rey@ucr.edu>"


import numpy as np
import pandas as pd
import geopandas as gpd
import warnings

__all__ = ['Compare_Segregation']

def _compare_segregation(seg_class_1, seg_class_2, iterations = 500, null_approach = "random_data", **kwargs):
    '''
    Perform inference comparison for a two segregation measures

    Parameters
    ----------

    seg_class_1: a PySAL segregation object to be compared to seg_class_2
    
    seg_class_2: a PySAL segregation object to be compared to seg_class_1
    
    iterations: number of iterations under null hyphothesis
    
    null_approach: argument that specifies which type of null hypothesis the inference will iterate.
    
        "random_data"            : random label the data in each iteration
        
        "random_spatial"         : random label the spatial unit context (geometries) in each iteration
        
    **kwargs: customizable parameters to pass to the segregation measures. Usually they need to be the same as both seg_class_1 and seg_class_2  was built.
    
    Attributes
    ----------

    p_value        : float
                     Two-Tailed p-value
    
    est_sim        : numpy array
                     Estimates of the segregation measure differences under the null hypothesis
                  
    est_point_diff : float
                     Point estimation of the difference between the segregation measures
                
    Notes
    -----
    This function performs inference to compare two segregation measures. This can be either two measures of the same locations in two different points in time or it can be two different locations at the same point in time.
    
    The null hypothesis is H0: Segregation_1 - Segregation_2 = 0 and, therefore, the est_sim attribute must be compared to the zero value.
    
    Based on Rey, Sergio J., and Myrna L. Sastré-Gutiérrez. "Interregional inequality dynamics in Mexico." Spatial Economic Analysis 5.3 (2010): 277-298.

    '''
    
    if(type(seg_class_1) != type(seg_class_2)):
        raise TypeError('seg_class_1 and seg_class_2 must be the same type/class.')
    
    data_1 = seg_class_1.core_data
    data_2 = seg_class_2.core_data
    
    point_estimation = seg_class_1.statistic - seg_class_2.statistic
    
    _class_name = str(type(seg_class_1))
    
    # This step is just to make sure the each frequecy column is from the same type in order to stack them
    data_1['group_pop_var'] = round(data_1['group_pop_var']).astype(int)
    data_1['total_pop_var'] = round(data_1['total_pop_var']).astype(int)
    
    data_2['group_pop_var'] = round(data_2['group_pop_var']).astype(int)
    data_2['total_pop_var'] = round(data_2['total_pop_var']).astype(int)
    
    data_1['grouping_variable'] = 'Group_1'
    data_2['grouping_variable'] = 'Group_2'
    
    stacked_data = pd.concat([data_1, data_2], ignore_index=True)
    
    est_sim = np.empty(iterations)
    
    if (null_approach == "random_data"):
        
        for i in np.array(range(iterations)):
            
            aux_rand = list(np.random.choice(stacked_data.shape[0], stacked_data.shape[0], replace = False))

            stacked_data['rand_group_pop'] = stacked_data.group_pop_var[aux_rand].reset_index()['group_pop_var']
            stacked_data['rand_total_pop'] = stacked_data.total_pop_var[aux_rand].reset_index()['total_pop_var']
            
            # Dropping variable to avoid confusion in the calculate_segregation function 
            # Building auxiliar data to avoid affecting the next iteration
            stacked_data_aux = stacked_data.drop(['group_pop_var', 'total_pop_var'], axis = 1)
            
            stacked_data_1 = stacked_data_aux.loc[stacked_data_aux['grouping_variable'] == 'Group_1']
            stacked_data_2 = stacked_data_aux.loc[stacked_data_aux['grouping_variable'] == 'Group_2']

            simulations_1 = seg_class_1._function(stacked_data_1, 'rand_group_pop', 'rand_total_pop', **kwargs)[0]
            simulations_2 = seg_class_2._function(stacked_data_2, 'rand_group_pop', 'rand_total_pop', **kwargs)[0]
            
            est_sim[i] = simulations_1 - simulations_2
    
    
    if (null_approach == "random_spatial"):
        
        for i in np.array(range(iterations)):
            
            aux_rand = list(np.random.choice(stacked_data.shape[0], stacked_data.shape[0], replace = False))

            stacked_data['rand_grouping'] = stacked_data.grouping_variable[aux_rand].reset_index()['grouping_variable']
            
            stacked_data_1 = stacked_data.loc[stacked_data['rand_grouping'] == 'Group_1']
            stacked_data_2 = stacked_data.loc[stacked_data['rand_grouping'] == 'Group_2']

            simulations_1 = seg_class_1._function(stacked_data_1, 'group_pop_var', 'total_pop_var', **kwargs)[0]
            simulations_2 = seg_class_2._function(stacked_data_2, 'group_pop_var', 'total_pop_var', **kwargs)[0]
            
            est_sim[i] = simulations_1 - simulations_2

            
    # Two-Tailed p-value
    p_value = (sum(est_sim > abs(point_estimation)) + sum(est_sim < -abs(point_estimation))) / iterations
        
    return p_value, est_sim, point_estimation, _class_name



class Compare_Segregation:
    '''
    Perform inference comparison for a two segregation measures

    Parameters
    ----------

    seg_class_1: a PySAL segregation object to be compared to seg_class_2
    
    seg_class_2: a PySAL segregation object to be compared to seg_class_1
    
    iterations: number of iterations under null hyphothesis
    
    null_approach: argument that specifies which type of null hypothesis the inference will iterate.
    
        "random_data"            : random label the data in each iteration
        
        "random_spatial"         : random label the spatial unit context (geometries) in each iteration
        
    **kwargs: customizable parameters to pass to the segregation measures. Usually they need to be the same as both seg_class_1 and seg_class_2  was built.
    
    Attributes
    ----------

    p_value        : float
                     Two-Tailed p-value
    
    est_sim        : numpy array
                     Estimates of the segregation measure differences under the null hypothesis
                  
    est_point_diff : float
                     Point estimation of the difference between the segregation measures
                
    Notes
    -----
    This function performs inference to compare two segregation measures. This can be either two measures of the same locations in two different points in time or it can be two different locations at the same point in time.
    
    The null hypothesis is H0: Segregation_1 - Segregation_2 = 0 and, therefore, the est_sim attribute must be compared to the zero value.
    
    Based on Rey, Sergio J., and Myrna L. Sastré-Gutiérrez. "Interregional inequality dynamics in Mexico." Spatial Economic Analysis 5.3 (2010): 277-298.

    '''

    def __init__(self, seg_class_1, seg_class_2, iterations = 500, null_approach = "random_data", **kwargs):
        
        aux = _compare_segregation(seg_class_1, seg_class_2, iterations, null_approach, **kwargs)

        self.p_value        = aux[0]
        self.est_sim        = aux[1]
        self.est_point_diff = aux[2]
        self._class_name    = aux[3]
        
    def plot(self):
        """
        Plot the Compare_Segregation class
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            warnings.warn('This method relies on importing `matplotlib` and `seaborn`')
    
        sns.distplot(self.est_sim, 
                     hist = True, 
                     color = 'darkblue', 
                     hist_kws={'edgecolor':'black'},
                     kde_kws={'linewidth': 2})
        plt.axvline(self.est_point_diff, color = 'red')
        plt.title('{} (Diff. value = {})'.format(self._class_name, round(self.est_point_diff, 3)))
        return plt.show()