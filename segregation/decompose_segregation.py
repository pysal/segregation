"""
Decomposition Segregation based Metrics
"""

__author__ = "Renan X. Cortes <renanc@ucr.edu>, Elijah Knaap <elijah.knaap@ucr.edu>, and Sergio J. Rey <sergio.rey@ucr.edu>"

import numpy as np
from segregation.util import _generate_counterfactual

def _decompose_segregation(index1, 
                           index2,
                           counterfactual_approach = 'composition'):
    
    """Decompose segregation differences into spatial and attribute components.

    Given two segregation indices of the same type, use Shapley decomposition
    to measure whether the differences between index measures arise from
    differences in spatial structure or population structure

    Parameters
    ----------
    index1 : segregation.SegIndex class
        First SegIndex class to compare.
    index2 : segregation.SegIndex class
        Second SegIndex class to compare.
    counterfactual_approach : str, one of
                              ["composition", "share", "dual_composition"]
        The technique used to generate the counterfactual population
        distributions.

    Returns
    -------
    tuple
        (shapley spatial component, shapley attribute component)

    """
    df1 = index1.core_data.copy()
    df2 = index2.core_data.copy()

    assert index1._function == index2._function, "Segregation indices must be of the same type"

    counterfac_df1, counterfac_df2 = _generate_counterfactual(df1, df2, 'group_pop_var', 'total_pop_var',
                                                              counterfactual_approach = counterfactual_approach)

    seg_func = index1._function

    # index for spatial 1, attribute 1
    G_S1_A1 = index1.statistic

    # index for spatial 2, attribute 2
    G_S2_A2 = index2.statistic

    # index for spatial 1 attribute 2 (counterfactual population for structure 1)
    G_S1_A2 = seg_func(counterfac_df1, 'counterfactual_group_pop', 'counterfactual_total_pop')[0]

    # index for spatial 2 attribute 1 (counterfactual population for structure 2)
    G_S2_A1 = seg_func(counterfac_df2, 'counterfactual_group_pop', 'counterfactual_total_pop')[0]

    # take the average difference in spatial structure, holding attributes constant
    C_S = 1 / 2 * (G_S1_A1 - G_S2_A1 + G_S1_A2 - G_S2_A2)

    # take the average difference in attributes, holding spatial structure constant
    C_A = 1 / 2 * (G_S1_A1 - G_S1_A2 + G_S2_A1 - G_S2_A2)

    return C_S, C_A, df1, df2, counterfac_df1, counterfac_df2



class Decompose_Segregation:

    def __init__(self, index1, index2, counterfactual_approach = 'composition'):
        
        aux = _decompose_segregation(index1, index2, counterfactual_approach)

        self.c_s             = aux[0]
        self.c_a             = aux[1]
        self._df1            = aux[2]
        self._df2            = aux[3]
        self._counterfac_df1 = aux[4]
        self._counterfac_df2 = aux[5]