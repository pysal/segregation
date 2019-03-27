

def generate_counterfactual(df1, df2, group_share, total_population,
                            approach, percent=True):
    """Generate a counterfactual variable.

    Given two cities, generate counterfactual distributions for a variable of
    interest by simulating the variable of one city into the spatial
    structure of the other.

    Parameters
    ----------
    df1 : pd.DataFrame
        Pandas dataframe holding data for city 1
    df2 : pd.DataFrame
        Pandas dataframe holding data for city 2
    group_share : str
        The variable (present on both dataframes) for which a counterfactual
        distribution should be generated. This variable should be a fraction
        referring to a share of a certain population group (e.g. a value of 35
        or 0.35 referring to 35% white)
    total_population : str
        The variable (present on both dataframes) representing the total
        population of the area
    approach : str, ["unit_composition", "city_composition", "dual_composition"]
        Which approach to use for generating the counterfactual.
        Options include "unit_composition", "city_composition", or
        "dual_composition"
    percent : bool
        Whether the focal_variable is formatted as a percentage or whole
        number, e.g. 45 vs 0.45. This parameter defines whether the resutls
        should be normalized by dividing by 100

    Returns
    -------
    two pandas.DataFrames
        df1 and df2  with appended columns 'counterfactual_share' and
        'counterfactual_total'

    """
    df1 = df1.copy()
    df2 = df2.copy()

    if approach == 'unit_composition':

        df1['counterfactual_share'] = df1['{variable}' .format(variable=group_share)].rank(
            pct=True).apply(df2['{variable}' .format(variable=group_share)].quantile)
        df1['counterfactual_total'] = df1['counterfactual_share'] * \
            df1['{total_pop}' .format(total_pop=total_population)]

        df2['counterfactual_share'] = df2['{variable}' .format(variable=group_share)].rank(
            pct=True).apply(df1['{variable}' .format(variable=group_share)].quantile)
        df2['counterfactual_total'] = df2['counterfactual_share'] * \
            df2['{total_pop}' .format(total_pop=total_population)]
        if percent:
            df1['counterfactual_total'] = df1['counterfactual_total'] / 100
            df2['counterfactual_total'] = df2['counterfactual_total'] / 100

    elif approach == 'city_composition':

        # TODO:
        raise NotImplementedError

    elif approach == 'dual_composition':

        # TODO:
        raise NotImplementedError

    return df1, df2


def decompose_index(index1, index2,
                    counterfactual_approach='unit_composition'):
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
                              ["unit_composition", "city_composition",
                              "dual_composition"]
        The technique used to generate the counterfactual population
        distribution.

    Returns
    -------
    tuple
        (shapley spatial component, shapley attribute component)

    """
    df1 = index1.core_data.copy()
    df2 = index2.core_data.copy()

    assert index1._function == index2._function, "Segregation indices must be of the same type"

    df1['group_share'] = df1.group_pop_var / df1.total_pop_var * 100
    df2['group_share'] = df2.group_pop_var / df2.total_pop_var * 100

    df1, df2 = generate_counterfactual(df1, df2, 'group_share', 'total_pop_var',
                                        approach=counterfactual_approach)
    df1.drop(columns=['group_pop_var'], inplace=True)
    df2.drop(columns=['group_pop_var'], inplace=True)

    seg_func = index1._function

    # index for spatial 1, attribute 1
    G_S1_A1 = index1.statistic

    # index for spatial 2, attribute 2
    G_S2_A2 = index2.statistic

    # index for spatial 1 attribute 2 (counterfactual population for structure 1)
    G_S1_A2 = seg_func(df1, 'counterfactual_total', 'total_pop_var')[0]

    # index for spatial 2 attribute 1 (counterfactual population for structure 2)
    G_S2_A1 = seg_func(df2, 'counterfactual_total', 'total_pop_var')[0]

    # take the average difference in spatial structure, holding attributes constant
    C_S = 1 / 2 * (G_S1_A1 - G_S2_A1 + G_S1_A2 - G_S2_A2)

    # take the average difference in attributes, holding spatial structure constant
    C_A = 1 / 2 * (G_S1_A1 - G_S1_A2 + G_S2_A1 - G_S2_A2)

    return C_S, C_A
