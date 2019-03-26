

def generate_counterfactual(df1, df2, focal_variable, population_variable, approach, percent=True):
    """Generate a counterfactual variable.

    Generate counterfactual distributions for a variable of interest in two
    cities by simulating the attributes of one city into the spatial structure
    of the other.

    Parameters
    ----------
    df1 : pd.DataFrame
        Pandas dataframe holding data for city 1
    df2 : pd.DataFrame
        Pandas dataframe holding data for city 2
    focal_variable : pd.Series
        The variable (present on both dataframes) for which a counterfactual
        distribution should be generated
    population_variable : pd.Series
        Description of parameter `population_variable`.
    approach : str
        Which approach to use for generating the counterfactual.
        Options include "unit_share", "city_share", or "dual_share"
    percent : bool
        Whether the focal_variable is formatted as a percentage or whole
        number, e.g. 45 vs 0.45. This parameter defines whether the resutls
        should be normalized by dividing by 100

    Returns
    -------
    two pandas.DataFrames
        df1 dataframe with 'counterfactual_share' and 'counterfactual_total' columns appended
        df2 dataframe with 'counterfactual_share' and 'counterfactual_total' columns appended

    """
    df1 = df1.copy()
    df2 = df2.copy()

    if approach == 'unit_composition':

        df1['counterfactual_share'] = df1['{variable}' .format(variable=focal_variable)].rank(
            pct=True).apply(df2['{variable}' .format(variable=focal_variable)].quantile)
        df1['counterfactual_total'] = df1['counterfactual_share'] * \
            df1[{population_variable} .format(total_pop=population_variable)]
        if percent:
            df1['counterfactual_total'] = df1['counterfactual_total'] / 100

        df2['counterfactual_share'] = df2['{variable}' .format(variable=focal_variable)].rank(
            pct=True).apply(df1['{variable}' .format(variable=focal_variable)].quantile)
        df2['counterfactual_total'] = df2['counterfactual_share'] * \
            df2['{total_pop}' .format(total_pop=population_variable)]
        if percent:
            df2['counterfactual_total'] = df2['counterfactual_total'] / 100


    elif approach == 'city_share':
        ## TODO:


    elif approach == 'dual_share':

        ## TODO:

    return df1, df2
