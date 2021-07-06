"""Tools for simulating spatial population distributions."""

import itertools

import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from joblib import Parallel, delayed
import multiprocessing


def _generate_estimate(input):
    if input[0].index_type=='singlegroup':
        df = input[1](
            input[0].data.copy(), group=input[0].group_pop_var, total=input[0].total_pop_var,
        )
        estimate = input[0]._function(df, input[0].group_pop_var, input[0].total_pop_var)
    else:
        df = input[1](
            input[0].data.copy(), groups=input[0].groups,
        )
        estimate = input[0]._function(df, input[0].groups)
    return estimate[0]


def simulate_null(
    iterations=1000, sim_func=None, seg_func=None, n_jobs=-1, segfunc_args=None, backend='threading'
):
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()    
    estimates= Parallel(n_jobs=n_jobs, backend=backend)(delayed(_generate_estimate)((seg_func, sim_func)) for i in tqdm(range(iterations)))
    return estimates

def simulate_reallocation_slow(df, group=None, total=None, groups=None):
    df = df.copy()
    df = df.reset_index(drop=True)
    # shuffle the spatial units so we allocate at random
    units = df.index.copy().values
    np.random.shuffle(units)
    units = list(units)
    if group:
        df[group] = df[group].astype(int)
        df[total] = df[total].astype(int)
        df["other"] = df[total] - df[group]
        # create a vector of "persons"
        g1_pop = [group for n in range(0, df[group].sum() + 1)]
        g2_pop = ["other" for n in range(0, df["other"].sum() + 1)]
        population = pd.DataFrame()
        population["pop"] = pd.Series(g1_pop + g2_pop)
        # set of indices we'll use to choose from
        pop_idx = population.index.to_series()

        with tqdm(total=len(units)) as pbar:
            while units:
                unit = units[0]
                # use the total population of selected unit to draw indices from the population
                unit_pop = np.random.choice(
                    pop_idx.values, df.loc[unit, total], replace=False
                )
                # assign the unit ID to each chosen person
                population.loc[unit_pop, "id"] = unit

                # remove indices so they can't be chosen again
                units.pop(0)
                pop_idx = pop_idx[~pop_idx.index.isin(unit_pop)]
                pbar.update(1)
        output = population.groupby("id")["pop"].value_counts().unstack()
        output.columns.name = None
        output = df[[df.geometry.name]].join(output)
        return output


def simulate_reallocation(df, group=None, total=None, groups=None):
    """Simulate the random reallocation of existing population groups across spatial units.

    Parameters
    ----------
    df : geopandas.GeoDataFrame
        geodataframe with population data to be randomized
    group : str, optional
        name of column on geodataframe that holds the group total
        (for use with single group indices)
    total : str, optional
        name of column on geodataframe that holds the total population for
        each unit (for use with single group indices)
    groups : list, optional
        list of columns on inut dataframe that hold total population counts
        for each group of interest

    Returns
    -------
    geopandas.GeoDataFrame
        geodataframe with randomly reallocated population

    Notes
    -------
    Simulates the random redistribution of the existing population. Given a pool
    of the total population in the region, randomly allocate each person to a
    geographic unit, subject to the total capacity of each unit. Results are
    guaranteed to respect regional and local totals for geographic units as well
    as regional totals and relative shares for groups
    """
    df = df.copy()
    # ensure we have a coilumn named "index"
    df = df.reset_index(drop=True)
    df = df.reset_index()
    geoms = df[[df.geometry.name]]
    if not total:
        total = "total"
        df["total"] = df[groups].sum(axis=1).astype(int)
    df = df[df[total] > 0]
    if group:
        df[total] = df[total].astype(int)
        df["other"] = df[total] - df[group]
        groups = [group, "other"]

    # create a list of group membership for each person
    members = [[group for i in range(df[group].sum().astype(int))] for group in groups]
    pop_groups = list(itertools.chain.from_iterable(members))

    # create a  list of 1s representing the population in each unit
    df["people"] = df[total].apply(lambda x: [1 for i in range(x)])

    # explode the dataframe to have n_rows = total_population
    df = df["people"].explode().reset_index()["index"].to_frame()
    df["groups"] = pop_groups

    # randomize people's group id
    df["groups"] = df["groups"].sample(frac=1).values

    # reaggregate by unit index
    df = df.groupby("index")["groups"].value_counts().unstack()
    df[total] = df[groups].sum(axis=1)
    df = df.join(geoms)

    return gpd.GeoDataFrame(df, geometry=geoms.geometry.name)


def simulate_evenness(df, group=None, total=None, groups=None):
    """Simulate even redistribution of population groups across spatial units.

    Parameters
    ----------
    df : geopandas.GeoDataFrame
        geodataframe with population data to be randomized
    group : str, optional
        name of column on geodataframe that holds the group total
        (for use with single group indices)
    total : str, optional
        name of column on geodataframe that holds the total population for
        each unit (for use with single group indices)
    groups : list, optional
        list of columns on inut dataframe that hold total population counts
        for each group of interest

    Returns
    -------
    geopandas.GeoDataFrame
        geodataframe with evenly distributed population groups

    Notes
    -------
    Simulates the random allocation of groups, given the total population of
    each geographic unit (randomizes group totals for each location). Given the total
    population of each location, take draws from a multinomial distribution to assign
    group categories for each person, where the probability of each group is equal to
    its regional share. Results are guaranteed to match regional and local population
    totals, but will include variation in the relative share of each group
    """
    df = df.copy()
    geoms = df[df.geometry.name].values
    if group:
        df[[group, total]] = df[[group, total]].astype(int)
        p_null = df[group].sum() / df[total].sum()

        output = gpd.GeoDataFrame()
        output[group] = np.random.binomial(n=df[total].values, p=p_null)
        output[total] = df[total].tolist()
    if groups:
        df = df[groups]
        global_prob_vector = df.sum(axis=0) / df.sum().sum()
        t = df[groups].sum(axis=1).astype(int)

        simul = list(
            map(lambda i: list(np.random.multinomial(i, global_prob_vector)), t)
        )
        output = pd.DataFrame(simul, columns=groups)
    output["geometry"] = geoms

    return gpd.GeoDataFrame(output)


def simulate_systematic_randomization(df, group=None, total=None, groups=None):
    """Simulate even redistribution of population groups across spatial units.

    Parameters
    ----------
    df : geopandas.GeoDataFrame
        geodataframe with population data to be randomized
    group : str, optional
        name of column on geodataframe that holds the group total
        (for use with single group indices)
    total : str, optional
        name of column on geodataframe that holds the total population for
        each unit (for use with single group indices)
    groups : list, optional
        list of columns on inut dataframe that hold total population counts
        for each group of interest

    Returns
    -------
    geopandas.GeoDataFrame
        geodataframe with systematically randomized population groups

    Notes
    -------
    Simulates the random allocation across geographic units, given the total population
    of each group (randomizes location totals for each group). Given the total population of
    each group in the region, take draws from a multinomial distribution where the
    probability of choosing each geographic unit is equal to the total regional share
    currently residing in the unit. Results will include regional and local variation in
    both total population and relative group shares.
    """
    if groups:
        df[total] = df[groups].sum(axis=1)
    if group:
        assert (
            total
        ), "If simulating a single group, you must also supply a total population column"
        df["other_group_pop"] = df[total] - df[group]
        groups = [group, "other_group_pop"]

    p_j = df[total] / df[total].sum()
    data_aux = {}
    for group in groups:
        n = df[group].sum()
        sim = np.random.multinomial(n, p_j)
        data_aux[group] = sim.tolist()
    df_aux = pd.DataFrame.from_dict(data_aux)

    return df_aux
