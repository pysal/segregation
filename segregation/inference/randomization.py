"""Tools for simulating spatial population distributions."""

import itertools

import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


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
    df = df.copy()
    df[[group, total]] = df[[group, total]].astype(int)
    if group:
        p_null = df[group].sum() / df[total].sum()

        output = gpd.GeoDataFrame()
        output[group] = np.random.binomial(n=df[total].values, p=p_null)
        output[total] = df[total].tolist()
        output["geometry"] = df[df.geometry.name]

    return output
