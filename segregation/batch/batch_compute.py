"""Batch compute wrappers for calculating all relevant statistics at once."""

import inspect
import warnings

import pandas as pd
from tqdm.auto import tqdm

from .. import multigroup, singlegroup
from .._base import SpatialImplicitIndex
from ..dynamics import compute_multiscalar_profile

singlegroup_classes = {}
for name, obj in inspect.getmembers(singlegroup):
    if inspect.isclass(obj):
        singlegroup_classes[name] = obj

multigroup_classes = {}
for name, obj in inspect.getmembers(multigroup):
    if inspect.isclass(obj):
        multigroup_classes[name] = obj

implicit_single_indices = {}
for name, obj in inspect.getmembers(singlegroup):
    if inspect.isclass(obj):
        if str(SpatialImplicitIndex) in [str(i) for i in obj.__bases__]:
            implicit_single_indices[name] = obj

implicit_multi_indices = {}
for name, obj in inspect.getmembers(multigroup):
    if inspect.isclass(obj):
        if str(SpatialImplicitIndex) in [str(i) for i in obj.__bases__]:
            implicit_multi_indices[name] = obj


def batch_compute_singlegroup(
    gdf, group_pop_var, total_pop_var, progress_bar=True, **kwargs
):
    """Batch compute single-group indices.

    Parameters
    ----------
    gdf : DataFrame or GeoDataFrame
        DataFrame holding demographic data for study region
    group_pop_var : str
        The name of variable in data that contains the population size of the group of interest
    total_pop_var : str
        Variable in data that contains the total population count of the unit
    progress_bar: bool
        Whether to show a progress bar during calculation
    **kwargs : dict
        additional keyword arguments passed to each index (e.g. for setting a random
        seed in indices like ModifiedGini or ModifiedDissm)

    Returns
    -------
    pandas.DataFrame
        dataframe with statistic name as dataframe index and statistic value as dataframe values
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = {}
        if progress_bar:
            pbar = tqdm(total=len(singlegroup_classes.keys()))

            for each in sorted(singlegroup_classes.keys()):
                pbar.set_description(each)
                fitted[each] = singlegroup_classes[each](
                    gdf, group_pop_var, total_pop_var, **kwargs
                ).statistic
                pbar.update(1)
        else:
            for each in sorted(singlegroup_classes.keys()):
                fitted[each] = singlegroup_classes[each](
                    gdf, group_pop_var, total_pop_var, **kwargs
                ).statistic
        fitted = pd.DataFrame.from_dict(fitted, orient="index").round(4)
        fitted.columns = ["Statistic"]
        fitted.index.name = "Name"
        return fitted


def batch_compute_multigroup(gdf, groups, **kwargs):
    """Batch compute multi-group indices.

    Parameters
    ----------
    gdf : DataFrame or GeoDataFrame
        DataFrame holding demographic data for study region
    groups : list
        The variables names in data of the groups of interest of the analysis.
    **kwargs : dict
        additional keyword arguments passed to each index (e.g. for setting a random
        seed in indices like ModifiedGini or ModifiedDissm)

    Returns
    -------
    pandas.DataFrame
        dataframe with statistic name as dataframe index and statistic value as dataframe values
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = {}
        for each in sorted(multigroup_classes.keys()):
            fitted[each] = multigroup_classes[each](gdf, groups, **kwargs).statistic
        fitted = pd.DataFrame.from_dict(fitted, orient="index").round(4)
        fitted.columns = ["Statistic"]
        fitted.index.name = "Name"
    return fitted


def batch_multiscalar_singlegroup(
    gdf, distances, group_pop_var, total_pop_var, progress_bar=True, **kwargs
):
    """Batch compute multiscalar profiles for single-group indices.

    Parameters
    ----------
    gdf : DataFrame or GeoDataFrame
        DataFrame holding demographic data for study region
    distances : list
        list of floats representing bandwidth distances that define a local
        environment.
    group_pop_var : str
        The name of variable in data that contains the population size of the group
        of interest
    total_pop_var : str
        Variable in data that contains the total population count of the unit
    progress_bar: bool
        Whether to show a progress bar during calculation
    **kwargs : dict
        additional keyword arguments passed to each index (e.g. for setting a random
        seed in indices like ModifiedGini or ModifiedDissm)

    Returns
    -------
    pandas.DataFrame
        pandas Dataframe with distance as dataframe index and each segregation
        statistic as dataframe columns
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        profs = []
        if progress_bar:
            pbar = tqdm(total=len(implicit_single_indices.keys()))
            for idx in sorted(implicit_single_indices.keys()):
                pbar.set_description(idx)
                prof = compute_multiscalar_profile(
                    gdf=gdf,
                    segregation_index=implicit_single_indices[idx],
                    distances=distances,
                    group_pop_var=group_pop_var,
                    total_pop_var=total_pop_var,
                    **kwargs
                )
                profs.append(prof)
                pbar.update(1)
        else:
            for idx in sorted(implicit_single_indices.keys()):
                prof = compute_multiscalar_profile(
                    gdf=gdf,
                    segregation_index=implicit_single_indices[idx],
                    distances=distances,
                    group_pop_var=group_pop_var,
                    total_pop_var=total_pop_var,
                    **kwargs
                )
                profs.append(prof)
        df = pd.concat(profs, axis=1)
        return df


def batch_multiscalar_multigroup(gdf, distances, groups, progress_bar=True, **kwargs):
    """Batch compute multiscalar profiles for multi-group indices.

    Parameters
    ----------
    gdf : DataFrame or GeoDataFrame
        DataFrame holding demographic data for study region
    distances : list
        list of floats representing bandwidth distances that define a local
        environment.
    groups : list
        The variables names in data of the groups of interest of the analysis.
    progress_bar: bool
        Whether to show a progress bar during calculation
    **kwargs : dict
        additional keyword arguments passed to each index (e.g. for setting a random
        seed in indices like ModifiedGini or ModifiedDissm)

    Returns
    -------
    pandas.DataFrame
        pandas Dataframe with distance as dataframe index and each segregation
        statistic as dataframe columns
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        profs = []
        if progress_bar:
            pbar = tqdm(total=len(implicit_multi_indices.keys()))
            for idx in sorted(implicit_multi_indices.keys()):
                pbar.set_description(idx)
                prof = compute_multiscalar_profile(
                    gdf=gdf,
                    segregation_index=implicit_multi_indices[idx],
                    distances=distances,
                    groups=groups,
                    **kwargs
                )
                profs.append(prof)
                pbar.update(1)

        else:
            for idx in sorted(implicit_multi_indices.keys()):
                prof = compute_multiscalar_profile(
                    gdf=gdf,
                    segregation_index=implicit_multi_indices[idx],
                    distances=distances,
                    groups=groups,
                    **kwargs
                )
                profs.append(prof)
        df = pd.concat(profs, axis=1)
        return df
