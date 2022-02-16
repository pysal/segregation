import numpy as np
import pandas as pd

from scipy.spatial.distance import pdist, squareform
from scipy.special import rel_entr as relative_entropy

from ..network import compute_travel_cost_matrix
from warnings import warn


def compute_divergence_profiles(
    gdf, groups, metric="euclidean", network=None, distance_matrix=None
):
    """
    A segregation metric using Kullback-Leiber (KL) divergence to quantify the
    difference in the population characteristics between (1) an area and (2) the total population.

    Parameters
    ----------
    data : pandas.DataFrame or geopandas.GeoDataFrame, required
        dataframe or geodataframe if spatial index holding data for location of interest
    groups : list, required
        list of columns on dataframe holding population totals for each group
    metric : str (optional; 'euclidean' by default)
        Distance metric for calculating pairwise distances,
        Accepts any inputs to `scipy.spatial.distance.pdist`.
        Ignored if passing a network or distance matrix
    network: pandana.Network object (optional, None by default)
        A pandana Network object used to compute distance between observations
    distance_matrix: numpy.array (optional; None by default)
        numpy array of distances between observations in the dataset

    Returns
    ----------
    aux : geopandas.GeoDataFrame
        geodataframe of the KL divergence measure, between the aggregated population and the
        total population, will converge to zero for the final row of each
        observation to represent that the total population is covered.
        population_covered : the population count within the aggregated population.
        Returns a concatenated object of Pandas dataframes. Each dataframe contains a
        set of divergence levels between an area and the total population. These areas
        become consecutively larger, starting from a single location and aggregating
        outward from this location, until the area represents the total population.
        Thus, together the divergence levels within a dataframe represent a profile
        of divergence from an area. The concatenated object is the collection of these
        divergence profiles for every areas within the total population.

    """
    # Store the observation index to return with the results
    indices = gdf.index.copy()
    centroids = gdf.geometry.centroid
    df = gdf[groups].values

    coordinates = np.column_stack((centroids.x, centroids.y))

    # If given a pandana network, use shortest network distance, otherwise use scikit
    if network:
        if metric != "network":
            warn(
                f"metric set to {metric} but a pandana.Network object was passed. Using network distances instead"
                "If you wish to use a scipy distance matrix, do not include a `network` argument`"
            )
        dist_matrix = compute_travel_cost_matrix(gdf, gdf, network).values
    elif distance_matrix:
        if metric != "precomputed":
            warn(
                f"metric set to {metric} but a distance_matrix argument was passed. Using precomputed distances instead"
            )
        dist_matrix = distance_matrix
    else:
        dist_matrix = squareform(pdist(coordinates, metric=metric))

    # Preparing list for results
    results = []

    # Loop to calculate KL divergence
    for (i, distances) in enumerate(dist_matrix):

        # Creating the q and r objects
        sorted_indices = np.argsort(distances)
        cumul_pop_by_group = np.cumsum(df[sorted_indices], axis=0)
        obs_cumul_pop = np.sum(cumul_pop_by_group, axis=1)[:, np.newaxis]
        q_cumul_proportions = cumul_pop_by_group / obs_cumul_pop
        total_pop_by_group = np.sum(df, axis=0, keepdims=True)
        total_pop = np.sum(df)
        r_total_proportions = total_pop_by_group / total_pop

        # Input q and r objects into relative entropy (KL divergence) function
        kl_divergence = relative_entropy(q_cumul_proportions, r_total_proportions).sum(
            axis=1
        )

        # Creating an output dataframe
        output = pd.DataFrame().from_dict(
            dict(
                observation=indices[i],
                distance=distances[sorted_indices],
                divergence=kl_divergence,
                population_covered=obs_cumul_pop.sum(axis=1),
            )
        )

        # Append (bring together) all outputs into results list
        results.append(output)

    aux = pd.concat(results)

    return aux
