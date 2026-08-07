from warnings import warn

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform
from scipy.special import rel_entr as relative_entropy

from ..network import compute_travel_cost_matrix

# Threshold (in bytes) above which the euclidean path falls back from a dense
# distance matrix to streaming cKDTree queries.  Default: 8 GB.
_DISTANCE_MATRIX_LIMIT = 8 * 1024**3


def _kl_profile_for_origin(sorted_indices, distances, df, indices, i):
    """Compute the KL divergence profile for a single origin.

    Parameters
    ----------
    sorted_indices : np.ndarray
        Unit indices sorted by ascending distance from the origin.
    distances : np.ndarray
        Corresponding distances (already sorted ascending).
    df : np.ndarray
        (n, k) array of group population counts.
    indices : pd.Index
        Original observation index for labelling.
    i : int
        Positional index of the origin.

    Returns
    -------
    pd.DataFrame
        Divergence profile for this origin.
    """
    cumul_pop_by_group = np.cumsum(df[sorted_indices], axis=0)
    obs_cumul_pop = np.sum(cumul_pop_by_group, axis=1)[:, np.newaxis]
    q_cumul_proportions = cumul_pop_by_group / obs_cumul_pop
    total_pop_by_group = np.sum(df, axis=0, keepdims=True)
    total_pop = np.sum(df)
    r_total_proportions = total_pop_by_group / total_pop

    kl_divergence = relative_entropy(q_cumul_proportions, r_total_proportions).sum(
        axis=1
    )

    return pd.DataFrame().from_dict(
        dict(
            observation=indices[i],
            distance=distances,
            divergence=kl_divergence,
            population_covered=obs_cumul_pop.sum(axis=1),
        )
    )


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
    network: pandarm.Network object (optional, None by default)
        A pandarm Network object used to compute distance between observations
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

    Notes
    -----
    For the default euclidean metric, the implementation uses a dense distance
    matrix via ``scipy.spatial.distance.pdist`` (fast, vectorized C) when the
    matrix fits comfortably in memory.  For very large datasets where the dense
    matrix would be impractical, it falls back to a streaming ``cKDTree`` path
    that keeps peak memory at *O(n)* per iteration.  The threshold is
    controlled by the ``_DISTANCE_MATRIX_LIMIT`` constant (default 2 GB).

    """
    # Store the observation index to return with the results
    indices = gdf.index.copy()
    centroids = gdf.geometry.centroid
    df = gdf[groups].values
    n = len(df)

    coordinates = np.column_stack((centroids.x, centroids.y))

    # Preparing list for results
    results = []

    if network:
        # --- Network path: dense matrix from pandarm ---
        if metric != "network":
            warn(
                f"metric set to {metric} but a pandarm.Network object was passed. Using network distances instead"
                "If you wish to use a scipy distance matrix, do not include a `network` argument`"
            )
        dist_matrix = compute_travel_cost_matrix(gdf, gdf, network).values
        for i in range(n):
            sorted_indices = np.argsort(dist_matrix[i])
            results.append(
                _kl_profile_for_origin(
                    sorted_indices, dist_matrix[i][sorted_indices], df, indices, i
                )
            )
    elif distance_matrix is not None:
        # --- Precomputed matrix path ---
        if metric != "precomputed":
            warn(
                f"metric set to {metric} but a distance_matrix argument was passed. Using precomputed distances instead"
            )
        for i in range(n):
            sorted_indices = np.argsort(distance_matrix[i])
            results.append(
                _kl_profile_for_origin(
                    sorted_indices, distance_matrix[i][sorted_indices], df, indices, i
                )
            )
    elif metric == "euclidean":
        # --- Euclidean path ---
        # Use the fast dense-matrix path (pdist) when the matrix fits in
        # memory; fall back to streaming cKDTree for very large datasets.
        matrix_bytes = n * n * 8
        if matrix_bytes <= _DISTANCE_MATRIX_LIMIT:
            dist_matrix = squareform(pdist(coordinates, metric=metric))
            for i in range(n):
                sorted_indices = np.argsort(dist_matrix[i])
                results.append(
                    _kl_profile_for_origin(
                        sorted_indices,
                        dist_matrix[i][sorted_indices],
                        df,
                        indices,
                        i,
                    )
                )
        else:
            # Streaming path: query in chunks to amortize Python→C overhead
            # while keeping peak memory at O(chunk × n) not O(n²).
            tree = cKDTree(coordinates)
            chunk_size = min(n, 5000)
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                _distances, _sorted_indices = tree.query(coordinates[start:end], k=n)
                for j in range(end - start):
                    i = start + j
                    results.append(
                        _kl_profile_for_origin(
                            _sorted_indices[j], _distances[j], df, indices, i
                        )
                    )
    else:
        # --- Non-euclidean scipy metric: fall back to dense matrix ---
        dist_matrix = squareform(pdist(coordinates, metric=metric))
        for i in range(n):
            sorted_indices = np.argsort(dist_matrix[i])
            results.append(
                _kl_profile_for_origin(
                    sorted_indices, dist_matrix[i][sorted_indices], df, indices, i
                )
            )

    aux = pd.concat(results)

    return aux
