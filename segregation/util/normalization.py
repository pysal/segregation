"""Normalization constants for distortion coefficients.

Computes the maximal-segregation distortion coefficient *N* following
Olteanu et al. (2019), "Segregation through the multiscalar lens"
(https://doi.org/10.1073/pnas.1900192116).  The normalized distortion is::

    ̃Δ_i = Δ_i / N

where *N* is the distortion coefficient of the most isolated person in the
smallest group under a theoretical extreme of complete segregation (k groups
sorted into k contiguous ghettos, ordered by group size).

The synthetic landscape is built by reallocating the *real* population totals
into the *real* areal units (same geometry, same distance metric), so *N* is
automatically on the same scale as the observed distortions.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import depth_first_order, minimum_spanning_tree
from scipy.spatial import cKDTree

from ..dynamics.divergence_profile import compute_divergence_profiles
from ..network import compute_travel_cost_matrix


def _build_knn_graph(gdf, metric, network, distance_matrix, k=30):
    """Build a sparse k-nearest-neighbor graph as a sparse adjacency matrix.

    The graph respects the same distance metric used by the distortion
    computation (euclidean, network, or precomputed), so units adjacent in the
    graph are genuinely close in the metric of interest.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input data with geometry and group columns.
    metric : str
        Distance metric label.
    network : pandarm.Network or None
    distance_matrix : np.ndarray or None
        Precomputed distance matrix.
    k : int, default 30
        Number of nearest neighbors per unit.

    Returns
    -------
    scipy.sparse.coo_matrix
        Sparse symmetric adjacency matrix of shape (n, n).
    """
    n = len(gdf)
    k = min(k, n - 1)
    if k < 1:
        raise ValueError("Need at least 2 observations to build a k-NN graph")

    centroids = gdf.geometry.centroid
    coords = np.column_stack((centroids.x, centroids.y))

    if network is not None:
        # Network: use pandarm.k_nearest_nodes
        node_ids = network.get_node_ids(centroids.x, centroids.y)
        result = network.k_nearest_nodes(node_ids, k=k, max_radius=0)
        imp_name = result.columns[-1]
        rows = result["source"].values
        cols = result["destination"].values
        vals = result[imp_name].values
        # Map node IDs back to positional indices
        id_to_pos = pd.Series(np.arange(n), index=gdf.index)
        src_pos = id_to_pos.loc[rows].values
        dst_pos = id_to_pos.loc[cols].values
    elif distance_matrix is not None:
        # Precomputed (dense or sparse): take k smallest per row
        if hasattr(distance_matrix, "tocoo"):
            # Sparse input
            dm = distance_matrix.tocsr()
            rows, cols, vals = [], [], []
            for i in range(n):
                row = dm.getrow(i).tocoo()
                if len(row.data) == 0:
                    continue
                knn_idx = np.argpartition(row.data, min(k, len(row.data) - 1))[:k]
                for j in knn_idx:
                    rows.append(i)
                    cols.append(row.col[j])
                    vals.append(row.data[j])
            rows, cols, vals = np.array(rows), np.array(cols), np.array(vals)
            src_pos, dst_pos = rows, cols
        else:
            # Dense
            src_pos = np.repeat(np.arange(n), k)
            dst_pos = np.empty(n * k, dtype=int)
            vals = np.empty(n * k, dtype=float)
            for i in range(n):
                row = distance_matrix[i].copy()
                row[i] = np.inf  # exclude self
                idx = np.argpartition(row, k - 1)[:k]
                dst_pos[i * k : (i + 1) * k] = idx
                vals[i * k : (i + 1) * k] = row[idx]
    else:
        # Euclidean via cKDTree
        tree = cKDTree(coords)
        for i in range(n):
            dists, idx = tree.query(coords[i], k=k + 1)
            # First result is self (distance 0); skip it
            if i == 0:
                src_pos = np.empty(n * k, dtype=int)
                dst_pos = np.empty(n * k, dtype=int)
                vals = np.empty(n * k, dtype=float)
            src_pos[i * k : (i + 1) * k] = i
            dst_pos[i * k : (i + 1) * k] = idx[1:]
            vals[i * k : (i + 1) * k] = dists[1:]

    # Build symmetric sparse graph (undirected)
    all_rows = np.concatenate([src_pos, dst_pos])
    all_cols = np.concatenate([dst_pos, src_pos])
    all_vals = np.concatenate([vals, vals])
    graph = coo_matrix((all_vals, (all_rows, all_cols)), shape=(n, n))

    return graph


def _ordering_for_normalization(gdf, metric, network, distance_matrix, k=30):
    """Return an ordering of unit indices contiguous in the distance metric.

    Builds a k-NN graph, computes its MST, and returns a DFS traversal
    that visits spatially adjacent units consecutively.  The smallest group
    will be placed at the leaf end (most isolated position).

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
    metric : str
    network : pandarm.Network or None
    distance_matrix : np.ndarray or None
    k : int, default 30

    Returns
    -------
    np.ndarray
        Permutation of unit indices (positional) forming a contiguous path.
    """
    n = len(gdf)
    if n < 2:
        return np.arange(n)

    graph = _build_knn_graph(gdf, metric, network, distance_matrix, k=k)
    mst = minimum_spanning_tree(graph.tocsr())
    mst = mst.tocsr()

    # Find the most peripheral node as the DFS root (largest mean distance)
    # Use the graph's edge weights as a proxy for distance
    row_sums = np.array(mst.sum(axis=1)).ravel()
    root = int(np.argmax(row_sums)) if row_sums.size > 0 else 0

    order, _ = depth_first_order(mst, i_start=root, directed=False)
    return order


def _maximal_segregation_distortion(
    gdf, groups, metric="euclidean", network=None, distance_matrix=None, k=30
):
    """Compute the maximal-segregation distortion coefficient N.

    Builds a synthetic landscape of complete segregation by reallocating the
    real population totals into the real areal units (same geometry), arranged
    so each group occupies a contiguous spatial block.  Returns the maximum
    distortion coefficient, which is used to normalize observed distortions::

        ̃Δ_i = Δ_i / N

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input data with geometry and group population columns.
    groups : list of str
        Column names holding group population totals.
    metric : str, default "euclidean"
        Distance metric (passed through to ``compute_divergence_profiles``).
    network : pandarm.Network, optional
    distance_matrix : np.ndarray, optional
    k : int, default 30
        Number of nearest neighbors for the ordering graph.

    Returns
    -------
    float
        The normalization constant N.
    """
    df = gdf[groups].values.astype(float)
    n = len(gdf)

    if n < 2:
        raise ValueError("Need at least 2 observations to compute normalization")

    # Group totals (control totals we must preserve)
    group_totals = df.sum(axis=0)
    grand_total = group_totals.sum()

    if grand_total == 0:
        raise ValueError("Total population is zero; cannot compute normalization")

    n_groups = len(groups)
    if n_groups < 2:
        raise ValueError(
            "Normalization requires at least 2 groups; with 1 group there is no segregation"
        )

    # Sort groups ascending by total (smallest group → most isolated position)
    group_order = np.argsort(group_totals)

    # Get spatial ordering
    unit_order = _ordering_for_normalization(gdf, metric, network, distance_matrix, k=k)

    # Partition units into k contiguous blocks, sizes proportional to group shares
    # Smallest group gets the last block (leaf end = most isolated)
    block_sizes = np.zeros(n_groups, dtype=int)
    for g_idx, g_pos in enumerate(group_order):
        share = group_totals[g_pos] / grand_total
        block_sizes[g_pos] = max(1, int(round(n * share)))

    # Adjust so blocks sum to n
    diff = n - block_sizes.sum()
    if diff != 0:
        # Add/remove from the largest group's block
        largest = int(np.argmax(group_totals))
        block_sizes[largest] += diff
        if block_sizes[largest] < 1:
            # Fallback: distribute the correction across all groups
            block_sizes[largest] = 1
            remaining = n - n_groups + 1
            for g_pos in group_order[:-1]:
                share = group_totals[g_pos] / grand_total
                block_sizes[g_pos] = max(1, int(round(remaining * share)))
            diff = n - block_sizes.sum()
            largest = int(np.argmax(group_totals))
            block_sizes[largest] += diff

    # Build synthetic population matrix
    synth = np.zeros_like(df)
    unit_idx = 0
    for g_pos in group_order:
        block_size = block_sizes[g_pos]
        block_units = unit_order[unit_idx : unit_idx + block_size]
        unit_idx += block_size

        # Each unit in this block is 100% this group
        # Distribute the group total proportional to each unit's original total population
        block_original_totals = df[block_units].sum(axis=1)
        total_in_block = block_original_totals.sum()
        if total_in_block > 0:
            shares = block_original_totals / total_in_block
        else:
            shares = np.ones(block_size) / block_size

        synth[block_units, g_pos] = group_totals[g_pos] * shares

    # Build synthetic GeoDataFrame (real geometry + synthetic population)
    synth_gdf = gdf.copy()
    for g_idx, g_name in enumerate(groups):
        synth_gdf[g_name] = synth[:, g_idx]

    # Compute distortion on synthetic landscape using the same distance params
    aux = compute_divergence_profiles(
        gdf=synth_gdf,
        groups=groups,
        metric=metric,
        network=network,
        distance_matrix=distance_matrix,
    )

    # Sum divergence per observation → distortion coefficients
    distortion = aux.groupby("observation").sum()["divergence"]
    N = float(distortion.max())

    return N
