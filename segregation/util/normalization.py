"""Normalization constants for distortion coefficients.

Computes the maximal-segregation distortion coefficient *N* following
Olteanu et al. (2019), "Segregation through the multiscalar lens"
(https://doi.org/10.1073/pnas.1900192116).  The normalized distortion is::

    ̃Δ_i = Δ_i / N

where *N* is the largest distortion coefficient arising under a theoretical
extreme of complete segregation: the k groups sorted into k contiguous
"ghettos" ordered by size, with the smallest group in the most isolated
position.

The synthetic landscape is built by reallocating the *real* population totals
into the *real* areal units (same geometry, same distance metric), so *N* is
automatically on the same scale as the observed distortions.

A landscape is laid out by sweeping outward from a *seed* unit: units are
ordered by their distance from the seed, so every prefix of the ordering is the
set of units nearest the seed, and each ghetto is therefore a contiguous
concentric region.  Because *N* is defined as a maximum, several corner seeds
are tried and the largest resulting coefficient is returned; adding seeds can
only raise *N*, never lower it.  The seeds are derived deterministically from
the geometry, so *N* is a function of the data alone.
"""

import numpy as np
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.distance import cdist

from ..dynamics.divergence_profile import compute_divergence_profiles


def _candidate_seeds(coordinates, n_seeds=4):
    """Positional indices of the units to sweep outward from.

    Corners of the study region are the candidate "most isolated" positions, so
    the seeds are drawn from the convex hull of the centroids, spread evenly
    around it.  Degenerate geometries with no proper hull (a single point, or
    perfectly collinear centroids) fall back to the unit farthest from the mean
    center.

    Parameters
    ----------
    coordinates : np.ndarray
        (n, 2) array of centroid coordinates.
    n_seeds : int, default 4
        Maximum number of seeds to return.

    Returns
    -------
    np.ndarray
        Sorted, unique positional indices of the seed units.
    """
    if n_seeds < 1:
        raise ValueError(f"n_seeds must be at least 1 (got {n_seeds})")

    center = coordinates.mean(axis=0)
    most_peripheral = int(np.argmax(((coordinates - center) ** 2).sum(axis=1)))

    if n_seeds == 1:
        return np.array([most_peripheral])

    try:
        hull = ConvexHull(coordinates).vertices
    except (QhullError, ValueError):
        # degenerate geometry: no area to take a hull of
        return np.array([most_peripheral])

    if len(hull) > n_seeds:
        hull = hull[np.linspace(0, len(hull), n_seeds, endpoint=False).astype(int)]

    return np.unique(hull)


def _distances_from_seed(
    gdf, seed, metric="euclidean", network=None, distance_matrix=None
):
    """Distance from ``seed`` to every unit, in the metric of interest.

    Using the same metric as the distortion computation keeps the sweep
    contiguous in the metric that the coefficients are measured in.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input data with geometry and group columns.
    seed : int
        Positional index of the unit to measure from.
    metric : str, default "euclidean"
        Any metric accepted by ``scipy.spatial.distance.cdist``.  Ignored if
        ``network`` or ``distance_matrix`` is passed.
    network : pandarm.Network, optional
        Network used to measure shortest-path distance from the seed.
    distance_matrix : np.ndarray, optional
        Precomputed dense distance matrix.

    Returns
    -------
    np.ndarray
        (n,) distances from the seed.
    """
    centroids = gdf.geometry.centroid
    n = len(gdf)

    if network is not None:
        node_ids = network.get_node_ids(centroids.x, centroids.y)
        distances = network.shortest_path_lengths(
            [node_ids.iloc[seed]] * n, list(node_ids)
        )
        return np.asarray(distances, dtype=float)

    if distance_matrix is not None:
        if hasattr(distance_matrix, "tocsr"):
            raise TypeError(
                "sparse distance matrices are not supported; pass a dense "
                "numpy array as `distance_matrix`"
            )
        return np.asarray(distance_matrix, dtype=float)[seed]

    coordinates = np.column_stack((centroids.x, centroids.y))
    return cdist(coordinates[seed][None, :], coordinates, metric=metric).ravel()


def _ordering_for_normalization(
    gdf, seed=None, metric="euclidean", network=None, distance_matrix=None
):
    """Order unit indices outward from ``seed``.

    Sorting by distance from the seed yields concentric rings, so any prefix of
    the ordering is exactly the set of units nearest the seed.  Slicing the
    ordering into consecutive blocks therefore produces contiguous ghettos.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
    seed : int, optional
        Positional index to sweep from.  Defaults to the most peripheral unit.
    metric : str, default "euclidean"
    network : pandarm.Network, optional
    distance_matrix : np.ndarray, optional

    Returns
    -------
    np.ndarray
        Permutation of positional unit indices, starting at the seed.
    """
    n = len(gdf)
    if n < 2:
        return np.arange(n)

    if seed is None:
        centroids = gdf.geometry.centroid
        coordinates = np.column_stack((centroids.x, centroids.y))
        seed = int(_candidate_seeds(coordinates, n_seeds=1)[0])

    distances = _distances_from_seed(
        gdf,
        seed,
        metric=metric,
        network=network,
        distance_matrix=distance_matrix,
    )
    return np.argsort(distances, kind="stable")


def _block_boundaries(unit_pop, group_totals):
    """End index of each group's block along the ordered units.

    Cuts the sweep where cumulative population reaches each group's running
    total, so every ghetto holds its group's population at roughly the local
    density of the real landscape.  Blocks are then nudged so each holds at
    least one unit, which keeps every group's population in the synthetic
    landscape.

    Parameters
    ----------
    unit_pop : np.ndarray
        (n,) total population of each unit, in sweep order.
    group_totals : np.ndarray
        (k,) group population totals, ordered smallest group first.

    Returns
    -------
    np.ndarray
        (k,) exclusive end index of each block; strictly increasing, last == n.
    """
    n = unit_pop.size
    k = group_totals.size

    edges = np.searchsorted(np.cumsum(unit_pop), np.cumsum(group_totals), side="left")
    edges = np.asarray(edges, dtype=int) + 1
    edges[-1] = n

    # Each block needs at least one unit and the blocks must stay ordered.  The
    # forward pass gives block i room for the i blocks before it; the backward
    # pass leaves room for the blocks after it.  Both invariants can hold at
    # once because n >= k.
    for i in range(k):
        edges[i] = max(edges[i], i + 1)
    for i in range(k - 2, -1, -1):
        edges[i] = min(edges[i], edges[i + 1] - 1)

    return edges


def _segregated_landscape(gdf, groups, unit_order):
    """Reallocate the real population into fully segregated contiguous blocks.

    Each group occupies one contiguous run of ``unit_order``, smallest group
    first (the seed end), with its total spread across the block in proportion
    to each unit's original total population.  Group totals are preserved
    exactly.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
    groups : list of str
    unit_order : np.ndarray
        Sweep ordering of positional unit indices.

    Returns
    -------
    geopandas.GeoDataFrame
        Copy of ``gdf`` with the group columns replaced by the synthetic
        population.
    """
    df = gdf[groups].values.astype(float)
    group_totals = df.sum(axis=0)
    unit_totals = df.sum(axis=1)

    # Smallest group first, so it lands in the seed (most isolated) block
    group_order = np.argsort(group_totals, kind="stable")
    edges = _block_boundaries(unit_totals[unit_order], group_totals[group_order])

    synth = np.zeros_like(df)
    start = 0
    for i, g_pos in enumerate(group_order):
        block_units = unit_order[start : edges[i]]
        start = edges[i]

        block_original_totals = unit_totals[block_units]
        total_in_block = block_original_totals.sum()
        if total_in_block > 0:
            shares = block_original_totals / total_in_block
        else:
            shares = np.full(block_units.size, 1 / block_units.size)

        synth[block_units, g_pos] = group_totals[g_pos] * shares

    synth_gdf = gdf.copy()
    for g_idx, g_name in enumerate(groups):
        synth_gdf[g_name] = synth[:, g_idx]

    return synth_gdf


def _maximal_segregation_distortion(
    gdf, groups, metric="euclidean", network=None, distance_matrix=None, n_seeds=4
):
    """Compute the maximal-segregation distortion coefficient N.

    Builds synthetic landscapes of complete segregation by reallocating the
    real population totals into the real areal units (same geometry), arranged
    so each group occupies a contiguous concentric block with the smallest
    group in the most isolated position.  Returns the largest distortion
    coefficient found, which is used to normalize observed distortions::

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
    n_seeds : int, default 4
        Number of corner positions to build a segregated landscape from.  N is
        the maximum over all of them, so raising this can only increase N (a
        tighter estimate of the theoretical maximum) at the cost of one more
        divergence profile per seed.

    Returns
    -------
    float
        The normalization constant N.

    Notes
    -----
    Each seed costs a full divergence profile over the synthetic landscape, so
    normalization multiplies the cost of the distortion computation by roughly
    ``n_seeds + 1``.
    """
    df = gdf[groups].values.astype(float)
    n = len(gdf)
    n_groups = len(groups)

    if n < 2:
        raise ValueError("Need at least 2 observations to compute normalization")

    if n_groups < 2:
        raise ValueError(
            "Normalization requires at least 2 groups; with 1 group there is no segregation"
        )

    if n < n_groups:
        raise ValueError(
            f"Normalization requires at least as many observations as groups, "
            f"so each group can occupy its own block (got {n} observations "
            f"and {n_groups} groups)"
        )

    if df.sum() == 0:
        raise ValueError("Total population is zero; cannot compute normalization")

    centroids = gdf.geometry.centroid
    coordinates = np.column_stack((centroids.x, centroids.y))
    seeds = _candidate_seeds(coordinates, n_seeds=n_seeds)

    N = -np.inf
    for seed in seeds:
        unit_order = _ordering_for_normalization(
            gdf,
            seed=int(seed),
            metric=metric,
            network=network,
            distance_matrix=distance_matrix,
        )
        synth_gdf = _segregated_landscape(gdf, groups, unit_order)

        aux = compute_divergence_profiles(
            gdf=synth_gdf,
            groups=groups,
            metric=metric,
            network=network,
            distance_matrix=distance_matrix,
        )
        # divergence --> distortion by summing at each location
        distortion = aux.groupby("observation").sum()["divergence"]
        N = max(N, float(distortion.max()))

    return N
