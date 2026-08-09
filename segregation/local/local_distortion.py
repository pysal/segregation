import geopandas as gpd

from .._base import MultiGroupIndex, SpatialExplicitIndex
from ..dynamics import compute_divergence_profiles
from ..util.normalization import _maximal_segregation_distortion


def _local_distortion(
    gdf,
    groups,
    metric="euclidean",
    network=None,
    distance_matrix=None,
    normalize=True,
    n_seeds=4,
):
    """
    A segregation metric, using Kullback-Leiber (KL) divergence to quantify the
    difference in the population characteristics between (1) an area and (2) the total population.

    This function utilises the methodology proposed in
    Olteanu et al. (2019): 'Segregation through the multiscalar lens'. Which can be
    found here: https://doi.org/10.1073/pnas.1900192116

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
    normalize: bool
        If True, normalize coefficients by the maximum theoretical segregation value
    n_seeds: int (optional; 4 by default)
        Number of corner positions used to build the maximally-segregated
        reference landscape. The normalization constant is the maximum over all
        of them, so raising this yields a tighter estimate of the theoretical
        maximum at the cost of one extra divergence profile per seed.
        Ignored when ``normalize`` is False.


    Returns
    ----------
    aux : geopandas.GeoDataFrame
        geodataframe of distortion coefficient values

    """
    # Store the observation index to return with the results
    geoms = gdf[gdf.geometry.name]

    aux = compute_divergence_profiles(
        gdf=gdf,
        groups=groups,
        network=network,
        metric=metric,
        distance_matrix=distance_matrix,
    )
    # divergence --> distortion by summing at each location
    aux = gpd.GeoDataFrame(
        aux.groupby("observation").sum()[["divergence"]], geometry=geoms
    ).rename(columns={"divergence": "distortion"})
    if normalize:
        N = _maximal_segregation_distortion(
            gdf,
            groups,
            metric=metric,
            network=network,
            distance_matrix=distance_matrix,
            n_seeds=n_seeds,
        )
        aux["distortion"] = aux["distortion"] / N
        aux.attrs["normalization_constant"] = N

    return aux


class LocalDistortion(MultiGroupIndex, SpatialExplicitIndex):
    """Multigroup Local Distortion Coefficients.

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
    network: pandarm.Network object (optional; None by default)
        A pandarm Network object used to compute distance between observations
    distance_matrix:
        numpy array of distances between observations in the dataset
    normalize: bool (optional; False by default)
        If True, normalize coefficients by the maximum theoretical segregation
        value for this dataset
    n_seeds: int (optional; 4 by default)
        Number of corner positions used to build the maximally-segregated
        reference landscape. Raising this tightens the normalization constant
        at the cost of one extra divergence profile per seed. Ignored when
        ``normalize`` is False.

    Attributes
    ----------
    statistics : pandas.Series
        KL Divergence coefficients
    core_data : a pandas DataFrame
        DataFrame that contains the columns used to perform the estimate.
    normalization_constant : float or None
        The maximal-segregation distortion coefficient used to normalize the
        coefficients, or None when ``normalize`` is False.

    Notes
    -----
    Olteanu et al. (2019): 'Segregation through the multiscalar lens'.  https://doi.org/10.1073/pnas.1900192116

    """

    def __init__(
        self,
        data,
        groups=None,
        metric="euclidean",
        network=None,
        distance_matrix=None,
        normalize=False,
        n_seeds=4,
        **kwargs,
    ):
        """Init."""

        MultiGroupIndex.__init__(self, data, groups)
        SpatialExplicitIndex.__init__(self)

        aux = _local_distortion(
            self.data,
            self.groups,
            network=network,
            metric=metric,
            normalize=normalize,
            distance_matrix=distance_matrix,
            n_seeds=n_seeds,
        )

        self.statistics = aux["distortion"]
        self.data = aux
        self._function = _local_distortion
        self.normalization_constant = aux.attrs.get("normalization_constant", None)
