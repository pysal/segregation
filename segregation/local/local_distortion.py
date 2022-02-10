import geopandas as gpd

from .._base import MultiGroupIndex, SpatialExplicitIndex
from ..dynamics import compute_divergence_profiles

def _local_distortion(
    gdf, groups, network=None, metric="euclidean", normalize=False
):
    """
    A segregation metric, using Kullback-Leiber (KL) divergence to quantify the
    difference in the population characteristics between (1) an area and (2) the total population.

    This function utilises the methodology proposed in
    Olteanu et al. (2019): 'Segregation through the multiscalar lens'. Which can be
    found here: https://doi.org/10.1073/pnas.1900192116

    Arguments
    ----------
    gdf: geopandas.GeoDataFrame
        geodataframe with group population counts (not percentages) to be included in the analysis.
    groups: list
        list of columns on gdf that contain population counts of interest
    network: pandana.Network object (optional)
        A pandana Network object used to compute distance between observations
    metric : str
        Distance metric for calculating pairwise distances,
        Accepts any inputs to `scipy.spatial.distance.pdist` 
        'euclidean' by default. Ignored if passing a network
    coefs: bool (default: True)
        whether to return KL divergence coefficients. If True, the function will
        return a geodataframe with one
    normalization:
        NOT YET IMPLEMENTED


    Returns
    ----------
    aux : geopandas.GeoDataFrame
        geodataframe of distortion coefficient values

    """
    # Store the observation index to return with the results
    geoms = gdf[gdf.geometry.name]
    centroids = gdf.geometry.centroid

    aux = compute_divergence_profiles(gdf=gdf, groups=groups, network=network, metric=metric)
    # divergence --> distortion by summing at each location
    aux = gpd.GeoDataFrame(
        aux.groupby("observation").sum()[["divergence"]], geometry=geoms
    ).rename(columns={'divergence': 'distortion'})
    if normalize:
        raise Exception("Not yet implemented")
        # Need to write a routine to determine the scaling factor... From the paper:

        # the maximum distortion coefficient in a theoretical extreme case of segregation. 
        # Theoretically, the maximal-segregation distortion coefficient is achieved when sorting 
        # the k groups into k ghettos, ordered by sizes, and then computing the coefficient for
        # the most isolated person in the smallest group

    return aux


class LocalDistortion(MultiGroupIndex, SpatialExplicitIndex):
    """Multigroup Local Distortion Coefficients.

    Parameters
    ----------
    data : pandas.DataFrame or geopandas.GeoDataFrame, required
        dataframe or geodataframe if spatial index holding data for location of interest
    groups : list, required
        list of columns on dataframe holding population totals for each group
    network : pandana.Network (optional)
        A pandana Network object used to compute distance between observations
    metric : str
        Distance metric for calculating pairwise distances,
        Accepts any inputs to `scipy.spatial.distance.pdist`
        'euclidean' by default. Ignored if passing a network
    normalization:
        NOT YET IMPLEMENTED

    Attributes
    ----------
    statistics : pandas.Series
        KL Divergence coefficients
    core_data : a pandas DataFrame
        DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Olteanu et al. (2019): 'Segregation through the multiscalar lens'.  https://doi.org/10.1073/pnas.1900192116

    """

    def __init__(
        self,
        data,
        network=None,
        groups=None,
        metric="euclidean",
        normalize=False,
        **kwargs

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
        )

        self.statistics = aux["distortion"]
        self.data = aux
        self._function = _local_distortion