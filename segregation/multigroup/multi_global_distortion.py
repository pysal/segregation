import geopandas as gpd

from .._base import MultiGroupIndex, SpatialExplicitIndex
from ..dynamics import compute_divergence_profiles


def _global_distortion(
    gdf, groups, network=None, metric="euclidean", distance_matrix=None, normalize=False
):
    """
    A segregation metric, using Kullback-Leiber (KL) divergence to quantify the
    difference in the population characteristics between (1) an area and (2) the total population.

    This function utilises the methodology proposed in
    Olteanu et al. (2019): 'Segregation through the multiscalar lens'. Which can be
    found here: https://doi.org/10.1073/pnas.1900192116

    Arguments
    ----------
    gdf : geopandas.GeoDataFrame
        geodataframe with group population counts (not percentages) to be included in the analysis.
    groups : list
        list of columns on gdf that contain population counts of interest
    metric : str (optional; 'euclidean' by default)
        Distance metric for calculating pairwise distances,
        Accepts any inputs to `scipy.spatial.distance.pdist`.
        Ignored if passing a network or distance matrix
    network: pandana.Network object (optional)
        A pandana Network object used to compute distance between observations
    distance_matrix: numpy.array
        numpy array of distances between observations in the dataset
    normalization:
        NOT YET IMPLEMENTED


    Returns
    ----------
    gdf: geopands.GeoDataFrame
        a geodataframe of input data used to compute the statistic
    statistic : float
        the global distortion index

    """
    # Store the observation index to return with the results
    geoms = gdf[gdf.geometry.name]
    df = gdf[groups].values

    total_pop = df.sum().sum()

    aux = compute_divergence_profiles(
        gdf=gdf,
        groups=groups,
        network=network,
        metric=metric,
        distance_matrix=distance_matrix,
    )

    #  this yeilds distortion coefficients
    aux = aux.groupby("observation").sum()[["divergence"]]

    if normalize:
        raise Exception("Not yet implemented")
        # Need to write a routine to determine the scaling factor... From the paper:

        # the maximum distortion coefficient in a theoretical extreme case of segregation.
        # Theoretically, the maximal-segregation distortion coefficient is achieved when sorting
        # the k groups into k ghettos, ordered by sizes, and then computing the coefficient for
        # the most isolated person in the smallest group

    # the global multiplies the population at each location by the distortion coefficient they experience
    aux["coefs"] = aux["divergence"] * df.sum(axis=1)
    stat = (1 / total_pop) * aux["coefs"].sum()

    out = gpd.GeoDataFrame(gdf, columns=groups, geometry=geoms, crs=geoms.crs)
    out["weighted_distortion"] = aux["coefs"]

    return stat, out


class GlobalDistortion(MultiGroupIndex, SpatialExplicitIndex):
    """Multigroup Global Distortion Index.

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
    normalization: bool
        NOT YET IMPLEMENTED

    Attributes
    ----------
    statistics : pandas.Series
        KL Divergence coefficients
    core_data : a pandas DataFrame
        DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on Bézenac, C., Clark, W. A. V., Olteanu, M., & Randon‐Furling, J. (2022). Measuring and Visualizing Patterns
    of Ethnic Concentration: The Role of Distortion Coefficients. Geographical Analysis, 54(1), 173–196.
    https://doi.org/10.1111/gean.12271

    Reference: :cite:`debezenac2021`.
    """

    def __init__(
        self,
        data,
        groups=None,
        metric="euclidean",
        network=None,
        distance_matrix=None,
        normalize=False,
        **kwargs
    ):
        """Init."""

        MultiGroupIndex.__init__(self, data, groups)
        SpatialExplicitIndex.__init__(self)

        aux = _global_distortion(
            self.data,
            self.groups,
            network=network,
            metric=metric,
            normalize=normalize,
            distance_matrix=distance_matrix,
        )

        self.statistic = aux[0]
        self.data = aux[1]
        self._function = _global_distortion
