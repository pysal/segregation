import geopandas as gpd

from .._base import MultiGroupIndex, SpatialExplicitIndex
from ..dynamics import compute_divergence_profiles
from ..util.normalization import _maximal_segregation_distortion


def _global_distortion(
    gdf,
    groups,
    network=None,
    metric="euclidean",
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
    network: pandarm.Network object (optional)
        A pandarm Network object used to compute distance between observations
    distance_matrix: numpy.array
        numpy array of distances between observations in the dataset
    normalize: bool
        If True, standardize the measure by the theoretical maximum segregation
        value for this dataset
    n_seeds: int (optional; 4 by default)
        Number of corner positions used to build the maximally-segregated
        reference landscape. The normalization constant is the maximum over all
        of them, so raising this yields a tighter estimate of the theoretical
        maximum at the cost of one extra divergence profile per seed.
        Ignored when ``normalize`` is False.


    Returns
    ----------
    statistic : float
        the global distortion index
    gdf: geopands.GeoDataFrame
        a geodataframe of input data used to compute the statistic
    N : float or None
        the normalization constant, or None when ``normalize`` is False

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

    N = None
    if normalize:
        N = _maximal_segregation_distortion(
            gdf,
            groups,
            metric=metric,
            network=network,
            distance_matrix=distance_matrix,
            n_seeds=n_seeds,
        )
        aux["divergence"] = aux["divergence"] / N

    # the global multiplies the population at each location by the distortion coefficient they experience
    aux["coefs"] = aux["divergence"] * df.sum(axis=1)
    stat = (1 / total_pop) * aux["coefs"].sum()

    out = gpd.GeoDataFrame(gdf, columns=groups, geometry=geoms, crs=geoms.crs)
    out["weighted_distortion"] = aux["coefs"]

    return stat, out, N


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
    network: pandarm.Network object (optional, None by default)
        A pandarm Network object used to compute distance between observations
    distance_matrix: numpy.array (optional; None by default)
        numpy array of distances between observations in the dataset
    normalize: bool (optional; True by default)
        If True, divide by the theoretical maximum Distortion: the largest
        *local* coefficient in the most segregated configuration possible given
        the study region's group totals. Note that 1.0 is not reachable for the
        global index -- see Notes.
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
        index, or None when ``normalize`` is False.

    Notes
    -----
    The index is the population-weighted mean of the local Distortion
    coefficients (de Bézenac et al. 2022, Eq. 4).

    When ``normalize`` is True, the divisor is the theoretical maximum
    Distortion, which the source defines for a *local unit*: "the maximum local
    distortion in the most segregated configuration possible given the global
    distribution of the population" (Note 4). Dividing a population-weighted
    mean by that local maximum means the normalized global index does **not**
    reach 1.0 even for a maximally segregated landscape. This is intended, not a
    defect -- the authors state plainly that "the Global Distortion upper bound
    is comparably out of reach" because "the normalization process is in fact
    formulated for the local unit ... and does not refer to a set of possible
    global configurations (unlike the two others) but to the most segregated
    unit of an ethnically concentric city" (p. 10).

    Consequently the normalized global value is not comparable to Dissimilarity
    or the H-index on a shared 0-1 scale. To compare across cities or over time,
    the source compares *relative variation* between measurements (the gradient
    of each measure) rather than effective values. Do not "fix" this by
    normalizing with the global index of the extreme configuration -- that would
    depart from the published definition.

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
        normalize=True,
        n_seeds=4,
        **kwargs,
    ):
        """Init."""

        MultiGroupIndex.__init__(self, data, groups)
        SpatialExplicitIndex.__init__(self)

        stat, data, N = _global_distortion(
            self.data,
            self.groups,
            network=network,
            metric=metric,
            normalize=normalize,
            distance_matrix=distance_matrix,
            n_seeds=n_seeds,
        )

        self.statistic = stat
        self.data = data
        self._function = _global_distortion
        self.normalization_constant = N
