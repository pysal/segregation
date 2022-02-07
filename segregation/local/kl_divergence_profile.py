import numpy as np
import geopandas as gpd
import pandas as pd

from scipy.spatial.distance import pdist, squareform
from scipy.special import rel_entr as relative_entropy

from .._base import MultiGroupIndex, SpatialExplicitIndex
from ..network import compute_travel_cost_matrix


def _kl_divergence_profile(
    gdf, groups, network=None, metric="euclidean", coefs=True, normalize=False
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
    metric: str
        Acceptable inputs to `scipy.spatial.distance.pdist` - including:
        ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’,
        ‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’, ‘jensenshannon’,
        ‘kulsinski’, ‘mahalanobis’, ‘matching’, ‘minkowski’, ‘rogerstanimoto’,
        ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’.
        Distance metric for calculating pairwise distances,
        using `scipy.spatial.distance.pdist` - 'euclidean' by default.
    coefs: bool (default: True)
        whether to return KL divergence coefficients. If True, the function will
        return a geodataframe with one
    normalization:
        NOT YET IMPLEMENTED


    Returns
    ----------
    observation : an identifier of the area that forms the centre of the aggregation
                  of population, from which the divergence is calculated.
    distance : how far the most recently aggregated area is from the 'observation'
               area, starting at zero for each observation to represent that only the
               'observation' area being aggregated.
    divergence : the KL divergence measure, between the aggregated population and the
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
    geoms = gdf[gdf.geometry.name]
    centroids = gdf.geometry.centroid
    df = gdf[groups].values

    coordinates = np.column_stack((centroids.x, centroids.y))

    # If given a pandana network, use shortest network distance, otherwise use scikit
    if network:
        dist_matrix = compute_travel_cost_matrix(gdf, gdf, network).values
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
    if coefs:
        aux = gpd.GeoDataFrame(
            aux.groupby("observation").sum()[["divergence"]], geometry=geoms
        )
        if normalize:
            raise Exception("Not yet implemented")
            # Need to write a routine to determine the scaling factor... From the paper:

            # the maximum distortion coefficient in a theoretical extreme case of segregation. 
            # Theoretically, the maximal-segregation distortion coefficient is achieved when sorting 
            # the k groups into k ghettos, ordered by sizes, and then computing the coefficient for
            # the most isolated person in the smallest group

    return aux


class LocalKLDivergence(MultiGroupIndex, SpatialExplicitIndex):
    """Multigroup Local KL Divergence Coefficients.

    Parameters
    ----------
    data : pandas.DataFrame or geopandas.GeoDataFrame, required
        dataframe or geodataframe if spatial index holding data for location of interest
    groups : list, required
        list of columns on dataframe holding population totals for each group
    network : pandana.Network
        pandana Network object representing the study area
    coefs: bool
        whether to return KL Divergence coefficients for each observation (instead of a matrix
        of values for each observation)

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
        coefs=True,
        normalize=False,
    ):
        """Init."""

        MultiGroupIndex.__init__(self, data, groups)
        SpatialExplicitIndex.__init__(self)

        aux = _kl_divergence_profile(
            self.data,
            self.groups,
            network=network,
            metric=metric,
            coefs=coefs,
            normalize=normalize,
        )

        self.statistics = aux["divergence"]
        self.data = aux
        self._function = _kl_divergence_profile
