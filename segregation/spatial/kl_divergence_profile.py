import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist, squareform
from scipy.special import rel_entr as relative_entropy


def kl_divergence_profile(populations, coordinates = None, metric = 'euclidean'):
    """
    A segregation metric, using Kullback-Leiber (KL) divergence to quantify the difference in the population characteristics between (1) an area and (2) the total population.

    This function utilises the methodology proposed in Olteanu et al. (2019): 'Segregation through the multiscalar lens'. Which can be found here: https://doi.org/10.1073/pnas.1900192116

    Arguments
    ----------
    populations : GeoPandas GeoDataFrame object
                  NumPy Array object
                  Population information of raw group numbers (not percentages) to be included in the analysis.
    coordinates : GeoPandas GeoSeries object
                  NumPy Array object
                  Spatial information relating to the areas to be included in the analysis.
    metric : Acceptable inputs to `scipy.spatial.distance.pdist` - including: ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’, ‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’, ‘jensenshannon’, ‘kulsinski’, ‘mahalanobis’, ‘matching’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’.
             Distance metric for calculating pairwise distances, using `scipy.spatial.distance.pdist` - 'euclidean' by default.

    Returns
    ----------
    Returns a concatenated object of Pandas dataframes. Each dataframe contains a set of divergence levels between an area and the total population. These areas become consecutively larger, starting from a single location and aggregating outward from this location, until the area represents the total population. Thus, together the divergence levels within a dataframe represent a profile of divergence from an area. The concatenated object is the collection of these divergence profiles for every areas within the total population.

    Example
    ----------
    Basic written example here, using test data if possible or
    creating simple dataframe to run
    cincin = load_example('Cincinnati')
    cincin.get_file_list()
    cincin_df = gpd.read_file(cincin.get_path('cincinnati.shp'))
    cincin_ethnicity = cincin_df[["WHITE", "BLACK", "AMINDIAN", "ASIAN", "HAWAIIAN", "OTHER_RACE", "geometry"]]
    cincin_ethnicity.head()
    kl_divergence_profile(cincin_ethnicity)
    """
    # Store the observation index to return with the results
    if hasattr(populations, 'index'):
        indices = populations.index
    else:
        indices = np.arange(len(populations))
    
    # Check for geometry present in populations argument
    if hasattr(populations, 'geometry'):
        if coordinates is None:
            coordinates = populations.geometry
        populations = populations.drop(populations.geometry.name, axis = 1).values
    populations = np.asarray(populations)
    
    #  Creating consistent coordinates - GeoSeries input
    if hasattr(coordinates,'geometry'):
        centroids = coordinates.geometry.centroid
        coordinates = np.column_stack((centroids.x, centroids.y)) 
    #  Creating consistent coordinates - Array input
    else:
        assert len(coordinates) == len(populations), "Length of coordinates input needs to be of the same length as populations input"
       
    # Creating distance matrix using defined metric (default euclidean distance)
    dist_matrix = squareform(pdist(coordinates, metric = metric))

    # Preparing list for results
    results = []

    # Loop to calculate KL divergence profile
    for (i, distances) in enumerate(dist_matrix):

        # Creating the q and r objects
        sorted_indices = np.argsort(distances)
        cumul_pop_by_group = np.cumsum(populations[sorted_indices], axis = 0)
        obs_cumul_pop = np.sum(cumul_pop_by_group, axis = 1)[:, np.newaxis]
        Q_cumul_proportions = cumul_pop_by_group / obs_cumul_pop
        total_pop_by_group = np.sum(populations, axis = 0, keepdims = True)
        total_pop = np.sum(populations)
        R_total_proportions = total_pop_by_group / total_pop

        # Input q and r objects into relative entropy (KL divergence) function
        kl_div_profile = relative_entropy(Q_cumul_proportions,
                                          R_total_proportions).sum(axis = 1)

        # Creating object for population at each distance
        pop_within_dist = obs_cumul_pop.sum(axis=1)

        # Creating an output dataframe
        output = pd.DataFrame().from_dict(dict(
            observation = indices[i],
            distance = distances[sorted_indices],
            divergence = kl_div_profile,
            nearby_population = pop_within_dist
        ))

        # Append (bring together) all outputs into results list
        results.append(output)

    return(pd.concat(results))

