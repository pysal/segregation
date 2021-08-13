import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial import distance_matrix
from scipy.special import rel_entr as relative_entropy
from scipy.interpolate import interp1d


def divergence_profile(populations, geodataframe):
    """
    Description of the function here.

    Reference to paper here.

    Arguments
    populations : form(s) of input
                  form(s) of input
    geodataframe : form(s) of input
                 form(s) of input
    selectioninput1 : form of input
        Description of what this is and what it does here.
    selectioninput2 : form of input
        Description of what this is and what it does here.

    Returns
    ----------
    Description of the format of return

    Example
    ----------
    Basic written example here, using test data if possible or
    creating simple dataframe to run
    """
    # Extract populations
    populations = geodataframe[populations].values.astype(float)

    # Creating a distance matrix
    centroids = geodataframe.geometry.centroid
    centroid_coords = np.column_stack((centroids.x, centroids.y))
    dist_matrix = distance_matrix(centroid_coords, centroid_coords)

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
                                          R_total_proportions).sum(axis=1)

        # Creating object for population at each distance
        pop_within_dist = obs_cumul_pop.sum(axis=1)

        # Creating an output dataframe
        output = pd.DataFrame().from_dict(dict(
            observation = i,
            distance = distances[sorted_indices],
            divergence = kl_div_profile,
            nearby_population = pop_within_dist
        ))

        # Append (bring together) all outputs into results list
        results.append(output)

    return(pd.concat(results))
