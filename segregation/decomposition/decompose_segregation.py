"""
Decomposition Segregation based Metrics
"""

__author__ = "Renan X. Cortes <renanc@ucr.edu>, Elijah Knaap <elijah.knaap@ucr.edu>, and Sergio J. Rey <sergio.rey@ucr.edu>"


import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from segregation.inference.comparative import _generate_counterfactual

# Including old and new api in __all__ so users can use both

__all__ = ["DecomposeSegregation"]

# The Deprecation calls of the classes are located in the end of this script #


def _decompose_segregation(index1, index2, counterfactual_approach="composition"):
    """Decompose segregation differences into spatial and attribute components.

    Given two segregation indices of the same type, use Shapley decomposition
    to measure whether the differences between index measures arise from
    differences in spatial structure or population structure

    Parameters
    ----------
    index1 : segregation.SegIndex class
        First SegIndex class to compare.
    index2 : segregation.SegIndex class
        Second SegIndex class to compare.
    counterfactual_approach : str, one of
                              ["composition", "share", "dual_composition"]
        The technique used to generate the counterfactual population
        distributions.

    Returns
    -------
    tuple
        (shapley spatial component,
         shapley attribute component,
         core data of index1,
         core data of index2,
         data with counterfactual variables for index1,
         data with counterfactual variables for index2)

    """
    df1 = index1.data.copy()
    df2 = index2.data.copy()

    assert (
        index1._function == index2._function
    ), "Segregation indices must be of the same type"

    counterfac_df1, counterfac_df2 = _generate_counterfactual(
        df1,
        df2,
        index1.group_pop_var,
        index1.total_pop_var,
        index2.group_pop_var,
        index2.total_pop_var,
        counterfactual_approach=counterfactual_approach,
    )

    seg_func = index1._function

    # index for spatial 1, attribute 1
    G_S1_A1 = index1.statistic

    # index for spatial 2, attribute 2
    G_S2_A2 = index2.statistic

    # index for spatial 1 attribute 2 (counterfactual population for structure 1)
    G_S1_A2 = seg_func(
        counterfac_df1, "counterfactual_group_pop", "counterfactual_total_pop"
    )[0]

    # index for spatial 2 attribute 1 (counterfactual population for structure 2)
    G_S2_A1 = seg_func(
        counterfac_df2, "counterfactual_group_pop", "counterfactual_total_pop"
    )[0]

    # take the average difference in spatial structure, holding attributes constant
    C_S = 1 / 2 * (G_S1_A1 - G_S2_A1 + G_S1_A2 - G_S2_A2)

    # take the average difference in attributes, holding spatial structure constant
    C_A = 1 / 2 * (G_S1_A1 - G_S1_A2 + G_S2_A1 - G_S2_A2)

    results = {"s1_a1": G_S1_A1, "s1_a2": G_S1_A2, "s2_a1": G_S2_A1, "s2_a2": G_S2_A2}

    return (
        C_S,
        C_A,
        df1,
        df2,
        counterfac_df1,
        counterfac_df2,
        counterfactual_approach,
        results,
    )


class DecomposeSegregation:
    """Decompose segregation differences into spatial and attribute components.

    Given two segregation indices of the same type, use Shapley decomposition
    to measure whether the differences between index measures arise from
    differences in spatial structure or population structure

    Parameters
    ----------
    index1 : segregation.SegIndex class
        First SegIndex class to compare.
    index2 : segregation.SegIndex class
        Second SegIndex class to compare.
    counterfactual_approach : str, one of {"composition", "share", "dual_composition"}
        The technique used to generate the counterfactual population
        distributions.

    Attributes
    ----------
    c_s : float
        Shapley's Spatial Component of the decomposition
    c_a : float
        Shapley's Attribute Component of the decomposition
    indices : dict
        Dictionary of index values for all four combinations of spatial/attribute data


    """

    def __init__(self, index1, index2, counterfactual_approach="composition"):
        """Initialize class."""
        aux = _decompose_segregation(index1, index2, counterfactual_approach)

        self.c_s = aux[0]
        self.c_a = aux[1]
        self._df1 = aux[2]
        self._df2 = aux[3]
        self._counterfac_df1 = aux[4]
        self._counterfac_df2 = aux[5]
        self._counterfactual_approach = aux[6]
        self.indices = aux[7]

    def plot(
        self,
        plot_type="cdfs",
        figsize=None,
        city_a=None,
        city_b=None,
        cmap="OrRd",
        scheme="equalinterval",
        k=10,
        suptitle_size=16,
        title_size=12,
        savefig=None,
        dpi=300,
    ):
        """Plot maps or CDFs of urban contexts used in calculating the Decomposition class.

        Parameters
        ----------
        plot_type : str, {'cdfs, 'maps'}
            which type of plot to generate. Options include `cdfs` and `maps` by default "cdfs"
        figsize : tuple, optional
            figsize parameter passed to matplotlib.pyplot
        city_a : str, optional
            Name of the first "city" to be used in plotting. If None, defaults to 'City A'
        city_b : str, optional
            Name of the second "city" to be used in plotting. If None, defaults to 'City B'
        cmap : str, optional
            matplotlib colormap used to shade the map, by default "OrRd"
        scheme : str, optional
            pysal.mapclassify classification scheme used to shade the map, by default "equalinterval"
        k : int, optional
            number of classes in pysal.mapclassify classification scheme, by default 10
        suptitle_size : int, optional
            size parameter passed to `matplotlib.Figure.suptitle`, by default 16
        title_size : int, optional
            size parameter passed to `matplotlib.Axes.set_title`, by default 12
        savefig : str, optional
            Location to save the figure if desired. If None, fig will not be saved
        dpi : int, optional
            dpi parameter passed to matplotlib.pyplot, by default 300

        Returns
        -------
        None
            Generates a new matplotlib.Figure instance and optionally saves to disk
        """
        if not city_a:
            city_a = "City A"
        if not city_b:
            city_b = "City B"

        if plot_type == "cdfs":
            if not figsize:
                figsize = (10, 10)
            fig, ax = plt.subplots(figsize=figsize)
            plt.suptitle(
                f"Decomposing differences between\n{city_a} and {city_b}",
                size=suptitle_size,
            )
            plt.title(
                f"Spatial Component = {round(self.c_s, 3)}, Attribute Component: {round(self.c_a, 3)}",
                size=title_size,
            )

            temp_a = self._counterfac_df1.copy()
            temp_a["Location"] = city_a
            temp_b = self._counterfac_df2.copy()
            temp_b["Location"] = city_b
            df = pd.concat([temp_a, temp_b]).reset_index()

            if self._counterfactual_approach == "composition":
                sns.ecdfplot(data=df, x="group_composition", hue="Location", ax=ax)
                return ax

            elif self._counterfactual_approach == "share":
                f = sns.ecdfplot(data=df, x="share", hue="Location", ax=ax)
                return f

            elif self._counterfactual_approach == "dual_composition":
                df["compl"] = 1 - df.group_composition
                f = sns.ecdfplot(data=df, x="group_composition", hue="Location", ax=ax)
                f2 = sns.ecdfplot(data=df, x="compl", hue="Location", ax=ax)
            if savefig:
                plt.savefig(savefig, dpi=dpi)

        if plot_type == "maps":
            if not figsize:
                figsize = (20, 20)
            fig, axs = plt.subplots(2, 2, figsize=figsize)
            plt.suptitle(
                f"Decomposing differences between\n{city_a} and {city_b}",
                size=suptitle_size,
            )
            plt.title(
                f"Spatial Component = {round(self.c_s, 3)}, Attribute Component: {round(self.c_a, 3)}"
            )

            # Original First Context (Upper Left)
            self._counterfac_df1.plot(
                column="group_composition",
                cmap=cmap,
                legend=True,
                scheme=scheme,
                k=k,
                ax=axs[0, 0],
            )
            axs[0, 0].set_title(
                f"{city_a}\nOriginal Composition", fontdict={"fontsize": title_size}
            )
            axs[0, 0].axis("off")

            # Counterfactual First Context (Bottom Left)
            self._counterfac_df1.plot(
                column="counterfactual_composition",
                cmap=cmap,
                scheme=scheme,
                k=k,
                legend=True,
                ax=axs[1, 0],
            )
            axs[1, 0].set_title(
                f"{city_a}\nCounterfactual Composition",
                fontdict={"fontsize": title_size},
            )
            axs[1, 0].axis("off")

            # Counterfactual Second Context (Upper Right)
            self._counterfac_df2.plot(
                column="counterfactual_composition",
                cmap=cmap,
                scheme=scheme,
                k=k,
                legend=True,
                ax=axs[0, 1],
            )
            axs[0, 1].set_title(
                f"{city_b}\nCounterfactual Composition",
                fontdict={"fontsize": title_size},
            )
            axs[0, 1].axis("off")

            # Original Second Context (Bottom Right)
            self._counterfac_df2.plot(
                column="group_composition",
                cmap=cmap,
                scheme=scheme,
                k=k,
                legend=True,
                ax=axs[1, 1],
            )
            axs[1, 1].set_title(
                f"{city_b}\nOriginal Composition", fontdict={"fontsize": title_size}
            )
            axs[1, 1].axis("off")
            if savefig:
                plt.savefig(savefig, dpi=dpi)
            return axs
