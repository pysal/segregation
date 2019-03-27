import matplotlib.pyplot as plt


def lorenz(group_share1, group_share2, label1='', label2=''):
    """Plot Lorenz curves for two series
    Convenience function for comparing inequality between two series by
    plotting their lorenz curves on the same graph

    Parameters
    ----------
    group_share1 : pd.Series
        pandas series with variable of interest. This is typically a population
        share such as percent native american
    group_share2 : pd.Series
        pandas series with variable of interest. This is typically a population
        share such as percent native american

    Returns
    -------
    type
        matplotlib Figure.

    """
    plt.step(group_share1.sort_values(), group_share1.rank(pct=True).sort_values(), label=label1)
    plt.step(group_share2.sort_values(), group_share2.rank(pct=True).sort_values(), label=label2)
    if (label1 != '' or label2 != ''):
        plt.legend()
    plt.show()
    fig = plt.gcf()
    return fig
