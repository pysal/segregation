"""Shared machinery for Reardon's rank-order segregation indices.

A rank-order index summarizes segregation across the whole distribution of an
ordered variable. For each of the K-1 thresholds implied by K ordered
categories, the population is dichotomized into "at or below" and "above", and
a pairwise binary segregation index Lambda(p_k) is computed, where p_k is the
share of the population at or below threshold k. A polynomial is fit through
the resulting points by weighted least squares, and the index is the integral
of that polynomial against a weight function -- which Reardon solved
analytically, so the integral reduces to a dot product with a vector of delta
coefficients.

Reference: :cite:`reardon2011measures`.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from geopandas import GeoDataFrame
from scipy.special import comb

__all__ = [
    "entropy_weight",
    "variance_ratio_weight",
    "sqrt_weight",
    "deltas_info_theory",
    "deltas_variance_ratio",
    "deltas_sqrt",
]


# ---------------------------------------------------------------------------
# weight functions
#
# each is normalized so that its integral over (0, 1) is used to normalize the
# corresponding delta coefficients. WLS is invariant to a global rescaling of
# the weights, so the normalization matters only for the deltas.
# ---------------------------------------------------------------------------


def entropy_weight(p):
    """Entropy weight E(p) used by the rank-order information theory index."""
    p = np.asarray(p, dtype=float)
    return -2 * (p * np.log(p) + (1 - p) * np.log(1 - p))


def variance_ratio_weight(p):
    """Weight I(p) used by the rank-order variance ratio index."""
    p = np.asarray(p, dtype=float)
    return 4 * p * (1 - p)


def sqrt_weight(p):
    """Weight V(p) used by the rank-order square root index."""
    p = np.asarray(p, dtype=float)
    return 2 * np.sqrt(p * (1 - p))


# ---------------------------------------------------------------------------
# delta coefficients
#
# delta_m is the analytic solution to the normalized integral of w(p) * p**m
# over (0, 1). Reardon tabulates the information theory values (1, 1/2, 11/36,
# 5/24, 137/900, ...).
# ---------------------------------------------------------------------------


def deltas_info_theory(degree):
    """Delta coefficients for the rank-order information theory index."""
    deltas = []
    for m in range(degree + 1):
        tail = sum(
            (-1) ** (m - n) * comb(m, n) / (m - n + 2) ** 2 for n in range(m + 1)
        )
        deltas.append(2 / (m + 2) ** 2 + 2 * tail)
    return np.array(deltas)


def deltas_variance_ratio(degree):
    """Delta coefficients for the rank-order variance ratio index."""
    return np.array([6 / ((m + 2) * (m + 3)) for m in range(degree + 1)])


def deltas_sqrt(degree):
    """Delta coefficients for the rank-order square root index."""
    deltas = []
    for m in range(degree + 1):
        product = 1.0
        for n in range(m + 1):
            product *= (2 * n + 1) / (2 * n + 4)
        deltas.append(4 * product)
    return np.array(deltas)


# ---------------------------------------------------------------------------
# estimation
# ---------------------------------------------------------------------------


def _segregation_profile(data, groups, index_class):
    """Compute the pairwise segregation index at each ordered threshold.

    Parameters
    ----------
    data : pandas.DataFrame or geopandas.GeoDataFrame
        dataframe holding population counts for each ordered category
    groups : list
        names of the ordered category columns, lowest to highest
    index_class : class
        a segregation.singlegroup index class used as the pairwise index

    Returns
    -------
    pandas.DataFrame
        one row per usable threshold with columns `threshold`, `p`, and
        `statistic`
    """
    counts = data[groups]
    cumulative = counts.cumsum(axis=1)
    total = cumulative[groups[-1]]
    grand_total = total.sum()

    if grand_total <= 0:
        raise ValueError("total population across all categories is zero")

    rows = []
    # the last category is not a threshold: everyone is at or below it
    for threshold in groups[:-1]:
        group = cumulative[threshold]
        p = group.sum() / grand_total

        # a threshold with nobody on one side carries zero weight and leaves
        # the pairwise index undefined, so it contributes nothing
        if not 0 < p < 1:
            continue

        frame = pd.DataFrame({"group": group, "total": total})
        statistic = index_class(frame, "group", "total").statistic

        if not np.isfinite(statistic):
            continue

        rows.append({"threshold": threshold, "p": p, "statistic": statistic})

    return pd.DataFrame(rows, columns=["threshold", "p", "statistic"])


def _wls_poly(p, y, weights, degree):
    """Fit a polynomial in the raw monomial basis by weighted least squares.

    The design must stay in the raw monomial basis [1, p, p**2, ...] because
    the delta coefficients are integrals of p**m. Substituting an orthogonal or
    centered basis without transforming the deltas would silently return the
    wrong statistic.
    """
    design = np.vander(np.asarray(p, dtype=float), degree + 1, increasing=True)
    return sm.WLS(np.asarray(y, dtype=float), design, weights=weights).fit()


def _rank_order(data, groups, degree, index_class, weight_func, delta_func):
    """Estimate a rank-order segregation index.

    Returns
    -------
    tuple
        (statistic, profile, deltas, fitted WLS results)
    """
    if int(degree) != degree or degree < 1:
        raise ValueError(f"`degree` must be a positive integer (got {degree!r})")
    degree = int(degree)

    profile = _segregation_profile(data, groups, index_class)

    if len(profile) < degree + 1:
        raise ValueError(
            f"a polynomial of degree {degree} requires at least {degree + 1} "
            f"usable thresholds, but only {len(profile)} are available. Pass a "
            "smaller `degree` or use data with more ordered categories."
        )

    p = profile["p"].to_numpy()
    # Reardon specifies WLS weights of w(p)**2, so the polynomial fits best
    # where the integrand contributes most
    model = _wls_poly(p, profile["statistic"].to_numpy(), weight_func(p) ** 2, degree)

    deltas = delta_func(degree)
    statistic = float(deltas @ model.params)

    return statistic, profile, deltas, model


def _core_data(data, groups):
    """Return the columns used to perform the estimate, with geometry if present."""
    core = data[groups]
    if isinstance(data, GeoDataFrame):
        core = data[[data.geometry.name]].join(core)
    return core
