"""Tests for Reardon's rank-order segregation indices."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from libpysal.examples import load_example
from scipy.integrate import quad

from segregation.rankorder import (
    RankOrderInfoTheory,
    RankOrderSqrt,
    RankOrderVarianceRatio,
)
from segregation.rankorder._utils import (
    _segregation_profile,
    deltas_info_theory,
    deltas_sqrt,
    deltas_variance_ratio,
    entropy_weight,
    sqrt_weight,
    variance_ratio_weight,
)
from segregation.singlegroup import CorrelationR, Entropy, HutchensSqrt

CATEGORIES = [f"cat_{i}" for i in range(1, 17)]


def _synthetic_tracts(seed=42, n_tracts=100, n_cats=16):
    """Deterministic tract-level counts across ordered categories."""
    rng = np.random.default_rng(seed)
    tract_means = rng.normal(8, 3, n_tracts)
    rows = []
    for mean in tract_means:
        probs = np.exp(-0.5 * ((np.arange(n_cats) - mean) / 3) ** 2)
        probs = probs / probs.sum()
        rows.append(rng.multinomial(rng.integers(500, 2000), probs))
    return pd.DataFrame(rows, columns=[f"cat_{i}" for i in range(1, n_cats + 1)])


df = _synthetic_tracts()

s_map = gpd.read_file(load_example("Sacramento1").get_path("sacramentot2.shp"))
SAC_GROUPS = ["HISP", "BLACK", "WHITE"]


# ---------------------------------------------------------------------------
# the delta coefficients: the analytic core of the method
# ---------------------------------------------------------------------------


def test_delta_table_info_theory():
    """Reardon's Table 1 values for H^R."""
    np.testing.assert_allclose(
        deltas_info_theory(4), [1, 1 / 2, 11 / 36, 5 / 24, 137 / 900]
    )


def test_delta_table_variance_ratio():
    np.testing.assert_allclose(deltas_variance_ratio(4), [1, 1 / 2, 0.3, 0.2, 1 / 7])


def test_delta_table_sqrt():
    np.testing.assert_allclose(deltas_sqrt(4), [1, 1 / 2, 0.3125, 0.21875, 0.1640625])


@pytest.mark.parametrize(
    "weight_func,delta_func",
    [
        (entropy_weight, deltas_info_theory),
        (variance_ratio_weight, deltas_variance_ratio),
        (sqrt_weight, deltas_sqrt),
    ],
)
def test_deltas_match_numerical_integration(weight_func, delta_func):
    """Each delta_m equals the normalized integral of w(p) * p**m it stands for.

    This validates the closed forms, the normalization, and the monomial basis
    together, independent of any index class.
    """
    normalizer = quad(lambda p: weight_func(p), 0, 1)[0]
    expected = [
        quad(lambda p, m=m: weight_func(p) * p**m, 0, 1)[0] / normalizer
        for m in range(7)
    ]
    np.testing.assert_allclose(delta_func(6), expected, atol=1e-9)


@pytest.mark.parametrize(
    "index_class,weight_func",
    [
        (RankOrderInfoTheory, entropy_weight),
        (RankOrderVarianceRatio, variance_ratio_weight),
        (RankOrderSqrt, sqrt_weight),
    ],
)
def test_analytic_equals_numerical_integral(index_class, weight_func):
    """delta' * beta equals numerically integrating the fitted polynomial.

    The whole point of the delta coefficients is that they replace a numerical
    integration; this asserts the replacement is exact.
    """
    index = index_class(df, CATEGORIES, degree=4)
    coefs = index.coefficients

    def fitted(p):
        return sum(c * p**m for m, c in enumerate(coefs))

    normalizer = quad(weight_func, 0, 1)[0]
    numeric = quad(lambda p: weight_func(p) * fitted(p), 0, 1)[0] / normalizer
    np.testing.assert_allclose(index.statistic, numeric, rtol=1e-8)


# ---------------------------------------------------------------------------
# the pairwise indices each rank-order index integrates
# ---------------------------------------------------------------------------


def test_pairwise_sqrt_is_reardon_square_root():
    """HutchensSqrt is Reardon's pairwise S(p) written in group-count form."""
    profile = _segregation_profile(df, CATEGORIES, HutchensSqrt)
    cumulative = df[CATEGORIES].cumsum(axis=1)
    total = cumulative[CATEGORIES[-1]].to_numpy(dtype=float)

    for row in profile.itertuples():
        x = cumulative[row.threshold].to_numpy(dtype=float)
        T = total.sum()
        P = x.sum() / T
        p = np.divide(x, total, out=np.zeros_like(x), where=total > 0)
        expected = 1 - np.sum((total / T) * np.sqrt(p * (1 - p)) / np.sqrt(P * (1 - P)))
        np.testing.assert_allclose(row.statistic, expected)


def test_pairwise_variance_ratio_is_correlation_ratio():
    """CorrelationR is the variance ratio sum_j t_j (p_j - P)^2 / (T P (1-P))."""
    profile = _segregation_profile(df, CATEGORIES, CorrelationR)
    cumulative = df[CATEGORIES].cumsum(axis=1)
    total = cumulative[CATEGORIES[-1]].to_numpy(dtype=float)

    for row in profile.itertuples():
        x = cumulative[row.threshold].to_numpy(dtype=float)
        T = total.sum()
        P = x.sum() / T
        p = np.divide(x, total, out=np.zeros_like(x), where=total > 0)
        expected = np.sum(total * (p - P) ** 2) / (T * P * (1 - P))
        np.testing.assert_allclose(row.statistic, expected)


def test_profile_shape_and_bounds():
    """K categories give K-1 thresholds, with p strictly increasing in (0, 1)."""
    index = RankOrderInfoTheory(df, CATEGORIES)
    assert len(index.profile) == len(CATEGORIES) - 1
    p = index.profile["p"].to_numpy()
    assert np.all((p > 0) & (p < 1))
    assert np.all(np.diff(p) > 0)


# ---------------------------------------------------------------------------
# estimates
# ---------------------------------------------------------------------------


def test_rank_order_info_theory():
    index = RankOrderInfoTheory(df, CATEGORIES)
    np.testing.assert_almost_equal(index.statistic, 0.20187570)


def test_rank_order_variance_ratio():
    index = RankOrderVarianceRatio(df, CATEGORIES)
    np.testing.assert_almost_equal(index.statistic, 0.22296188)


def test_rank_order_sqrt():
    index = RankOrderSqrt(df, CATEGORIES)
    np.testing.assert_almost_equal(index.statistic, 0.16716787)


def test_attributes():
    index = RankOrderInfoTheory(df, CATEGORIES, degree=4)
    assert index.degree == 4
    assert len(index.coefficients) == 5
    assert len(index.deltas) == 5
    assert index.standard_error > 0
    assert 0 <= index.r_squared <= 1
    assert index.index_type == "rankorder"
    assert list(index.profile.columns) == ["threshold", "p", "statistic"]


def test_complete_integration_is_zero():
    """Identical category composition everywhere means no sorting."""
    counts = np.tile([10.0, 20.0, 30.0, 40.0], (25, 1)) * np.arange(1, 26)[:, None]
    data = pd.DataFrame(counts, columns=["a", "b", "c", "d"])
    for cls in (RankOrderInfoTheory, RankOrderVarianceRatio, RankOrderSqrt):
        index = cls(data, ["a", "b", "c", "d"], degree=2)
        np.testing.assert_allclose(index.statistic, 0.0, atol=1e-12)


def test_sorting_increases_the_index():
    """Sorting households by category across units raises every index."""
    mixed = pd.DataFrame(
        {"a": [100.0] * 4, "b": [100.0] * 4, "c": [100.0] * 4, "d": [100.0] * 4}
    )
    sorted_ = pd.DataFrame(
        {
            "a": [340.0, 40.0, 10.0, 10.0],
            "b": [40.0, 340.0, 10.0, 10.0],
            "c": [10.0, 10.0, 340.0, 40.0],
            "d": [10.0, 10.0, 40.0, 340.0],
        }
    )
    for cls in (RankOrderInfoTheory, RankOrderVarianceRatio, RankOrderSqrt):
        assert (
            cls(sorted_, ["a", "b", "c", "d"], degree=2).statistic
            > cls(mixed, ["a", "b", "c", "d"], degree=2).statistic
        )


def test_scale_invariance():
    """Scaling every unit's counts by a constant leaves the index unchanged."""
    scaled = df.copy()
    scaled[CATEGORIES] = scaled[CATEGORIES] * 7
    for cls in (RankOrderInfoTheory, RankOrderVarianceRatio, RankOrderSqrt):
        np.testing.assert_allclose(
            cls(df, CATEGORIES).statistic, cls(scaled, CATEGORIES).statistic
        )


def test_organizational_equivalence():
    """Splitting a unit into two of identical composition changes nothing."""
    base = pd.DataFrame(
        {"a": [40.0, 10.0, 5.0], "b": [10.0, 40.0, 5.0], "c": [5.0, 5.0, 40.0]}
    )
    split = pd.DataFrame(
        {
            "a": [20.0, 20.0, 10.0, 5.0],
            "b": [5.0, 5.0, 40.0, 5.0],
            "c": [2.5, 2.5, 5.0, 40.0],
        }
    )
    # 3 categories give only 2 thresholds, so the fit must be linear
    for cls in (RankOrderInfoTheory, RankOrderVarianceRatio, RankOrderSqrt):
        np.testing.assert_allclose(
            cls(base, ["a", "b", "c"], degree=1).statistic,
            cls(split, ["a", "b", "c"], degree=1).statistic,
        )


def test_degree_stability():
    """The estimate is stable across the polynomial orders Reardon recommends."""
    estimates = [
        RankOrderInfoTheory(df, CATEGORIES, degree=d).statistic for d in range(2, 7)
    ]
    assert np.ptp(estimates) < 0.01


def test_threshold_invariance():
    """Collapsing adjacent categories barely moves the estimate.

    Insensitivity to how categories are cut is the property that motivates the
    rank-order approach in the first place.
    """
    collapsed = pd.DataFrame(
        {
            f"pair_{i}": df[CATEGORIES[2 * i]] + df[CATEGORIES[2 * i + 1]]
            for i in range(8)
        }
    )
    full = RankOrderInfoTheory(df, CATEGORIES).statistic
    coarse = RankOrderInfoTheory(collapsed, list(collapsed.columns)).statistic
    np.testing.assert_allclose(full, coarse, atol=0.005)


# ---------------------------------------------------------------------------
# spatial versions
# ---------------------------------------------------------------------------


def test_spatial_implicit_distance():
    """Passing a distance lags every ordered column into an egohood."""
    gdf = s_map.to_crs(s_map.estimate_utm_crs())
    aspatial = RankOrderInfoTheory(gdf, SAC_GROUPS, degree=1)
    spatial = RankOrderInfoTheory(gdf, SAC_GROUPS, degree=1, distance=2000)
    assert np.isfinite(spatial.statistic)
    assert spatial.spatial_type == "implicit"
    assert spatial.statistic < aspatial.statistic


def test_spatial_implicit_weights():
    """The libpysal weights path works alongside the distance path."""
    from libpysal.weights import Kernel

    gdf = s_map.to_crs(s_map.estimate_utm_crs()).reset_index(drop=True)
    w = Kernel.from_dataframe(gdf, bandwidth=2000, function="triangular")
    index = RankOrderInfoTheory(gdf, SAC_GROUPS, degree=1, w=w)
    assert np.isfinite(index.statistic)


def test_spatial_all_three_indices():
    gdf = s_map.to_crs(s_map.estimate_utm_crs())
    for cls in (RankOrderInfoTheory, RankOrderVarianceRatio, RankOrderSqrt):
        index = cls(gdf, SAC_GROUPS, degree=1, distance=2000)
        assert np.isfinite(index.statistic)


# ---------------------------------------------------------------------------
# validation and edge cases
# ---------------------------------------------------------------------------


def test_empty_category_is_dropped_not_fatal():
    """A category nobody occupies costs a threshold but is not an error."""
    padded = df.copy()
    padded.insert(0, "empty_low", 0.0)
    index = RankOrderInfoTheory(padded, ["empty_low"] + CATEGORIES)
    assert np.isfinite(index.statistic)
    # the empty leading category yields p == 0 and carries no information
    assert "empty_low" not in set(index.profile["threshold"])
    np.testing.assert_allclose(
        index.statistic, RankOrderInfoTheory(df, CATEGORIES).statistic
    )


def test_degree_too_high_raises():
    small = df[CATEGORIES[:4]]
    with pytest.raises(ValueError, match="usable thresholds"):
        RankOrderInfoTheory(small, CATEGORIES[:4], degree=6)


def test_too_few_categories_raises():
    with pytest.raises(ValueError, match="at least 3 ordered categories"):
        RankOrderInfoTheory(df, CATEGORIES[:2])


def test_groups_must_be_a_list():
    with pytest.raises(TypeError):
        RankOrderInfoTheory(df, "cat_1")


def test_missing_column_raises():
    with pytest.raises(ValueError, match="not present"):
        RankOrderInfoTheory(df, ["cat_1", "cat_2", "nope"])


def test_negative_counts_raise():
    bad = df.copy()
    bad.loc[0, "cat_1"] = -5
    with pytest.raises(ValueError, match="non-negative"):
        RankOrderInfoTheory(bad, CATEGORIES)


def test_bad_degree_raises():
    with pytest.raises(ValueError, match="positive integer"):
        RankOrderInfoTheory(df, CATEGORIES, degree=0)


def test_zero_population_raises():
    empty = pd.DataFrame({"a": [0.0, 0.0], "b": [0.0, 0.0], "c": [0.0, 0.0]})
    with pytest.raises(ValueError, match="zero"):
        RankOrderInfoTheory(empty, ["a", "b", "c"], degree=1)


def test_pairwise_classes_are_the_expected_ones():
    """Each rank-order index integrates the pairwise index it claims to."""
    for cls, pairwise in [
        (RankOrderInfoTheory, Entropy),
        (RankOrderVarianceRatio, CorrelationR),
        (RankOrderSqrt, HutchensSqrt),
    ]:
        expected = _segregation_profile(df, CATEGORIES, pairwise)
        actual = cls(df, CATEGORIES).profile
        np.testing.assert_allclose(actual["statistic"], expected["statistic"])


# ---------------------------------------------------------------------------
# agreement with the reference implementation
# ---------------------------------------------------------------------------


def _notebook_pipeline(counts, cats, pairwise, weight_func, delta_func, degree=4):
    """Reardon's pipeline written out longhand, as in the reference notebook.

    Deliberately independent of `segregation.rankorder`: it builds the design
    with sklearn's PolynomialFeatures and fits with statsmodels directly.
    """
    import statsmodels.api as sm
    from sklearn.preprocessing import PolynomialFeatures

    total = counts[cats].sum(axis=1)
    grand_total = total.sum()

    stats, ps = [], []
    for k in range(1, len(cats)):
        group = counts[cats[:k]].sum(axis=1)
        frame = pd.DataFrame({"group": group, "total": total})
        stats.append(pairwise(frame, "group", "total").statistic)
        ps.append(group.sum() / grand_total)

    p = np.array(ps)
    design = PolynomialFeatures(degree=degree, include_bias=True).fit_transform(
        p.reshape(-1, 1)
    )
    model = sm.WLS(np.array(stats), design, weights=weight_func(p) ** 2).fit()
    return float(np.dot(delta_func(degree), model.params))


@pytest.mark.parametrize(
    "index_class,pairwise,weight_func,delta_func",
    [
        (RankOrderInfoTheory, Entropy, entropy_weight, deltas_info_theory),
        (
            RankOrderVarianceRatio,
            CorrelationR,
            variance_ratio_weight,
            deltas_variance_ratio,
        ),
        (RankOrderSqrt, HutchensSqrt, sqrt_weight, deltas_sqrt),
    ],
)
def test_matches_longhand_reference_pipeline(
    index_class, pairwise, weight_func, delta_func
):
    """The classes reproduce the longhand pipeline exactly."""
    expected = _notebook_pipeline(df, CATEGORIES, pairwise, weight_func, delta_func)
    np.testing.assert_allclose(
        index_class(df, CATEGORIES, degree=4).statistic, expected, rtol=1e-12
    )
