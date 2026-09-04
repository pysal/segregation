"""Tests for the Hutchens square root index."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from libpysal.examples import load_example

from segregation.singlegroup import Dissim, Gini, HutchensSqrt

s_map = gpd.read_file(load_example("Sacramento1").get_path("sacramentot2.shp"))
df = s_map[["geometry", "HISP", "TOT_POP"]]


def _reardon_s(x, t):
    """Square root index written in Reardon's unit-share form.

    An independent expression of the same quantity, used to check the
    implementation's group-count form.
    """
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    T = t.sum()
    P = x.sum() / T
    p = np.divide(x, t, out=np.zeros_like(x), where=t > 0)
    return 1 - np.sum((t / T) * np.sqrt(p * (1 - p)) / np.sqrt(P * (1 - P)))


def test_HutchensSqrt():
    index = HutchensSqrt(df, "HISP", "TOT_POP")
    np.testing.assert_almost_equal(index.statistic, 0.07847550)


def test_range():
    """The index is bounded on [0, 1]."""
    index = HutchensSqrt(df, "HISP", "TOT_POP")
    assert 0.0 <= index.statistic <= 1.0


def test_distinct_from_dissim_and_gini():
    """A distinct index, not an alias of one already in the battery."""
    h = HutchensSqrt(df, "HISP", "TOT_POP").statistic
    assert not np.isclose(h, Dissim(df, "HISP", "TOT_POP").statistic)
    assert not np.isclose(h, Gini(df, "HISP", "TOT_POP").statistic)


def test_equals_reardon_square_root_form():
    """O_s in group-count form equals the unit-share form Reardon integrates."""
    index = HutchensSqrt(df, "HISP", "TOT_POP")
    np.testing.assert_almost_equal(
        index.statistic, _reardon_s(df["HISP"], df["TOT_POP"])
    )


def test_complete_integration():
    """Identical composition everywhere gives 0."""
    data = pd.DataFrame({"group": [10.0, 20.0, 30.0], "total": [40.0, 80.0, 120.0]})
    np.testing.assert_almost_equal(HutchensSqrt(data, "group", "total").statistic, 0.0)


def test_complete_segregation():
    """Fully sorted groups give 1."""
    data = pd.DataFrame({"group": [100.0, 0.0], "total": [100.0, 100.0]})
    np.testing.assert_almost_equal(HutchensSqrt(data, "group", "total").statistic, 1.0)


def test_scale_invariance():
    """P1: scaling one group's counts by a positive constant changes nothing."""
    base = pd.DataFrame({"group": [10.0, 40.0, 25.0], "total": [100.0, 80.0, 60.0]})
    scaled = base.copy()
    scaled["group"] = base["group"] * 2
    # total must move with the focal group so the complement is unchanged
    scaled["total"] = scaled["group"] + (base["total"] - base["group"])
    np.testing.assert_almost_equal(
        HutchensSqrt(base, "group", "total").statistic,
        HutchensSqrt(scaled, "group", "total").statistic,
    )


def test_symmetry_in_types():
    """P5: swapping which group is focal leaves the statistic unchanged."""
    swapped = df.copy()
    swapped["OTHER"] = swapped["TOT_POP"] - swapped["HISP"]
    np.testing.assert_almost_equal(
        HutchensSqrt(df, "HISP", "TOT_POP").statistic,
        HutchensSqrt(swapped, "OTHER", "TOT_POP").statistic,
    )


def test_organizational_equivalence():
    """P3: splitting a unit into two of identical composition changes nothing."""
    base = pd.DataFrame({"group": [30.0, 10.0, 50.0], "total": [100.0, 40.0, 90.0]})
    split = pd.DataFrame(
        {
            "group": [15.0, 15.0, 10.0, 50.0],
            "total": [50.0, 50.0, 40.0, 90.0],
        }
    )
    np.testing.assert_almost_equal(
        HutchensSqrt(base, "group", "total").statistic,
        HutchensSqrt(split, "group", "total").statistic,
    )


def test_neighborhood_division():
    """P4: splitting a unit into two of differing composition raises the index.

    This is the property Dissim violates, and the reason this index is worth
    having alongside it.
    """
    base = pd.DataFrame({"group": [40.0, 10.0], "total": [100.0, 100.0]})
    # the first unit is split 50/50 into halves of differing composition, both
    # of which stay on the same side of the global group share
    split = pd.DataFrame({"group": [25.0, 15.0, 10.0], "total": [50.0, 50.0, 100.0]})
    assert (
        HutchensSqrt(split, "group", "total").statistic
        > HutchensSqrt(base, "group", "total").statistic
    )
    # Dissim is unmoved by the very same split
    np.testing.assert_almost_equal(
        Dissim(base, "group", "total").statistic,
        Dissim(split, "group", "total").statistic,
    )


def test_zero_population_units():
    """Empty units contribute nothing rather than producing nan."""
    with_empty = pd.DataFrame(
        {"group": [30.0, 10.0, 0.0], "total": [100.0, 40.0, 0.0]}
    )
    without = pd.DataFrame({"group": [30.0, 10.0], "total": [100.0, 40.0]})
    stat = HutchensSqrt(with_empty, "group", "total").statistic
    assert np.isfinite(stat)
    np.testing.assert_almost_equal(
        stat, HutchensSqrt(without, "group", "total").statistic
    )


def test_degenerate_populations():
    """No focal group (or nothing but) is reported as zero segregation."""
    none = pd.DataFrame({"group": [0.0, 0.0], "total": [100.0, 40.0]})
    every = pd.DataFrame({"group": [100.0, 40.0], "total": [100.0, 40.0]})
    assert HutchensSqrt(none, "group", "total").statistic == 0.0
    assert HutchensSqrt(every, "group", "total").statistic == 0.0


def test_group_larger_than_total_raises():
    bad = pd.DataFrame({"group": [50.0, 10.0], "total": [40.0, 100.0]})
    with pytest.raises(ValueError):
        HutchensSqrt(bad, "group", "total")


def test_spatial_implicit():
    """The egohood version runs and smooths the statistic downward."""
    projected = df.to_crs(df.estimate_utm_crs())
    aspatial = HutchensSqrt(projected, "HISP", "TOT_POP").statistic
    spatial = HutchensSqrt(projected, "HISP", "TOT_POP", distance=2000).statistic
    assert np.isfinite(spatial)
    assert spatial < aspatial
