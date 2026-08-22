import geopandas as gpd
import numpy as np
import pytest
from libpysal.examples import load_example
from segregation.local import LocalDistortion
from segregation.multigroup import GlobalDistortion
from segregation.multigroup.multi_global_distortion import _global_distortion

GROUPS = ["WHITE", "BLACK", "ASIAN", "HISP"]


@pytest.fixture
def s_map():
    m = gpd.read_file(load_example("Sacramento1").get_path("sacramentot2.shp"))
    return m.to_crs(m.estimate_utm_crs())


@pytest.fixture
def segregated_strip():
    """Twelve equally-populated units in a line, three contiguous groups."""
    from shapely.geometry import Point

    n, per_unit = 12, 10.0
    pops = {"a": np.zeros(n), "b": np.zeros(n), "c": np.zeros(n)}
    pops["a"][0:2] = per_unit
    pops["b"][2:6] = per_unit
    pops["c"][6:12] = per_unit
    return gpd.GeoDataFrame(pops, geometry=[Point(i, 0) for i in range(n)], crs=3857)


def test_GlobalDistortion(s_map):
    """Regression value for the unnormalized global index."""
    index = GlobalDistortion(s_map, groups=GROUPS, normalize=False)
    np.testing.assert_almost_equal(index.statistic, 15.28564652573017, decimal=7)


def test_returns_three_tuple_regardless_of_normalize(s_map):
    """The return arity must not depend on a keyword argument."""
    assert len(_global_distortion(s_map, GROUPS)) == 3
    assert len(_global_distortion(s_map, GROUPS, normalize=True)) == 3


def test_normalization_constant_none_when_not_normalized(s_map):
    index = GlobalDistortion(s_map, groups=GROUPS, normalize=False)
    assert index.normalization_constant is None


def test_normalized_is_raw_over_n(s_map):
    """Normalizing must rescale the index by exactly 1/N."""
    raw = GlobalDistortion(s_map, groups=GROUPS, normalize=False)
    norm = GlobalDistortion(s_map, groups=GROUPS, normalize=True)
    assert norm.normalization_constant > 0
    np.testing.assert_allclose(
        norm.statistic, raw.statistic / norm.normalization_constant, rtol=1e-10
    )


def test_weighted_distortion_column_present(s_map):
    index = GlobalDistortion(s_map, groups=GROUPS)
    assert "weighted_distortion" in index.data.columns
    assert np.isfinite(index.data["weighted_distortion"]).all()


def test_n_seeds_reaches_the_index(s_map):
    """n_seeds must be reachable from the class, and raise N monotonically."""
    one = GlobalDistortion(s_map, groups=GROUPS, normalize=True, n_seeds=1)
    many = GlobalDistortion(s_map, groups=GROUPS, normalize=True, n_seeds=8)
    assert many.normalization_constant >= one.normalization_constant
    assert many.statistic <= one.statistic + 1e-12


def test_normalized_global_ceiling_is_out_of_reach(segregated_strip):
    """The normalized global index must NOT reach 1.0, by design.

    N is the maximum *local* coefficient of the extreme configuration, while
    the global index is a population-weighted *mean*, so the global upper bound
    is "comparably out of reach" (de Bézenac et al. 2022, p. 10). If this test
    starts failing because the value hit 1.0, someone has renormalized the
    global index against the extreme configuration's global value -- that is a
    departure from the published definition, not a fix.
    """
    groups = ["a", "b", "c"]
    local = LocalDistortion(segregated_strip, groups=groups, normalize=True)
    glob = GlobalDistortion(segregated_strip, groups=groups, normalize=True)

    # the local index does hit exactly 1.0 on this landscape
    np.testing.assert_allclose(local.statistics.max(), 1.0, rtol=1e-10)
    # the global index provably cannot
    assert 0.0 <= glob.statistic < 1.0
    assert glob.statistic < local.statistics.max()
