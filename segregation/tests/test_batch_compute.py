import geopandas as gpd
import numpy as np
from libpysal.examples import load_example
from segregation.batch import (
    batch_compute_multigroup,
    batch_compute_singlegroup,
    batch_multiscalar_singlegroup,
    batch_multiscalar_multigroup,
)

s_map = gpd.read_file(load_example("Sacramento1").get_path("sacramentot2.shp"))


def test_batch_single():
    fit = batch_compute_singlegroup(
        s_map.to_crs(s_map.estimate_utm_crs()),
        group_pop_var="HISP",
        total_pop_var="TOT_POP",
        distance=2000,
        center="mean",
        function="triangular",
    )
    np.testing.assert_array_almost_equal(
        fit.Statistic,
        [
            0.69054281,
            0.11680228,
            0.85111649,
            0.12183352,
            0.2922234,
            0.26387812,
            0.12434798,
            0.07455852,
            0.80433776,
            0.21999488,
            0.29235174,
            0.76992477,
            0.23070386,
            0.07668069,
            0.39315898,
            0.7716798,
            0.2283202,
            0.4524337,
            0.28390356,
            0.38273706,
            0.31126985,
            -0.1034416,
            0.51876119,
            0.12637769,
            0.26119743,
            0.22847334,
            1.08922873,
        ],
        decimal=3,
    )


def test_batch_multi():
    mfit = batch_compute_multigroup(
        s_map.to_crs(s_map.estimate_utm_crs()),
        distance=2000,
        groups=["HISP", "BLACK", "WHITE"],
    )
    np.testing.assert_array_almost_equal(
        mfit.Statistic,
        [
            0.37768411,
            0.11294892,
            0.78242435,
            0.50485431,
            0.14435762,
            0.15922567,
            0.13971496,
            0.11631982,
            0.55281779,
            0.44718221,
        ],
        decimal=3,
    )


def test_batch_multiscalar_multi():
    mfit = batch_multiscalar_multigroup(
        s_map.to_crs(s_map.estimate_utm_crs()),
        distances=[500, 1000],
        groups=["HISP", "BLACK", "WHITE"],
    )
    assert mfit.shape == (3, 10)


def test_batch_multiscalar_single():
    mfit = batch_multiscalar_singlegroup(
        s_map.to_crs(s_map.estimate_utm_crs()),
        distances=[500, 1000],
        group_pop_var="HISP",
        total_pop_var="TOT_POP",
    )
    assert mfit.shape == (3, 13)

