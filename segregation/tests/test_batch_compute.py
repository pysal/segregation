from tkinter import N
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
    np.random.seed(1234)
    fit = batch_compute_singlegroup(
        s_map.to_crs(s_map.estimate_utm_crs()),
        group_pop_var="HISP",
        total_pop_var="TOT_POP",
        distance=2000,
        center="mean",
        function="triangular",
        seed=1234,
        backend='loky'
        # loky is slower but more robust in testing

    )
    np.testing.assert_array_almost_equal(
        fit.Statistic,
        [
            0.6905,
            0.061,
            0.8511,
            0.1218,
            0.2922,
            0.2639,
            0.1243,
            0.0746,
            0.8043,
            0.22,
            0.2924,
            0.86,
            0.2,
            0.0767,
            0.3932,
            0.7717,
            0.2283,
            0.4524,
            0.2839,
            0.3827,
            0.3113,
            -0.1034,
            0.52,
            0.1264,
            0.2612,
            0.2285,
            1.0468,
        ],
        decimal=2,
    )


def test_batch_multi():
    np.random.seed(1234)
    mfit = batch_compute_multigroup(
        s_map.to_crs(s_map.estimate_utm_crs()),
        distance=2000,
        groups=["HISP", "BLACK", "WHITE"],
    )
    np.testing.assert_array_almost_equal(
        mfit.Statistic,
        [   11.78 ,
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
    np.random.seed(1234)

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
        seed=1234,
        backend='loky'
        # loky is slower but more robust in testing
    )
    assert mfit.shape == (3, 13)

