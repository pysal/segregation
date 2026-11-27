import geopandas as gpd
import numpy as np
import pytest
from libpysal.examples import load_example

from segregation.singlegroup import AbsoluteCentralization


class TestAbsoluteCentralization:
    def test(self):
        s_map = gpd.read_file(load_example("Sacramento1").get_path("sacramentot2.shp"))
        df = s_map[["geometry", "HISP", "TOT_POP"]]
        index = AbsoluteCentralization(df, "HISP", "TOT_POP")
        np.testing.assert_almost_equal(index.statistic, 0.6891422368736286)

    def test_haversine(self):
        s_map = gpd.read_file(load_example("Sacramento1").get_path("sacramentot2.shp"))
        df = s_map[["geometry", "HISP", "TOT_POP"]]
        # haversine on lon/lat should rank units by distance as euclidean
        # does on a projected CRS (UTM zone 10N)
        geographic = AbsoluteCentralization(
            df, "HISP", "TOT_POP", center=0, metric="haversine"
        )
        projected = AbsoluteCentralization(
            df.to_crs(32610), "HISP", "TOT_POP", center=0
        )
        np.testing.assert_almost_equal(
            geographic.statistic, projected.statistic, decimal=2
        )
