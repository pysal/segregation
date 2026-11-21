import geopandas as gpd
import numpy as np
from libpysal.examples import load_example

from segregation.singlegroup import BoundarySpatialDissim


class TestBoundarySpatialDissim:
    def test(self):
        s_map = gpd.read_file(load_example("Sacramento1").get_path("sacramentot2.shp"))
        df = s_map[["geometry", "HISP", "TOT_POP"]]
        index = BoundarySpatialDissim(df, "HISP", "TOT_POP")
        np.testing.assert_almost_equal(index.statistic, 0.2638936888653678, decimal=2)

    def test_not_standardized(self):
        s_map = gpd.read_file(load_example("Sacramento1").get_path("sacramentot2.shp"))
        df = s_map[["geometry", "HISP", "TOT_POP"]]
        index = BoundarySpatialDissim(df, "HISP", "TOT_POP", standardize=False)
        np.testing.assert_almost_equal(index.statistic, 0.2667626367289605, decimal=4)
