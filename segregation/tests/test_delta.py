import geopandas as gpd
import numpy as np
from libpysal.examples import load_example

from segregation.singlegroup import Delta


class TestDelta:
    def test(self):
        s_map = gpd.read_file(load_example("Sacramento1").get_path("sacramentot2.shp"))
        df = s_map[["geometry", "HISP", "TOT_POP"]]
        index = Delta(df, "HISP", "TOT_POP")
        np.testing.assert_almost_equal(index.statistic, 0.8044969214141899)
