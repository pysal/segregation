import geopandas as gpd
import numpy as np
from libpysal.examples import load_example

from segregation.multigroup import MultiRelativeDiversity


class TestMultiRelativeDiversity:
    def test(self):
        s_map = gpd.read_file(load_example("Sacramento1").get_path("sacramentot2.shp"))
        groups_list = ["WHITE", "BLACK", "ASIAN", "HISP"]
        df = s_map[groups_list]
        index = MultiRelativeDiversity(df, groups_list)
        np.testing.assert_almost_equal(index.statistic, 0.15820019878220337)
