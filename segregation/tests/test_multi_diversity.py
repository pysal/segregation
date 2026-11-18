import geopandas as gpd
import numpy as np
from libpysal.examples import load_example

from segregation.multigroup import MultiDiversity


class TestMultiDiversity:
    def test(self):
        s_map = gpd.read_file(load_example("Sacramento1").get_path("sacramentot2.shp"))
        groups_list = ["WHITE", "BLACK", "ASIAN", "HISP"]
        df = s_map[groups_list]
        index = MultiDiversity(df, groups_list)
        np.testing.assert_almost_equal(index.statistic, 0.9733112243997906)

        index_norm = MultiDiversity(df, groups_list, normalized=True)
        np.testing.assert_almost_equal(index_norm.statistic, 0.7020956383415715)
