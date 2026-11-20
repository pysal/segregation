import geopandas as gpd
import numpy as np
from libpysal.examples import load_example

from segregation.local import MultiLocalDiversity


class TestMultiLocalDiversity:
    def test(self):
        s_map = gpd.read_file(load_example("Sacramento1").get_path("sacramentot2.shp"))
        groups_list = ["WHITE", "BLACK", "ASIAN", "HISP"]
        df = s_map[groups_list]
        index = MultiLocalDiversity(df, groups_list)
        np.testing.assert_almost_equal(
            index.statistics[0:10],
            np.array(
                [
                    0.34332326,
                    0.56109229,
                    0.70563225,
                    0.29713472,
                    0.22386084,
                    0.29742517,
                    0.12322789,
                    0.11274579,
                    0.09402405,
                    0.25129616,
                ]
            ),
        )
