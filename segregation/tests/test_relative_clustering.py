import geopandas as gpd
import numpy as np
from libpysal.examples import load_example

from segregation.singlegroup import RelativeClustering


class TestRelativeClustering:
    def test(self):
        s_map = gpd.read_file(load_example("Sacramento1").get_path("sacramentot2.shp"))
        s_map = s_map.to_crs(s_map.estimate_utm_crs())
        df = s_map[["geometry", "HISP", "TOT_POP"]]
        index = RelativeClustering(df, "HISP", "TOT_POP")
        np.testing.assert_almost_equal(index.statistic, 0.5172753899054523)
