import geopandas as gpd
import numpy as np
from libpysal.examples import load_example

from segregation.singlegroup import RelativeCentralization


class TestRelativeCentralization:
    def test(self):
        s_map = gpd.read_file(load_example("Sacramento1").get_path("sacramentot2.shp"))
        df = s_map[["geometry", "HISP", "TOT_POP"]]
        index = RelativeCentralization(df, "HISP", "TOT_POP")
        np.testing.assert_almost_equal(index.statistic, -0.11194177550430595)

    def test_haversine(self):
        s_map = gpd.read_file(load_example("Sacramento1").get_path("sacramentot2.shp"))
        df = s_map[["geometry", "HISP", "TOT_POP"]]
        # haversine on lon/lat should rank units by distance as euclidean
        # does on a projected CRS (UTM zone 10N)
        geographic = RelativeCentralization(
            df, "HISP", "TOT_POP", center=0, metric="haversine"
        )
        projected = RelativeCentralization(
            df.to_crs(32610), "HISP", "TOT_POP", center=0
        )
        np.testing.assert_almost_equal(
            geographic.statistic, projected.statistic, decimal=3
        )
