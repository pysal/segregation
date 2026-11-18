import geopandas as gpd
import numpy as np
from libpysal.examples import load_example

from segregation.singlegroup import ModifiedDissim


class TestModifiedDissimTester:
    def test(self):
        s_map = gpd.read_file(load_example("Sacramento1").get_path("sacramentot2.shp"))
        df = s_map[["geometry", "HISP", "TOT_POP"]]
        index = ModifiedDissim(df, "HISP", "TOT_POP", seed=1234, backend="loky")
        np.testing.assert_almost_equal(index.statistic, 0.31075891224250635, decimal=3)
