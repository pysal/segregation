import unittest
from libpysal.examples import load_example
import geopandas as gpd
import numpy as np
from segregation.singlegroup import MinMax


class SpatialMinMax_Tester(unittest.TestCase):
    def test_SpatialMinMax(self):
        s_map = gpd.read_file(load_example("Sacramento1").get_path("sacramentot2.shp"))
        df = s_map[['geometry', 'HISP', 'TOT_POP']]
        df = df.to_crs(df.estimate_utm_crs())
        index = MinMax(df, 'HISP', 'TOT_POP', distance=2000, function='triangular')
        np.testing.assert_almost_equal(index.statistic, 0.4524336967483127, decimal=4)


if __name__ == '__main__':
    unittest.main()
