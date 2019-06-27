import unittest
import libpysal
import geopandas as gpd
import numpy as np
from segregation.spatial import SpatialProxProf


class Spatial_Prox_Prof_Tester(unittest.TestCase):
    def test_Spatial_Prox_Prof(self):
        s_map = gpd.read_file(libpysal.examples.get_path("sacramentot2.shp"))
        df = s_map[['geometry', 'HISP_', 'TOT_POP']]
        index = SpatialProxProf(df, 'HISP_', 'TOT_POP')
        np.testing.assert_almost_equal(index.statistic, 0.22847334404621394)


if __name__ == '__main__':
    unittest.main()
