import unittest
from libpysal.examples import load_example
import geopandas as gpd
import numpy as np
from segregation.spatial import BoundarySpatialDissim


class Boundary_Spatial_Dissim_Tester(unittest.TestCase):
    def test_Boundary_Spatial_Dissim(self):
        s_map = gpd.read_file(load_example("Sacramento1").get_path("sacramentot2.shp"))
        df = s_map[['geometry', 'HISP_', 'TOT_POP']]
        index = BoundarySpatialDissim(df, 'HISP_', 'TOT_POP')
        np.testing.assert_almost_equal(index.statistic, 0.2667626367289605)


if __name__ == '__main__':
    unittest.main()
