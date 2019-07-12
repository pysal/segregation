import unittest
import libpysal
import geopandas as gpd
import numpy as np
from segregation.spatial import SurfaceSpatialDissim


class SurfaceDissimTester(unittest.TestCase):
    def test_SurfaceSpatialDissim(self):
        s_map = gpd.read_file(libpysal.examples.get_path("sacramentot2.shp"))
        df = s_map[['geometry', 'HISP_', 'WHITE_']]
        index = SurfaceSpatialDissim(df, ['HISP_', 'WHITE_'])
        np.testing.assert_almost_equal(index.statistic, 0.256840165)


if __name__ == '__main__':
    unittest.main()
