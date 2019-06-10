import unittest
import libpysal
import geopandas as gpd
import numpy as np
from segregation.aspatial import Multi_Information_Theory


class Multi_Information_Theory_Tester(unittest.TestCase):
    def test_Multi_Information_Theory(self):
        s_map = gpd.read_file(libpysal.examples.get_path("sacramentot2.shp"))
        groups_list = ['WHITE_', 'BLACK_', 'ASIAN_','HISP_']
        df = s_map[groups_list]
        index = Multi_Information_Theory(df, groups_list)
        np.testing.assert_almost_equal(index.statistic, 0.1710160297858887)


if __name__ == '__main__':
    unittest.main()