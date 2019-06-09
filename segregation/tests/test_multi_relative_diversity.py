import unittest
import libpysal
import geopandas as gpd
import numpy as np
from segregation.aspatial import Multi_Relative_Diversity


class Multi_Relative_Diversity_Tester(unittest.TestCase):
    def test_Multi_Relative_Diversity(self):
        s_map = gpd.read_file(libpysal.examples.get_path("sacramentot2.shp"))
        groups_list = ['WHITE_', 'BLACK_', 'ASIAN_','HISP_']
        df = s_map[groups_list]
        index = Multi_Relative_Diversity(df, groups_list)
        np.testing.assert_almost_equal(index.statistic, 0.15820019878220337)


if __name__ == '__main__':
    unittest.main()