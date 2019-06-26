import unittest
import libpysal
import geopandas as gpd
import numpy as np
from segregation.local import Multi_Local_Diversity


class Multi_Local_Diversity_Tester(unittest.TestCase):
    def test_Multi_Local_Diversity(self):
        s_map = gpd.read_file(libpysal.examples.get_path("sacramentot2.shp"))
        groups_list = ['WHITE_', 'BLACK_', 'ASIAN_','HISP_']
        df = s_map[groups_list]
        index = Multi_Local_Diversity(df, groups_list)
        np.testing.assert_almost_equal(index.statistics[0:10], np.array([0.34332326, 0.56109229, 0.70563225, 0.29713472, 0.22386084,
																		 0.29742517, 0.12322789, 0.11274579, 0.09402405, 0.25129616]))


if __name__ == '__main__':
    unittest.main()
