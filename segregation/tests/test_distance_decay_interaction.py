import unittest
from libpysal.examples import load_example
import geopandas as gpd
import numpy as np
from segregation.singlegroup import DistanceDecayInteraction


class Distance_Decay_Interaction_Tester(unittest.TestCase):
    def test_Distance_Decay_Interaction(self):
        s_map = gpd.read_file(load_example("Sacramento1").get_path("sacramentot2.shp"))
        s_map = s_map.to_crs(s_map.estimate_utm_crs())
        df = s_map[['geometry', 'HISP', 'TOT_POP']]
        index = DistanceDecayInteraction(df, 'HISP', 'TOT_POP')
        np.testing.assert_almost_equal(index.statistic, 0.8620769567792631)


if __name__ == '__main__':
    unittest.main()
