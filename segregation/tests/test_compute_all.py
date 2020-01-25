import unittest
from libpysal.examples import load_example
import geopandas as gpd
import numpy as np
from segregation.compute_all import ComputeAllAspatialSegregation, ComputeAllSpatialSegregation, ComputeAllSegregation


class ComputeAll_Tester(unittest.TestCase):
    def test_ComputeAll(self):
        s_map = gpd.read_file(load_example("Sacramento1").get_path("sacramentot2.shp"))
        
        np.random.seed(123)
        res = ComputeAllAspatialSegregation(s_map, 'HISP_', 'TOT_POP')
        np.testing.assert_almost_equal(np.array(res.computed['Value']), np.array([0.32184656, 0.43506511, 0.09459761, 0.15079259, 0.76803845,
        0.23196155, 0.13768748, 0.32141961, 0.29520516, 0.09164042,
        0.31074587, 0.42179274, 0.48696508]), decimal = 3)
    
        np.random.seed(123)
        res = ComputeAllSpatialSegregation(s_map, 'HISP_', 'TOT_POP')
        np.testing.assert_almost_equal(np.array(res.computed['Value']), np.array([ 0.26119743,  0.68914224,  0.00518929,  0.85128245,  0.80449692,
       -0.11194178,  0.00909563,  0.12733821,  0.83965834,  0.15621625,
        0.22847334,  1.00266235,  0.26676264,  0.31117181,  0.17119951]), decimal = 3)
    
        np.random.seed(123)
        res = ComputeAllSegregation(s_map, 'HISP_', 'TOT_POP')
        np.testing.assert_almost_equal(np.array(res.computed['Value']), np.array([ 0.32184656,  0.43506511,  0.09459761,  0.15079259,  0.76803845,
        0.23196155,  0.13768748,  0.32136001,  0.29520516,  0.09164042,
        0.31073946,  0.42179953,  0.48696508,  0.26119743,  0.68914224,
        0.00518929,  0.85128245,  0.80449692, -0.11194178,  0.00909563,
        0.12733821,  0.83965834,  0.15621625,  0.22847334,  1.00266235,
        0.26676264,  0.31117181,  0.17119951]), decimal = 3)

if __name__ == '__main__':
    unittest.main()
