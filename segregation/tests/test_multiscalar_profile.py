import unittest
from libpysal.examples import load_example
import geopandas as gpd
import numpy as np
from segregation.singlegroup import Dissim
from segregation.dynamics import compute_multiscalar_profile


s_map = gpd.read_file(load_example("Sacramento1").get_path("sacramentot2.shp"))
df = s_map[['geometry', 'HISP', 'TOT_POP']]
df = df.to_crs(df.estimate_utm_crs())
profile = compute_multiscalar_profile(df, Dissim, group_pop_var='HISP', total_pop_var='TOT_POP', distances = [500, 1000, 1500])

np.testing.assert_array_almost_equal(profile.values, [0.32184656, 0.32192322, 0.3200861 , 0.30831014], decimal=3)
