from libpysal.examples import load_example
import geopandas as gpd
import numpy as np
from segregation.multigroup import MultiDissim
from segregation.dynamics import compute_multiscalar_profile
import quilt3
import pandana as pdna


p = quilt3.Package.browse('osm/metro_networks_8k', "s3://spatial-ucr/")
p['40900.h5'].fetch()
net = pdna.Network.from_hdf5('40900.h5')

def test_multiscalar():
    s_map = gpd.read_file(load_example("Sacramento1").get_path("sacramentot2.shp"))
    df = s_map.to_crs(s_map.estimate_utm_crs())
    profile = compute_multiscalar_profile(
        gdf=df,
        segregation_index=MultiDissim,
        distances=[500, 1000, 1500, 2000],
        groups=["HISP", "BLACK", "WHITE"],
    )
    np.testing.assert_array_almost_equal(
        profile.values, [0.4246, 0.4246, 0.4173, 0.4008, 0.3776], decimal=4
    )


def test_multiscalar_network():
    s_map = gpd.read_file(load_example("Sacramento1").get_path("sacramentot2.shp"))
    df = s_map.to_crs(s_map.estimate_utm_crs())
    profile = compute_multiscalar_profile(
        gdf=df,
        segregation_index=MultiDissim,
        distances=[500, 1000],
        groups=["HISP", "BLACK", "WHITE"],
    )
    np.testing.assert_array_almost_equal(
        profile.values, [0.4247, 0.4246, 0.4173], decimal=4
    )
