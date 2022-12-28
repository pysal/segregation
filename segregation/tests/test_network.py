from libpysal.examples import load_example
import geopandas as gpd
import pandana as pdna
from segregation.network import calc_access


import quilt3 as q3
b = q3.Bucket("s3://spatial-ucr")
b.fetch("osm/metro_networks_8k/40900.h5", "./40900.h5")

def test_calc_access():
    variables = ['WHITE', 'BLACK', 'ASIAN', 'HISP']
    s_map = gpd.read_file(load_example("Sacramento1").get_path("sacramentot2.shp"))
    df = s_map[['FIPS', 'geometry'] + variables]
    df = df[df.FIPS.str.startswith('06061')]
    df = df[(df.centroid.x < -121) & (df.centroid.y < 38.85)]
    df.crs = {'init': 'epsg:4326'}
    df[variables] = df[variables].astype(float)
    test_net = pdna.Network.from_hdf5('40900.h5')

    acc = calc_access(df, test_net, distance=1., variables=variables)
    assert acc.WHITE.sum() > 100
