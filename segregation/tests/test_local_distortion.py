import geopandas as gpd
import numpy as np
import pytest
from libpysal.examples import load_example
from segregation.local import LocalDistortion


def test_LocalDiversity():
    s_map = gpd.read_file(load_example("Sacramento1").get_path("sacramentot2.shp"))
    s_map = s_map.to_crs(s_map.estimate_utm_crs())
    groups_list = ["WHITE", "BLACK", "ASIAN", "HISP"]
    index = LocalDistortion(s_map, groups=groups_list, normalize=False)
    np.testing.assert_almost_equal(
        index.statistics.values,
        np.array(
            [
                25.39661811,
                24.30782048,
                24.60991437,
                24.36648691,
                27.83680606,
                24.65638555,
                24.88922494,
                24.92836492,
                25.14415759,
                24.01771598,
                23.97749865,
                23.42248598,
                14.77405384,
                22.51638462,
                18.75242164,
                26.92174443,
                21.33143264,
                25.94099694,
                7.51574987,
                23.87897129,
                24.11579856,
                24.24919432,
                19.94738295,
                23.67435917,
                23.66220075,
                23.88176657,
                21.9073434,
                21.69630955,
                21.53651026,
                23.83644812,
                13.59853066,
                10.14442666,
                19.66398573,
                23.65314736,
                21.31471342,
                13.63432815,
                12.11440418,
                21.08898192,
                16.66283344,
                13.02324441,
                14.30926067,
                17.02810183,
                16.7338965,
                25.19860884,
                22.95484669,
                12.61147126,
                14.204861,
                13.19747972,
                10.82991424,
                14.31906992,
                27.59371577,
                26.46648197,
                11.69197751,
                23.95014558,
                13.64158922,
                9.62490954,
                25.00844825,
                11.04728627,
                19.85083909,
                9.38059428,
                16.96196282,
                13.00744593,
                27.0202913,
                14.74658408,
                8.84934127,
                27.27518906,
                7.27616926,
                10.22894946,
                7.25080181,
                12.78901301,
                12.90493072,
                11.58245316,
                13.71169195,
                6.29432362,
                2.35877359,
                24.24015016,
                9.5871897,
                5.58603887,
                13.29166783,
                11.50790836,
                2.677827,
                3.49343887,
                4.24967279,
                5.33635179,
                25.68103948,
                8.28016966,
                6.8711452,
                8.96364021,
                11.71234874,
                25.01418507,
                3.33323728,
                12.43716423,
                23.51145612,
                13.89380442,
                12.63169812,
                11.86449082,
                17.02485166,
                20.46177457,
                2.32853799,
                12.81339418,
                15.05348515,
                10.26230791,
                9.98804818,
                8.88361087,
                7.50873816,
                6.77690426,
                5.55719876,
                2.74324202,
                4.13581674,
                13.31433335,
                10.09255934,
                4.70828825,
                11.83332145,
                10.73623195,
                4.15826593,
                12.03225237,
                19.07966512,
                18.10391352,
                3.12447887,
                2.40766623,
                22.598487,
                19.799451,
                23.49219834,
                12.649395,
                3.68895467,
                9.08607696,
                2.80264455,
                6.10869683,
                12.29965067,
                12.73940509,
                11.29353983,
                13.47973953,
                9.7306413,
                3.82411005,
                8.1931548,
                2.49678327,
                4.27708591,
                2.86215486,
                13.48956176,
                11.70693319,
                12.86094904,
                4.40769759,
                20.31308085,
                2.63372518,
                11.38235628,
                7.05885611,
                11.06573066,
                11.70479192,
                10.43061806,
                11.87005861,
                9.8528621,
                9.40490022,
                8.74282753,
                3.49880998,
                12.98351356,
                13.38584099,
                13.58820213,
                12.79255652,
                12.48129634,
                11.71747096,
                5.63045744,
                12.98984268,
                5.0224995,
                12.12849733,
                10.96744044,
                3.21536378,
                7.08427054,
                10.89236298,
                10.68537811,
                10.62380306,
                9.47601061,
                8.77868188,
                7.23287923,
                13.27643096,
                13.23185944,
                1.99222569,
                3.89049264,
                5.56832484,
                15.5297813,
                11.37645644,
                6.89491513,
                8.84812347,
                12.52195449,
                3.33162522,
                13.37084358,
                4.18451751,
                9.77115111,
                8.31399318,
                15.16555377,
                8.14063645,
                6.91296,
                4.05280486,
                6.26233005,
                10.74506609,
                2.62821931,
                11.61994107,
                5.50090748,
                8.87544413,
                2.07310641,
                6.7632975,
                6.747503,
                7.66973533,
                5.11202183,
                5.1259091,
                5.21945887,
                2.6278925,
                13.3404676,
                2.95294966,
                12.33984744,
                4.24737386,
                12.59369448,
                12.85608803,
                5.13530389,
                10.88078102,
                13.07017361,
                5.02606355,
                12.97705611,
                3.12574748,
                13.10159546,
                5.22364692,
                5.49566307,
                2.35362922,
                5.00599119,
                4.09054877,
                9.89791706,
                3.33047614,
                2.8955676,
                3.56367414,
                4.30090555,
                4.11760182,
                4.05941767,
                3.98568945,
                13.80440881,
                3.35381429,
                6.36559194,
                12.43385529,
                4.38590497,
                3.62604335,
                3.44507718,
                4.40333109,
                4.14084529,
                3.2518651,
                2.97698131,
                14.34919391,
                12.83602074,
                5.02181116,
                12.10793633,
                15.45994447,
                16.07926916,
                8.51065472,
                14.04552929,
                2.95647641,
                15.39259637,
                10.99062551,
                3.20224053,
                10.72758372,
                13.35374073,
                4.71684428,
                12.57768599,
                10.13482377,
                5.88446476,
                14.4413251,
                3.19465648,
                5.79614113,
                18.23180005,
                11.98720248,
                14.35411293,
                16.04847729,
                13.36409764,
                12.14261212,
                2.62615234,
                2.99467377,
                8.17852905,
                4.28099541,
                12.81908459,
                2.92684096,
                11.48318198,
                14.88608949,
                14.38246855,
                9.67731878,
                17.01889773,
                14.13369012,
                6.95508024,
                3.88635619,
                14.1852798,
                5.12194273,
                16.30902956,
                3.47213125,
                14.30721204,
                13.88102294,
                23.17314272,
                15.19845156,
                15.15350962,
                15.16833744,
                14.91015326,
                14.72176297,
                5.79854202,
                14.65392677,
                16.18949326,
                10.17517515,
                19.31783481,
                19.17129223,
                5.04695617,
                14.87903255,
                4.96065978,
                18.95232945,
                16.15851494,
                14.82842524,
                5.71152858,
                6.83259955,
                21.9225959,
                15.76250262,
                16.65568344,
                22.95994267,
                26.95103295,
                25.5026361,
                19.0855866,
                23.27474007,
                27.27569254,
                25.83641696,
                24.23275891,
                9.77593791,
                21.26728676,
                23.68622366,
                27.04252573,
                21.844736,
                26.93346352,
                28.82465976,
                31.25100858,
                32.80067067,
                36.24105943,
                28.93541014,
                31.50091364,
                28.76452339,
                36.23281269,
                33.31318445,
                28.1362903,
                28.06935251,
                31.70611368,
                32.21571938,
                31.44066684,
                30.54718229,
                34.20847625,
                23.75852978,
                37.74246523,
                34.85263569,
                32.60861259,
                40.58759056,
                28.75820084,
                40.66633579,
                39.17743613,
                36.61858754,
                13.30881027,
                33.2165451,
                33.99236867,
                32.95550584,
                36.67039103,
                39.87643588,
                23.46040494,
                18.24277744,
                39.62681687,
                28.94000499,
                39.32200684,
                22.33577412,
                24.02489418,
                28.6814168,
                28.54757338,
                32.70266705,
                35.83246679,
                21.5959394,
                6.52414532,
                27.4140926,
                19.34536888,
                16.89923494,
                32.67249503,
                27.4525894,
                22.44009398,
                18.47682668,
                29.37146556,
                26.62930762,
                22.34878022,
                20.00850054,
                25.07393409,
                28.76867152,
                32.34310948,
                20.46021478,
                18.43423649,
                19.38940189,
                21.47902291,
                22.67837779,
                23.8358755,
                14.20077882,
                17.92407473,
                18.87479737,
                17.65660508,
                8.36480324,
                17.23432627,
                19.687141,
                14.8972051,
                15.99540884,
                16.39727766,
                15.83193257,
                18.93337774,
            ]
        ),
    )


@pytest.fixture
def s_map():
    m = gpd.read_file(load_example("Sacramento1").get_path("sacramentot2.shp"))
    return m.to_crs(m.estimate_utm_crs())


def test_normalization_no_raise(s_map):
    """normalize=True should not raise."""
    groups_list = ["WHITE", "BLACK", "ASIAN", "HISP"]
    index = LocalDistortion(s_map, groups=groups_list, normalize=True)
    assert index.statistics is not None


def test_normalization_values_in_unit_interval(s_map):
    """Normalized distortion coefficients should lie in [0, 1]."""
    groups_list = ["WHITE", "BLACK", "ASIAN", "HISP"]
    index = LocalDistortion(s_map, groups=groups_list, normalize=True)
    assert index.statistics.min() >= 0.0
    assert index.statistics.max() <= 1.0 + 1e-10  # allow tiny float error


def test_normalization_constant_exposed(s_map):
    """The normalization constant should be a positive finite float."""
    groups_list = ["WHITE", "BLACK", "ASIAN", "HISP"]
    index = LocalDistortion(s_map, groups=groups_list, normalize=True)
    N = index.normalization_constant
    assert N is not None
    assert np.isfinite(N)
    assert N > 0


def test_normalization_constant_none_when_not_normalized(s_map):
    """normalization_constant should be None when normalize=False."""
    groups_list = ["WHITE", "BLACK", "ASIAN", "HISP"]
    index = LocalDistortion(s_map, groups=groups_list, normalize=False)
    assert index.normalization_constant is None


def test_normalization_two_groups(s_map):
    """Normalization should work with only 2 groups."""
    groups_list = ["WHITE", "BLACK"]
    index = LocalDistortion(s_map, groups=groups_list, normalize=True)
    assert index.statistics.min() >= 0.0
    assert index.statistics.max() <= 1.0 + 1e-10


@pytest.fixture
def segregated_strip():
    """A hand-built maximally-segregated landscape.

    Twelve equally-populated units in a line, with three groups occupying
    contiguous blocks ordered smallest-to-largest from one end.  This is
    exactly the configuration the normalization constant is defined against,
    so its largest normalized coefficient must be 1.0.
    """
    from shapely.geometry import Point

    n, per_unit = 12, 10.0
    pops = {"a": np.zeros(n), "b": np.zeros(n), "c": np.zeros(n)}
    pops["a"][0:2] = per_unit  # total  20
    pops["b"][2:6] = per_unit  # total  40
    pops["c"][6:12] = per_unit  # total  60
    return gpd.GeoDataFrame(
        pops, geometry=[Point(i, 0) for i in range(n)], crs=3857
    )


def test_normalization_max_is_one(segregated_strip):
    """A maximally-segregated landscape should normalize to exactly 1.0."""
    index = LocalDistortion(segregated_strip, groups=["a", "b", "c"], normalize=True)
    np.testing.assert_allclose(index.statistics.max(), 1.0, rtol=1e-10)


def test_normalization_constant_matches_own_landscape(segregated_strip):
    """N must equal the raw max distortion of the landscape it is built from."""
    from segregation.util.normalization import _maximal_segregation_distortion

    groups_list = ["a", "b", "c"]
    N = _maximal_segregation_distortion(segregated_strip, groups_list)
    raw = LocalDistortion(segregated_strip, groups=groups_list, normalize=False).statistics
    np.testing.assert_allclose(raw.max(), N, rtol=1e-10)


def test_normalization_is_deterministic(s_map):
    """N is a function of the data alone — no randomness, no hidden state."""
    from segregation.util.normalization import _maximal_segregation_distortion

    groups_list = ["WHITE", "BLACK", "ASIAN", "HISP"]
    first = _maximal_segregation_distortion(s_map, groups_list)
    second = _maximal_segregation_distortion(s_map, groups_list)
    assert first == second


def test_more_seeds_never_lowers_n(s_map):
    """N is a maximum over seeds, so adding seeds can only raise it."""
    from segregation.util.normalization import _maximal_segregation_distortion

    groups_list = ["WHITE", "BLACK", "ASIAN", "HISP"]
    Ns = [
        _maximal_segregation_distortion(s_map, groups_list, n_seeds=k)
        for k in (1, 2, 4, 8)
    ]
    assert np.all(np.diff(Ns) >= -1e-9), f"N decreased with more seeds: {Ns}"


def test_n_seeds_reaches_the_index(s_map):
    """n_seeds must be reachable from the public class, not hardcoded inside."""
    groups_list = ["WHITE", "BLACK", "ASIAN", "HISP"]
    one = LocalDistortion(s_map, groups=groups_list, normalize=True, n_seeds=1)
    many = LocalDistortion(s_map, groups=groups_list, normalize=True, n_seeds=8)
    assert many.normalization_constant >= one.normalization_constant
    # a larger constant means uniformly smaller normalized coefficients
    assert many.statistics.max() <= one.statistics.max() + 1e-12


def test_ordering_keeps_positional_argument_order(s_map):
    """Positional args stay (gdf, metric, network, distance_matrix).

    `seed` is keyword-only precisely so that adding it could not silently
    displace `metric` for any caller passing positionally.
    """
    from segregation.util.normalization import _ordering_for_normalization

    positional = _ordering_for_normalization(s_map, "euclidean", None, None)
    keyword = _ordering_for_normalization(s_map, metric="euclidean")
    np.testing.assert_array_equal(positional, keyword)


def test_ordering_rejects_non_integer_seed(s_map):
    """A bad seed must say so, not raise a cryptic numpy IndexError."""
    from segregation.util.normalization import _ordering_for_normalization

    with pytest.raises(TypeError, match="seed must be an integer"):
        _ordering_for_normalization(s_map, seed="euclidean")
    with pytest.raises(IndexError, match="out of range"):
        _ordering_for_normalization(s_map, seed=10**6)


def test_candidate_seeds_handles_collinear_geometry(segregated_strip):
    """Collinear centroids have no convex hull; fall back rather than raise."""
    from segregation.util.normalization import _candidate_seeds

    coords = np.column_stack(
        (segregated_strip.geometry.x, segregated_strip.geometry.y)
    )
    seeds = _candidate_seeds(coords, n_seeds=4)
    assert len(seeds) >= 1
    assert np.all((seeds >= 0) & (seeds < len(coords)))


def test_normalization_requires_units_for_every_group():
    """Fewer units than groups must raise, not silently drop a group."""
    from shapely.geometry import Point
    from segregation.util.normalization import _maximal_segregation_distortion

    gdf = gpd.GeoDataFrame(
        {"a": [10.0, 0, 0], "b": [0.0, 10, 0], "c": [0.0, 0, 10], "d": [1.0, 1, 1]},
        geometry=[Point(0, 0), Point(1, 0), Point(2, 0)],
        crs=3857,
    )
    with pytest.raises(ValueError, match="at least as many observations as groups"):
        _maximal_segregation_distortion(gdf, ["a", "b", "c", "d"])


def test_block_boundaries_never_empty():
    """Every group gets at least one unit, so no population is ever dropped."""
    from segregation.util.normalization import _block_boundaries

    cases = [
        (np.full(12, 10.0), np.array([20.0, 40.0, 60.0])),  # exact cuts
        (np.full(4, 1.0), np.array([1.0, 1.0, 1.0, 1.0])),  # n == k
        (np.full(5, 1.0), np.array([0.0, 0.001, 4.999])),  # near-empty groups
        (np.full(5, 1.0), np.array([4.999, 0.001, 0.0])),  # ... at the far end
        (np.arange(1.0, 11.0), np.array([1.0, 54.0])),  # lopsided
    ]
    for unit_pop, group_totals in cases:
        edges = _block_boundaries(unit_pop, group_totals)
        assert edges[-1] == unit_pop.size
        assert np.all(np.diff(edges) >= 1), f"empty block for {group_totals}"
        assert edges[0] >= 1


def test_ordering_covers_disconnected_geography():
    """A disconnected study region must still order every unit exactly once."""
    from shapely.geometry import Point
    from segregation.util.normalization import _ordering_for_normalization

    pts = [Point(x, y) for x in range(4) for y in range(4)]
    pts += [Point(1e6 + x, 1e6 + y) for x in range(3) for y in range(3)]
    rng = np.random.default_rng(0)
    gdf = gpd.GeoDataFrame(
        {
            "a": rng.integers(1, 50, len(pts)).astype(float),
            "b": rng.integers(1, 50, len(pts)).astype(float),
        },
        geometry=pts,
        crs=3857,
    )
    order = _ordering_for_normalization(gdf)
    np.testing.assert_array_equal(np.sort(order), np.arange(len(pts)))

    index = LocalDistortion(gdf, groups=["a", "b"], normalize=True)
    assert index.statistics.min() >= 0.0
    assert index.statistics.max() <= 1.0 + 1e-10


def test_sparse_distance_matrix_raises(segregated_strip):
    """Sparse distance matrices are unsupported and must say so clearly."""
    from scipy.sparse import csr_array
    from scipy.spatial.distance import pdist, squareform
    from segregation.util.normalization import _maximal_segregation_distortion

    coords = np.column_stack(
        (segregated_strip.geometry.x, segregated_strip.geometry.y)
    )
    dm = csr_array(squareform(pdist(coords)))
    with pytest.raises(TypeError, match="sparse distance matrices"):
        _maximal_segregation_distortion(
            segregated_strip, ["a", "b", "c"], metric="precomputed", distance_matrix=dm
        )


def test_normalization_with_precomputed_matrix(s_map):
    """A precomputed euclidean matrix should reproduce the euclidean result."""
    from scipy.spatial.distance import pdist, squareform
    from segregation.util.normalization import _maximal_segregation_distortion

    groups_list = ["WHITE", "BLACK", "ASIAN", "HISP"]
    centroids = s_map.geometry.centroid
    dm = squareform(pdist(np.column_stack((centroids.x, centroids.y))))
    N_matrix = _maximal_segregation_distortion(
        s_map, groups_list, metric="precomputed", distance_matrix=dm
    )
    N_euclidean = _maximal_segregation_distortion(s_map, groups_list)
    np.testing.assert_allclose(N_matrix, N_euclidean, rtol=1e-10)


def test_ckdtree_regression(s_map):
    """cKDTree path should produce identical results to the old dense-matrix path."""
    groups_list = ["WHITE", "BLACK", "ASIAN", "HISP"]
    index = LocalDistortion(s_map, groups=groups_list, normalize=False)
    # The hardcoded expected values from the original test
    expected = np.array(
        [
            25.39661811,
            24.30782048,
            24.60991437,
            24.36648691,
            27.83680606,
            24.65638555,
            24.88922494,
            24.92836492,
            25.14415759,
            24.01771598,
        ]
    )
    np.testing.assert_almost_equal(index.statistics.values[:10], expected, decimal=6)


def test_divergence_profile_rejects_sparse_matrix(segregated_strip):
    """The rejection belongs upstream, so every caller gets a clear error."""
    from scipy.sparse import csr_array, csr_matrix
    from scipy.spatial.distance import pdist, squareform
    from segregation.dynamics.divergence_profile import compute_divergence_profiles

    coords = np.column_stack(
        (segregated_strip.geometry.x, segregated_strip.geometry.y)
    )
    dense = squareform(pdist(coords))
    for sparse in (csr_matrix(dense), csr_array(dense)):
        with pytest.raises(TypeError, match="sparse distance matrices"):
            compute_divergence_profiles(
                gdf=segregated_strip,
                groups=["a", "b", "c"],
                metric="precomputed",
                distance_matrix=sparse,
            )
