import unittest
import pandas as pd
import geopandas as gpd
import numpy as np
from libpysal.examples import load_example
from segregation.inference import SingleValueTest, TwoValueTest
from segregation.multigroup import MultiDissim
from segregation.singlegroup import Dissim


class Inference_Tester(unittest.TestCase):
    def test_Inference(self):
        s_map = gpd.read_file(load_example("Sacramento1").get_path("sacramentot2.shp"))
        # note need to recast as datafrme
        s_map_no_geom = pd.DataFrame(
            gpd.read_file(
                load_example("Sacramento1").get_path("sacramentot2.shp")
            ).drop(columns=["geometry"])
        )
        index1 = Dissim(s_map, "HISP", "TOT_POP")
        index2 = Dissim(s_map, "BLACK", "TOT_POP")

        groups_list = ["WHITE", "BLACK", "ASIAN", "HISP"]
        m_index = MultiDissim(s_map, groups_list)

        m_index_1 = MultiDissim(s_map[0:200], groups_list)
        m_index_2 = MultiDissim(s_map[200:], groups_list)

        # Single Value Tests #
        np.random.seed(123)
        res = SingleValueTest(
            index1,
            null_approach="systematic",
            iterations_under_null=50,
            backend="multiprocessing",
        )
        np.testing.assert_almost_equal(res.est_sim.mean(), 0.017621, decimal=2)

        index1_no_geom = Dissim(s_map_no_geom, "HISP", "TOT_POP")

        np.random.seed(123)
        res = SingleValueTest(
            index1_no_geom,
            null_approach="systematic",
            iterations_under_null=50,
            backend="multiprocessing",
        )
        np.testing.assert_almost_equal(res.est_sim.mean(), 0.017621, decimal=2)

        np.random.seed(123)
        res = SingleValueTest(
            index1, null_approach="bootstrap", iterations_under_null=50
        )
        np.testing.assert_almost_equal(res.est_sim.mean().round(2), 0.32, decimal=2)

        np.random.seed(123)
        res = SingleValueTest(
            index1, null_approach="evenness", iterations_under_null=50
        )
        np.testing.assert_almost_equal(
            res.est_sim.mean(), 0.01596295861644252, decimal=2
        )

        np.random.seed(123)
        res = SingleValueTest(
            index1, null_approach="geographic_permutation", iterations_under_null=50
        )
        np.testing.assert_almost_equal(
            res.est_sim.mean(), 0.32184656076566864, decimal=2
        )

        np.random.seed(123)
        res = SingleValueTest(
            index1, null_approach="systematic_permutation", iterations_under_null=50
        )
        np.testing.assert_almost_equal(res.est_sim.mean().round(2), 0.016, decimal=2)

        np.random.seed(123)
        res = SingleValueTest(
            index1, null_approach="even_permutation", iterations_under_null=50
        )
        np.testing.assert_almost_equal(
            res.est_sim.mean(), 0.01619436868061094, decimal=2
        )

        np.random.seed(123)
        res = SingleValueTest(
            m_index, null_approach="bootstrap", iterations_under_null=50
        )
        np.testing.assert_almost_equal(res.est_sim.mean().round(2), 0.41, decimal=2)

        np.random.seed(123)
        res = SingleValueTest(
            m_index, null_approach="evenness", iterations_under_null=50
        )
        np.testing.assert_almost_equal(
            res.est_sim.mean(), 0.01633979237418177, decimal=2
        )

        np.random.seed(123)
        res = SingleValueTest(
            m_index, null_approach="person_permutation", iterations_under_null=50
        )
        np.testing.assert_almost_equal(
            res.est_sim.mean(), 0.01633979237418177, decimal=2
        )

        # Two Value Tests #
        np.random.seed(123)
        res = TwoValueTest(
            index1,
            index2,
            null_approach="random_label",
            iterations_under_null=50,
            backend="multiprocessing",
        )
        np.testing.assert_almost_equal(
            res.est_sim.mean(), -0.0031386146371949076, decimal=2
        )

        np.random.seed(123)
        res = TwoValueTest(
            index1,
            index2,
            null_approach="composition",
            iterations_under_null=50,
        )
        np.testing.assert_almost_equal(
            res.est_sim.mean(), -0.005032145622504718, decimal=2
        )

        np.random.seed(123)
        res = TwoValueTest(
            index1,
            index2,
            null_approach="share",
            iterations_under_null=50,
        )
        np.testing.assert_almost_equal(
            res.est_sim.mean(), -0.034350440515125, decimal=2
        )

        np.random.seed(123)
        res = TwoValueTest(
            index1,
            index2,
            null_approach="dual_composition",
            iterations_under_null=50,
        )
        np.testing.assert_almost_equal(
            res.est_sim.mean(), -0.004771386292706747, decimal=2
        )

        np.random.seed(123)
        res = TwoValueTest(
            m_index_1, m_index_2, null_approach="random_label", iterations_under_null=50
        )
        np.testing.assert_almost_equal(
            res.est_sim.mean(), -0.0024327144012562685, decimal=2
        )

        np.random.seed(123)
        res = TwoValueTest(
            m_index_1, m_index_2, null_approach="bootstrap", iterations_under_null=50
        )
        np.testing.assert_almost_equal(res.est_sim[0].mean(), 0.38738, decimal=2)

        np.random.seed(123)
        res = TwoValueTest(
            m_index_1,
            m_index_2,
            null_approach="person_permutation",
            iterations_under_null=50,
        )
        np.testing.assert_almost_equal(res.est_sim.mean(), 0.0, decimal=2)


if __name__ == "__main__":
    unittest.main()
