import unittest
import libpysal
import libpysal.api as ps
import mapclassify.api as mc
import giddy.api as gapi
import numpy as np

class Gini_Tester(unittest.TestCase):

    def setUp(self):
        f = libpysal.open(libpysal.examples.get_path("mexico.csv"))
        vnames = ["pcgdp%d" % dec for dec in range(1940, 2010, 10)]
        y = np.transpose(np.array([f.by_col[v] for v in vnames]))
        self.y = y[:, 0]
        regimes = np.array(f.by_col('hanson98'))
        self.w = ps.block_weights(regimes)

    def test_Gini(self):
        g = gapi.Gini(self.y)
        np.testing.assert_almost_equal(g.g, 0.35372371173452849)

    def test_Gini_Spatial(self):
        np.random.seed(12345)
        g = gapi.Gini_Spatial(self.y, self.w)
        np.testing.assert_almost_equal(g.g, 0.35372371173452849)
        np.testing.assert_almost_equal(g.wg, 884130.0)
        np.testing.assert_almost_equal(g.wcg, 4353856.0)
        np.testing.assert_almost_equal(g.p_sim, 0.040)
        np.testing.assert_almost_equal(g.e_wcg, 4170356.7474747472)

class test_Theil(unittest.TestCase):
    def test___init__(self):
        # theil = Theil(y)
        f = libpysal.open(libpysal.examples.get_path("mexico.csv"))
        vnames = ["pcgdp%d" % dec for dec in range(1940, 2010, 10)]
        y = np.transpose(np.array([f.by_col[v] for v in vnames]))
        theil_y = gapi.Theil(y)
        np.testing.assert_almost_equal(theil_y.T, np.array([0.20894344, 0.15222451, 0.10472941, 0.10194725, 0.09560113, 0.10511256, 0.10660832]))


class test_TheilD(unittest.TestCase):
    def test___init__(self):
        # theil_d = TheilD(y, partition)
        f = libpysal.open(libpysal.examples.get_path("mexico.csv"))
        vnames = ["pcgdp%d" % dec for dec in range(1940, 2010, 10)]
        y = np.transpose(np.array([f.by_col[v] for v in vnames]))
        regimes = np.array(f.by_col('hanson98'))
        theil_d = gapi.TheilD(y, regimes)
        np.testing.assert_almost_equal(theil_d.bg, np.array([0.0345889, 0.02816853, 0.05260921, 0.05931219, 0.03205257, 0.02963731, 0.03635872]))


class test_TheilDSim(unittest.TestCase):
    def test___init__(self):
        f = libpysal.open(libpysal.examples.get_path("mexico.csv"))
        vnames = ["pcgdp%d" % dec for dec in range(1940, 2010, 10)]
        y = np.transpose(np.array([f.by_col[v] for v in vnames]))
        regimes = np.array(f.by_col('hanson98'))
        np.random.seed(10)
        theil_ds = gapi.TheilDSim(y, regimes, 999)
        np.testing.assert_almost_equal(theil_ds.bg_pvalue, np.array(
            [0.4, 0.344, 0.001, 0.001, 0.034, 0.072, 0.032]))


suite = unittest.TestSuite()
test_classes = [Gini_Tester, test_Theil, test_TheilD, test_TheilDSim]
for i in test_classes:
    a = unittest.TestLoader().loadTestsFromTestCase(i)
    suite.addTest(a)



if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)
