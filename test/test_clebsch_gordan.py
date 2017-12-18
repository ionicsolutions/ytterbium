import unittest

from sympy import S

from ..clebsch_gordan import factor


class TestCesiumD2(unittest.TestCase):
    """Test the generation of Clebsch-Gordans for the Cesium D2 line
    against the values given in Steck, Cesium Numbers, v1.6"""

    def factor(self, F, mF, Fprime, mFprime):
        """Custom factor generator which fixes J, Jprime, and I for Cs D2."""
        return factor(F, mF, Fprime, mFprime,
                      J=S("1/2"), Jprime=S("3/2"), I=S("7/2"))

    def test_F3_pi(self):
        expect = [
            (2, -2, S("sqrt(5/42)")),
            (2, -1, S("sqrt(4/21)")),
            (2, 0, S("sqrt(3/14)")),
            (2, 1, S("sqrt(4/21)")),
            (2, 2, S("sqrt(5/42)")),
            (3, -3, S("-sqrt(9/32)")),
            (3, -2, S("-sqrt(1/8)")),
            (3, -1, S("-sqrt(1/32)")),
            (3, 0, S("0")),
            (3, 1, S("sqrt(1/32)")),
            (3, 2, S("sqrt(1/8)")),
            (3, 3, S("sqrt(9/32)")),
            (4, -3, S("-sqrt(5/96)")),
            (4, -2, S("-sqrt(5/56)")),
            (4, -1, S("-sqrt(25/224)")),
            (4, 0, S("-sqrt(5/42)")),
            (4, 1, S("-sqrt(25/224)")),
            (4, 2, S("-sqrt(5/56)")),
            (4, 3, S("-sqrt(5/96)")),
        ]

        for Fprime, mF, value in expect:
            self.assertEqual(self.factor(3, mF, Fprime, mF), value)


class TestCesiumD1(unittest.TestCase):
    """Test the generation of Clebsch-Gordans for the Cesium D1 line
    against the values given in Steck, Cesium Numbers, v1.6"""

    def factor(self, F, mF, Fprime, mFprime):
        """Custom factor generator which fixes J, Jprime, and I for Cs D1."""
        return factor(F, mF, Fprime, mFprime,
                      J=S("1/2"), Jprime=S("1/2"), I=S("7/2"))


if __name__ == "__main__":
    unittest.main()
