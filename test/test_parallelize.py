import unittest

from ..parallelize import vary, mesolve
from ..Yb174.fourlevel import FourLevelSystem


class TestVary(unittest.TestCase):

    def test_no_parameter_raises_exception(self):
        with self.assertRaises(ValueError):
            _ = vary(0)

    def test_wrong_parameter_raises_exception(self):
        FLS = FourLevelSystem()
        with self.assertRaises(AttributeError):
            _ = vary(FLS, this_does_not_work=[0, 1, 3])

    def test_setting_readonly_parameter_raises_exception(self):
        FLS = FourLevelSystem()
        with self.assertRaises(AttributeError):
            _ = vary(FLS, decay=[22.3, 55.6])

    def test_creating_identical_hamiltonians_raises_exception(self):
        FLS = FourLevelSystem()
        with self.assertRaises(ValueError):
            _ = vary(FLS, B=[0.0, 0.0, 0.0, 0.0])

    def test_system_is_reset_after_variation(self):
        FLS = FourLevelSystem()
        FLS.B = 4.0
        _ = vary(FLS, B=range(10))
        self.assertEqual(FLS.B, 4.0)

    def test_parameter_order_is_preserved(self):
        FLS = FourLevelSystem()
        _, parameters = vary(FLS, B=[0, 1, 2],
                             delta=[-100.0, -50.0, 50.0, 100.0])
        self.assertEqual(parameters[0][0], 0)
        self.assertEqual(parameters[3][1], 100.0)
        self.assertEqual(parameters[8][0], 2)
        self.assertEqual(parameters[9][1], -50.0)

        _, parameters = vary(FLS, delta=[-100.0, -50.0, 50.0, 100.0],
                             B=[0, 1, 2])
        self.assertEqual(parameters[0][0], -100.0)
        self.assertEqual(parameters[3][1], 0)
        self.assertEqual(parameters[8][0], 50.0)
        self.assertEqual(parameters[8][1], 2)

    def test_hamiltonians_single_parameter(self):
        FLS = FourLevelSystem()

        hamiltonians, _ = vary(FLS, delta=[-100.0, 100.0])

        FLS = FourLevelSystem(delta=-100.0)
        self.assertEqual(FLS.H, hamiltonians[0])

        FLS = FourLevelSystem(delta=100.0)
        self.assertEqual(FLS.H, hamiltonians[1])

    def test_hamiltonians_multiple_parameters(self):
        FLS = FourLevelSystem()

        hamiltonians, parameters = vary(FLS, delta=[-30.0, -20.0, -10.0],
                                        sat=[0.0, 0.3, 0.5, 1.0],
                                        B=[-5.0, 0.0, 5.0])

        rebuilt = [FourLevelSystem(delta=delta, sat=sat, B=B).H
                   for delta, sat, B in tuple(parameters)]

        for i, hamiltonian in enumerate(rebuilt):
            self.assertEqual(hamiltonian, hamiltonians[i])


if __name__ == "__main__":
    unittest.main()