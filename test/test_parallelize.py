import unittest

import matplotlib.pyplot as plt
import numpy as np
import qutip

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

    def test_number_is_correct_single_parameter(self):
        FLS = FourLevelSystem()
        hamiltonians, _ = vary(FLS, B=range(10))
        self.assertEqual(len(hamiltonians), 10)

    def test_number_is_correct_multiple_parameters(self):
        FLS = FourLevelSystem()

        hamiltonians, _ = vary(FLS, B=range(5), delta=range(22))
        self.assertEqual(len(hamiltonians), 110)

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


class TestMesolve(unittest.TestCase):

    def test_requires_iterable(self):
        FLS = FourLevelSystem()

        with self.assertRaises(TypeError):
            _ = mesolve(FLS, FLS.basis[0], np.linspace(0, 0.1e-6, num=500),
                        FLS.decay, FLS.basis[0] * FLS.basis[0].dag())

    # issue #1
    # def test_results_are_ordered(self):
    #     detunings = [-100.0, -50.0, 0.0]
    #
    #     FLS = FourLevelSystem()
    #     FLS.sat = 1.0
    #     FLS.B = 0.5
    #     populations = [state * state.dag() for state in FLS.basis]
    #
    #     hamiltonians, parameters = vary(FLS, delta=detunings)
    #
    #     times = np.linspace(0.0, 0.5e-6, num=500)
    #
    #     results = mesolve(hamiltonians, FLS.basis[0], times,
    #                       FLS.decay, populations)
    #
    #     for i, multi_result in enumerate(results):
    #         FLS.delta = detunings[i]
    #         result = qutip.mesolve(FLS.H, FLS.basis[0], times,
    #                                FLS.decay, populations)
    #
    #         if False:
    #             for s, expect in enumerate(result.expect):
    #                 plt.plot(times, expect, "-", label="%d single" % s)
    #                 plt.plot(times[0::20], multi_result.expect[s][0::20],
    #                          "*", label="%d multi" % s)
    #             plt.legend()
    #             plt.show()
    #             plt.close()
    #
    #         for s, _ in enumerate(populations):
    #             diff = (result.expect[s] - multi_result.expect[s]) < 1e-6
    #             self.assertTrue(np.all(diff))


if __name__ == "__main__":
    unittest.main()
