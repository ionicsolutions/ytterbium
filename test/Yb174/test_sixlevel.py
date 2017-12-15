import unittest

import matplotlib.pyplot as plt
import numpy as np
import qutip
from scipy.optimize import curve_fit

from ...Yb174.sixlevel import SixLevelSystem


def exponential_decay(t, tau):
    return np.exp(-t / tau)


class TestUndrivenSystem(unittest.TestCase):

    def setUp(self):
        self.SLS = SixLevelSystem(delta=0.0, sat=0.0, polarization=(1, 0, 0),
                                  B=0)
        self.population = [state * state.dag() for state in self.SLS.basis]
        self.times = np.linspace(0, 0.40 * 10 ** -6, num=1000)

    def test_single_initial_condition(self):
        psi0 = self.SLS.basis[1]

        result = qutip.mesolve(self.SLS.H, psi0, self.times, self.SLS.decay,
                               self.population)

        self.assertEqual(result.expect[0][0], 0.0)
        self.assertEqual(result.expect[1][0], 1.0)
        self.assertEqual(result.expect[2][0], 0.0)
        self.assertEqual(result.expect[3][0], 0.0)
        self.assertEqual(result.expect[4][0], 0.0)
        self.assertEqual(result.expect[5][0], 0.0)

    def test_single_decay_time(self):
        """Linewidth from excited-state decay is within 100 Hz of the defined linewidth."""
        for i in (1, 3):
            psi0 = self.SLS.basis[i]

            result = qutip.mesolve(self.SLS.H, psi0, self.times, self.SLS.decay, self.population)

            popt, pcov = curve_fit(exponential_decay, self.times, result.expect[i])
            perr = np.sqrt(np.diag(pcov))

            if False:
                plt.plot(self.times, result.expect[i], "--")
                plt.plot(self.times, exponential_decay(self.times, *popt), "o")
                plt.show()
                plt.close()

            linewidth = 1 / (2 * np.pi * popt[0])
            self.assertTrue(abs(self.SLS.linewidth - linewidth) < 100.0)
            self.assertTrue(perr[0] < 100.0)

    def test_single_decay_ratio(self):
        """The population in the ground-states after decay from a single excited state is
        equal to the expected ratio."""
        psi0 = self.SLS.basis[1]

        result = qutip.mesolve(self.SLS.H, psi0, self.times, self.SLS.decay, self.population)

        if False:
            for i in range(6):
                plt.plot(self.times, result.expect[i], "--")
            plt.show()
            plt.close()

        self.assertEqual(round(result.expect[0][-1], 2), 1 / 2)
        self.assertEqual(round(result.expect[2][-1], 2), round(1 / 3, 2))
        self.assertEqual(round(result.expect[4][-1], 2), round(1 / 6, 2))

        psi0 = self.SLS.basis[3]

        result = qutip.mesolve(self.SLS.H, psi0, self.times, self.SLS.decay, self.population)

        if False:
            for i in range(6):
                plt.plot(self.times, result.expect[i], "--")
            plt.show()
            plt.close()

        self.assertEqual(round(result.expect[2][-1], 2), round(1 / 6, 2))
        self.assertEqual(round(result.expect[4][-1], 2), round(1 / 3, 2))
        self.assertEqual(round(result.expect[5][-1], 2), 1 / 2)


class TestDrivenSystem(unittest.TestCase):

    def setUp(self):
        self.SLS = SixLevelSystem(delta=0.0, sat=0.0, polarization=(1, 0, 0),
                                  B=0)
        self.population = [state * state.dag() for state in self.SLS.basis]
        self.times = np.linspace(0, 2.00 * 10 ** -6, num=1000)

    def test_no_laser_no_change(self):
        psi0 = self.SLS.basis[0]

        result = qutip.mesolve(self.SLS.H, psi0, self.times, self.SLS.decay, self.population)

        self.assertEqual(result.expect[0][-1], 1.0)
        self.assertEqual(result.expect[1][-1], 0.0)
        self.assertEqual(result.expect[2][-1], 0.0)
        self.assertEqual(result.expect[3][-1], 0.0)
        self.assertEqual(result.expect[4][-1], 0.0)
        self.assertEqual(result.expect[5][-1], 0.0)

    def test_wrong_polarization_no_transfer(self):
        psi0 = self.SLS.basis[5]

        self.SLS.sat = 1.0

        result = qutip.mesolve(self.SLS.H, psi0, self.times, self.SLS.raw_decay[1][0], self.population)

        if False:
            for i in range(6):
                plt.plot(self.times, result.expect[i], label="%d" % i)
            plt.legend()
            plt.show()
            plt.close()

        self.assertEqual(result.expect[5][-1], 1.0)

    def test_saturation_two_level(self):
        psi0 = self.SLS.basis[0]

        self.SLS.sat = 1/2
        self.SLS.polarization = (0, 1, 0)

        result = qutip.mesolve(self.SLS.H, psi0, self.times, self.SLS.raw_decay[1][0], self.population)

        if False:
            for i in range(6):
                plt.plot(self.times, result.expect[i], label="%d" % i)
            plt.legend()
            plt.show()
            plt.close()

        self.assertEqual(round(result.expect[0][-1] + result.expect[1][-1], 6), 1.0)
        self.assertEqual(round(result.expect[0][-1], 6), 0.75)
        self.assertEqual(round(result.expect[1][-1], 6), 0.25)


if __name__ == "__main__":
    unittest.main()
