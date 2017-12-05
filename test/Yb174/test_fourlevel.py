import unittest
from scipy.optimize import curve_fit
import qutip
import numpy as np
import matplotlib.pyplot as plt

from Yb174.fourlevel import FourLevelSystem


def exponential_decay(t, tau):
    return np.exp(-t/tau)


class TestUndrivenSystem(unittest.TestCase):

    def setUp(self):
        self.FLS = FourLevelSystem(delta=0.0, sat=0.0, polarization=(1, 0, 0),
                                   B=0)
        self.population = [state * state.dag() for state in self.FLS.basis]
        self.times = np.linspace(0, 0.15*10**-6, num=1000)

    def test_single_initial_condition(self):
        psi0 = self.FLS.basis[2]

        result = qutip.mesolve(self.FLS.H, psi0, self.times, self.FLS.decay,
                               self.population)

        self.assertEqual(result.expect[0][0], 0.0)
        self.assertEqual(result.expect[1][0], 0.0)
        self.assertEqual(result.expect[2][0], 1.0)
        self.assertEqual(result.expect[3][0], 0.0)

    def test_single_decay_time(self):
        """Linewidth from excited-state decay is within 100 Hz of the defined linewidth."""
        for i in range(2):
            psi0 = self.FLS.basis[2 + i]

            result = qutip.mesolve(self.FLS.H, psi0, self.times, self.FLS.decay, self.population)

            popt, pcov = curve_fit(exponential_decay, self.times, result.expect[2 + i])
            perr = np.sqrt(np.diag(pcov))

            if False:
                plt.plot(self.times, result.expect[2], "--")
                plt.plot(self.times, exponential_decay(self.times, *popt), "o")
                plt.show()
                plt.close()

            linewidth = 2*np.pi/popt[0]
            self.assertTrue(self.FLS.linewidth - linewidth < 100.0)
            self.assertTrue(perr[0] < 100.0)

    def test_single_decay_ratio(self):
        """The population in the ground-states after decay from a single excited state is
        equal to the expected ratio."""
        psi0 = self.FLS.basis[2]

        result = qutip.mesolve(self.FLS.H, psi0, self.times, self.FLS.decay, self.population)

        self.assertTrue(round(result.expect[0][-1], 6) == round(1/3, 6))
        self.assertTrue(round(result.expect[1][-1], 6) == round(2/3, 6))

        psi0 = self.FLS.basis[3]

        result = qutip.mesolve(self.FLS.H, psi0, self.times, self.FLS.decay, self.population)

        self.assertTrue(round(result.expect[1][-1], 6) == round(1/3, 6))
        self.assertTrue(round(result.expect[0][-1], 6) == round(2/3, 6))

    def test_double_initial_condition(self):
        psi0 = 1/np.sqrt(2)*(self.FLS.basis[2] + self.FLS.basis[3])

        result = qutip.mesolve(self.FLS.H, psi0, self.times, self.FLS.decay, self.population)

        self.assertEqual(result.expect[0][0], 0.0)
        self.assertEqual(result.expect[1][0], 0.0)
        self.assertEqual(round(result.expect[2][0], 6), round(1/2, 6))
        self.assertEqual(round(result.expect[2][0], 6), round(1/2, 6))

    def test_double_decay_time(self):
        """Linewidth from excited-state decay is within 100 Hz of the defined linewidth."""
        psi0 = 1 / np.sqrt(2) * (self.FLS.basis[2] + self.FLS.basis[3])

        result = qutip.mesolve(self.FLS.H, psi0, self.times, self.FLS.decay, self.population)

        excited_state_population = result.expect[2] + result.expect[3]

        popt, pcov = curve_fit(exponential_decay, self.times, excited_state_population)
        perr = np.sqrt(np.diag(pcov))

        if False:
            plt.plot(self.times, excited_state_population, "--")
            plt.plot(self.times, exponential_decay(self.times, *popt), "o")
            plt.show()
            plt.close()

        linewidth = 2 * np.pi / popt[0]
        self.assertTrue(self.FLS.linewidth - linewidth < 100.0)
        self.assertTrue(perr[0] < 100.0)

    def test_double_decay_ratio(self):
        """The population in the ground-states after decay from a equally populated
        excited states is equal to the expected ratio."""
        psi0 = 1 / np.sqrt(2) * (self.FLS.basis[2] + self.FLS.basis[3])

        result = qutip.mesolve(self.FLS.H, psi0, self.times, self.FLS.decay, self.population)

        self.assertTrue(round(result.expect[0][-1], 6) == round(1/2, 6))
        self.assertTrue(round(result.expect[1][-1], 6) == round(1/2, 6))


if __name__ == "__main__":
    unittest.main()