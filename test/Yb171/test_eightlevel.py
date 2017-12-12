import unittest

import matplotlib.pyplot as plt
import numpy as np
import qutip
from scipy.optimize import curve_fit

from ...Yb171.eightlevel import EightLevelSystem


def exponential_decay(t, tau):
    return np.exp(-t / tau)


class TestUndrivenSystem(unittest.TestCase):

    def setUp(self):
        self.ELS = EightLevelSystem()

        self.population = [state * state.dag() for state in self.ELS.basis]

        self.times = np.linspace(0.0, 1.0 * 10 ** -6, num=1000)

    def test_ground_states_remain(self):
        """Population in the ground states is constant."""
        psi0 = 1/np.sqrt(4) * sum(self.ELS.basis[0:4])

        result = qutip.mesolve(self.ELS.H, psi0,
                               self.times, self.ELS.decay, self.population)

        for i in range(4):
            self.assertTrue(abs(result.expect[i][-1] - 0.25) < 1e-6)

    def test_branching_ratios(self):
        """The population in the ground states is 1/3 each when the
        atom is initially excited into one of the excited state levels.
        """

        final_states = {4: [1, 2, 3],
                        5: [0, 1, 2],
                        6: [0, 1, 3],
                        7: [0, 2, 3]}

        for i in range(4, 8):
            psi0 = self.ELS.basis[i]

            result = qutip.mesolve(self.ELS.H, psi0,
                                   self.times, self.ELS.decay, self.population)

            for state in final_states[i]:
                self.assertTrue(abs(result.expect[state][-1] - 1/3) < 1e-6)

    def test_single_decay_time(self):
        """Linewidth from excited-state decay is within 100 Hz
        of the defined linewidth.
        """
        for i in range(4, 8):
            psi0 = self.ELS.basis[i]

            result = qutip.mesolve(self.ELS.H, psi0,
                                   self.times, self.ELS.decay, self.population)

            popt, pcov = curve_fit(exponential_decay, self.times,
                                   result.expect[i])
            perr = np.sqrt(np.diag(pcov))

            if False:
                plt.plot(self.times, result.expect[i], "--")
                plt.plot(self.times, exponential_decay(self.times, *popt), "o")
                plt.show()
                plt.close()

            linewidth = 1 / (2 * np.pi * popt[0])
            self.assertTrue(abs(self.ELS.linewidth - linewidth) < 100.0)
            self.assertTrue(perr[0] < 100.0)


class TestDrivenSystem(unittest.TestCase):

    def setUp(self):
        self.ELS = EightLevelSystem()

        self.population = [state * state.dag() for state in self.ELS.basis]

        self.times = np.linspace(0.0, 1.0 * 10 ** -6, num=1000)

    def test_drive_it(self):
        psi0 = self.ELS.basis[2]

        self.ELS.polarization = (1, 1, 1)
        self.ELS.sat = 1.0

        result = qutip.mesolve(self.ELS.H, psi0, self.times,
                               self.ELS.decay, self.population)

        if False:
            for i in range(4):
                plt.plot(self.times, result.expect[i], label="%d" % i)
            plt.legend()
            plt.show()
            plt.close()

            for i in range(4, 8):
                plt.plot(self.times, result.expect[i], label="%d" % i)
            plt.legend()
            plt.show()
            plt.close()

    def test_p_splitting(self):
        psi0 = self.ELS.basis[2]

        self.ELS.polarization = (1, 1, 1)
        self.ELS.B = 0.2
        self.ELS.sat = 10.0

        self.ELS.delta = self.ELS.p_splitting

        result = qutip.mesolve(self.ELS.H, psi0, self.times,
                               self.ELS.decay, self.population)

        if False:
            for i in range(8):
                plt.plot(self.times, result.expect[i], label="%d" % i)
            plt.legend()
            plt.show()
            plt.close()

    def test_s_splitting(self):
        psi0 = self.ELS.basis[0]

        self.ELS.polarization = (1, 1, 1)
        self.ELS.B = 0.2
        self.ELS.sat = 10.0

        self.ELS.delta = self.ELS.s_splitting + self.ELS.p_splitting

        result = qutip.mesolve(self.ELS.H, psi0, self.times, self.ELS.decay, self.population)
        if False:
            for i in range(8):
                plt.plot(self.times, result.expect[i], label="%d" % i)
            plt.legend()
            plt.show()
            plt.close()


if __name__ == "__main__":
    unittest.main()
