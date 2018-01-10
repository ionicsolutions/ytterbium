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
        self.times = np.linspace(0.0, 0.15 * 10 ** -6, num=200)

    def test_single_decay_time(self):
        for i in (4, 5, 6, 7):
            psi0 = self.ELS.basis[i]

            result = qutip.mesolve(self.ELS.H, psi0, self.times,
                                   self.ELS.decay, self.population)

            popt, pcov = curve_fit(exponential_decay, self.times, result.expect[i])
            perr = np.sqrt(np.diag(pcov))

            if False:
                plt.plot(self.times[::5], result.expect[i][::5], "--")
                plt.plot(self.times, exponential_decay(self.times, *popt), "o")
                plt.show()
                plt.close()

            linewidth = 1 / (2 * np.pi * popt[0])
            self.assertTrue(abs(self.ELS.linewidth - linewidth) < 100.0)
            self.assertTrue(perr[0] < 100.0)

    def test_single_decay_ratio(self):

        decays_to = {
            4: (1, 2, 3),
            5: (0, 1, 2),
            6: (0, 1, 3),
            7: (0, 2, 3)
        }

        for i in (4, 5, 6, 7):
            psi0 = self.ELS.basis[i]

            result = qutip.mesolve(self.ELS.H, psi0, self.times,
                                   self.ELS.decay, self.population)

            for k in decays_to[i]:
                self.assertTrue(abs(result.expect[k][-1] - 1/3) < 1e-6)

    def test_combined_decay_ratio(self):
        psi0 = 1/np.sqrt(4) * sum([self.ELS.basis[i] for i in (4, 5, 6, 7)])

        result = qutip.mesolve(self.ELS.H, psi0, self.times,
                               self.ELS.decay, self.population)

        if False:
            for i in range(4):
                plt.plot(self.times, result.expect[i], label="%d" % i)
            plt.legend()
            plt.show()
            plt.close()

        for i in range(4):
            self.assertTrue(abs(result.expect[i][-1] - 1 / 4) < 1e-6)

    def test_combined_decay_time(self):
        psi0 = 1 / np.sqrt(4) * sum([self.ELS.basis[i] for i in (4, 5, 6, 7)])

        result = qutip.mesolve(self.ELS.H, psi0, self.times,
                               self.ELS.decay, self.population)

        excited_state_population = sum([result.expect[i] for i in (4, 5, 6, 7)])

        popt, pcov = curve_fit(exponential_decay, self.times, excited_state_population)
        perr = np.sqrt(np.diag(pcov))

        if False:
            plt.plot(self.times, excited_state_population, "--")
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
        self.times = np.linspace(0.0, 1.0 * 10 ** -6, num=2000)

    def test_saturation_two_level(self):
        psi0 = self.ELS.basis[2]

        self.ELS.sat = 1/3
        self.ELS.polarization = (1, 0, 0)
        self.ELS.delta = 0

        result = qutip.mesolve(self.ELS.H, psi0, self.times,
                               self.ELS.raw_decay[4][2], self.population)

        if False:
            for i in range(8):
                plt.plot(self.times, result.expect[i], label="%d" % i)
            plt.legend()
            plt.show()
            plt.close()

        self.assertTrue(abs(result.expect[2][-1] - 3/4) < 1e-6)
        self.assertTrue(abs(result.expect[4][-1] - 1/4) < 1e-6)

    def test_forbidden_transitions(self):
        psi0 = self.ELS.basis[2]

        self.ELS.polarization = (1, 0, 0)
        self.ELS.sat = 1.0
        self.ELS.delta = self.ELS.p_splitting

        result = qutip.mesolve(self.ELS.H, psi0, self.times,
                               self.ELS.decay, self.population)

        if False:
            for i in range(8):
                plt.plot(self.times, result.expect[i], label="%d" % i)
            plt.legend()
            plt.show()
            plt.close()

        self.assertTrue(abs(result.expect[2][-1] - 1.0) < 0.001)
        self.assertTrue(result.expect[6][-1] < 0.001)

        psi0 = self.ELS.basis[0]

        self.ELS.polarization = (1, 0, 0)
        self.ELS.sat = 1.0
        self.ELS.delta = self.ELS.s_splitting

        result = qutip.mesolve(self.ELS.H, psi0, self.times,
                               self.ELS.decay, self.population)

        if False:
            for i in range(8):
                plt.plot(self.times, result.expect[i], label="%d" % i)
            plt.legend()
            plt.show()
            plt.close()

        self.assertTrue(abs(result.expect[0][-1] - 1.0) < 0.001)
        self.assertTrue(result.expect[4][-1] < 0.001)

    def test_p_splitting(self):
        psi0 = self.ELS.basis[2]

        self.ELS.polarization = (0, 1, 1)
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

        self.assertTrue(result.expect[2][-1] < 1e-3)

        self.ELS.delta = 0.5 * self.ELS.p_splitting

        result = qutip.mesolve(self.ELS.H, psi0, self.times,
                               self.ELS.decay, self.population)

        if False:
            for i in range(8):
                plt.plot(self.times, result.expect[i], label="%d" % i)
            plt.legend()
            plt.show()
            plt.close()

        self.assertTrue(result.expect[2][-1] > 0.98)

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

        self.assertTrue(result.expect[0][-1] < 1e-3)

        self.ELS.delta = (self.ELS.s_splitting + self.ELS.p_splitting)/2

        result = qutip.mesolve(self.ELS.H, psi0, self.times, self.ELS.decay, self.population)

        if False:
            for i in range(8):
                plt.plot(self.times, result.expect[i], label="%d" % i)
            plt.legend()
            plt.show()
            plt.close()

        self.assertTrue(result.expect[0][-1] > 0.98)


class TestMagneticField(unittest.TestCase):

    def setUp(self):
        self.ELS = EightLevelSystem()
        self.population = [state * state.dag() for state in self.ELS.basis]
        self.times = np.linspace(0.0, 0.2 * 10 ** -6, num=300)

    def test_transitions(self):

        psi0 = self.ELS.basis[0]

        self.ELS.polarization = (1, 1, 1)
        self.ELS.B = 30.0
        self.ELS.sat = 10.0

        detunings = np.linspace(-50.0, 50.0, num=50) + (self.ELS.s_splitting + self.ELS.p_splitting)

        populations = [[] for _ in range(8)]

        for detuning in detunings:
            self.ELS.delta = detuning

            result = qutip.mesolve(self.ELS.H, psi0, self.times, self.ELS.decay, self.population)

            for i in range(8):
                populations[i].append(result.expect[i][-1])

        if False:
            for i, population in enumerate(populations):
                plt.plot(detunings, population, label="%d" % i)

            plt.legend()
            plt.show()
            plt.close()


if __name__ == "__main__":
    unittest.main()
