import unittest

import matplotlib.pyplot as plt
import numpy as np
import qutip
from scipy.optimize import curve_fit

from lib.parallelize import mesolve
from Yb171.raman.fourlevel import FourLevelSystem


class TestSingleLaser(unittest.TestCase):
    """Basic tests of the model where only one laser is on."""

    def setUp(self):
        self.FLS = FourLevelSystem()

        self.population = [state * state.dag() for state in self.FLS.basis]
        self.times = np.linspace(0, 10.0 * 10 ** -6, num=1000)

    def test_pi_saturation(self):
        psi0 = self.FLS.basis[0]

        self.FLS.sat_pi = 1.0

        result = qutip.mesolve(self.FLS.H, psi0, self.times, self.FLS.raw_decay[3][0],
                               self.population)

        if False:
            for i in range(4):
                plt.plot(self.times, result.expect[i], label="%d" % i)
            plt.legend()
            plt.show()
            plt.close()

        self.assertTrue(abs(result.expect[0][-1] - 0.75) < 1e-6)
        self.assertTrue(abs(result.expect[3][-1] - 0.25) < 1e-6)

    def test_sigma_saturation(self):
        psi0 = self.FLS.basis[2]

        self.FLS.sat_sig = 1.0

        result = qutip.mesolve(self.FLS.H, psi0, self.times, self.FLS.raw_decay[3][2],
                               self.population)

        if False:
            for i in range(4):
                plt.plot(self.times, result.expect[i], label="%d" % i)
            plt.legend()
            plt.show()
            plt.close()

        self.assertTrue(abs(result.expect[2][-1] - 0.75) < 1e-6)
        self.assertTrue(abs(result.expect[3][-1] - 0.25) < 1e-6)

    def test_pi_pumping(self):
        psi0 = self.FLS.basis[0]

        self.FLS.sat_pi = 1.0

        result = qutip.mesolve(self.FLS.H, psi0, self.times, self.FLS.decay,
                               self.population)

        if False:
            for i in range(4):
                plt.plot(self.times, result.expect[i], label="%d" % i)
            plt.legend()
            plt.show()
            plt.close()

        self.assertTrue(result.expect[0][-1] < 1e-6)
        self.assertTrue(abs(result.expect[1][-1] - result.expect[2][-1]) < 1e-6)
        self.assertTrue(abs(result.expect[1][-1] - 0.5) < 1e-6)
        self.assertTrue(np.allclose(result.expect[1][-1], result.expect[2][-1]))

    def test_sigma_pumping(self):
        psi0 = self.FLS.basis[2]

        self.FLS.sat_sig = 1.0

        result = qutip.mesolve(self.FLS.H, psi0, self.times, self.FLS.decay,
                               self.population)

        if False:
            for i in range(4):
                plt.plot(self.times, result.expect[i], label="%d" % i)
            plt.legend()
            plt.show()
            plt.close()

        self.assertTrue(result.expect[2][-1] < 1e-6)
        self.assertTrue(abs(result.expect[0][-1] - result.expect[1][-1]) < 1e-6)
        self.assertTrue(abs(result.expect[0][-1] - 0.5) < 1e-6)
        self.assertTrue(np.allclose(result.expect[0][-1], result.expect[1][-1]))


def resonance_peak(pulse_duration):
    def excited_state_population(detuning, delta0, omega):
        omega_ = np.sqrt(omega ** 2 + (detuning - delta0) ** 2)
        return omega**2/omega_**2 * np.sin(omega_/2 * pulse_duration)**2
    return excited_state_population


class TestRamanTransitions(unittest.TestCase):

    def setUp(self):
        self.FLS = FourLevelSystem()

        self.population = [state * state.dag() for state in self.FLS.basis]
        self.times = np.linspace(0, 0.5 * 10 ** -6, num=500)

        self.FLS.sat_pi = 1000.0
        self.FLS.sat_sig = 1000.0

    def test_near_resonant_rotation(self):
        """Verify that Rabi oscillations occur when the system is driven with
        two strong lasers at two-photon resonance close to resonance."""
        psi0 = self.FLS.basis[0]

        self.FLS.delta = 0
        self.FLS.detuning = 2

        result = qutip.mesolve(self.FLS.H, psi0, self.times, self.FLS.decay,
                               self.population)

        if True:
            for i in range(4):
                plt.plot(self.times, result.expect[i], label="%d" % i)
            plt.legend()
            plt.show()
            plt.close()