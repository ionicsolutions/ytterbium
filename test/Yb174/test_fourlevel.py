import unittest

import matplotlib.pyplot as plt
import numpy as np
import qutip
from scipy.optimize import curve_fit

from ...Yb174 import FourLevelSystem


def exponential_decay(t, tau):
    return np.exp(-t / tau)


def lorentzian(v, gamma, A):
    return A / (2 * np.pi) * (gamma / 2) ** 2 / (v ** 2 + (gamma / 2) ** 2)


def R(s0, gamma):
    def r(delta):
        return (s0 * gamma / 2) / (1 + s0 + (2 * delta / gamma) ** 2)

    return r


class TestUndrivenSystem(unittest.TestCase):

    def setUp(self):
        self.FLS = FourLevelSystem(delta=0.0, sat=0.0, polarization=(1, 0, 0),
                                   B=0)
        self.population = [state * state.dag() for state in self.FLS.basis]
        self.times = np.linspace(0, 0.15 * 10 ** -6, num=1000)

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
                plt.plot(self.times, result.expect[2 + i], "--")
                plt.plot(self.times, exponential_decay(self.times, *popt), "o")
                plt.show()
                plt.close()

            linewidth = 1 / (2 * np.pi * popt[0])
            self.assertTrue(abs(self.FLS.linewidth - linewidth) < 100.0)
            self.assertTrue(perr[0] < 100.0)

    def test_single_decay_ratio(self):
        """The population in the ground-states after decay from a single excited state is
        equal to the expected ratio."""
        psi0 = self.FLS.basis[2]

        result = qutip.mesolve(self.FLS.H, psi0, self.times, self.FLS.decay, self.population)

        self.assertTrue(round(result.expect[0][-1], 6) == round(1 / 3, 6))
        self.assertTrue(round(result.expect[1][-1], 6) == round(2 / 3, 6))

        psi0 = self.FLS.basis[3]

        result = qutip.mesolve(self.FLS.H, psi0, self.times, self.FLS.decay, self.population)

        self.assertTrue(round(result.expect[1][-1], 6) == round(1 / 3, 6))
        self.assertTrue(round(result.expect[0][-1], 6) == round(2 / 3, 6))

    def test_double_initial_condition(self):
        psi0 = 1 / np.sqrt(2) * (self.FLS.basis[2] + self.FLS.basis[3])

        result = qutip.mesolve(self.FLS.H, psi0, self.times, self.FLS.decay, self.population)

        self.assertEqual(result.expect[0][0], 0.0)
        self.assertEqual(result.expect[1][0], 0.0)
        self.assertEqual(round(result.expect[2][0], 6), round(1 / 2, 6))
        self.assertEqual(round(result.expect[2][0], 6), round(1 / 2, 6))

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

        linewidth = 1 / (2 * np.pi * popt[0])
        self.assertTrue(abs(self.FLS.linewidth - linewidth) < 100.0)
        self.assertTrue(perr[0] < 100.0)

    def test_double_decay_ratio(self):
        """The population in the ground-states after decay from a equally populated
        excited states is equal to the expected ratio."""
        psi0 = 1 / np.sqrt(2) * (self.FLS.basis[2] + self.FLS.basis[3])

        result = qutip.mesolve(self.FLS.H, psi0, self.times, self.FLS.decay, self.population)

        self.assertTrue(round(result.expect[0][-1], 6) == round(1 / 2, 6))
        self.assertTrue(round(result.expect[1][-1], 6) == round(1 / 2, 6))


class TestDrivenSystem(unittest.TestCase):

    def setUp(self):
        self.FLS = FourLevelSystem(delta=0.0, sat=0.0, polarization=(1, 0, 0),
                                   B=0)
        self.population = [state * state.dag() for state in self.FLS.basis]
        self.times = np.linspace(0, 2.0 * 10 ** -6, num=1000)

    def test_no_laser_no_change(self):
        psi0 = self.FLS.basis[0]

        result = qutip.mesolve(self.FLS.H, psi0, self.times, self.FLS.decay, self.population)

        self.assertEqual(result.expect[0][-1], 1.0)
        self.assertEqual(result.expect[1][-1], 0.0)
        self.assertEqual(result.expect[2][-1], 0.0)
        self.assertEqual(result.expect[3][-1], 0.0)

    def test_saturation_two_level(self):
        psi0 = self.FLS.basis[0]

        self.FLS.sat = 1/3

        result = qutip.mesolve(self.FLS.H, psi0, self.times, self.FLS.raw_decay[2][0], self.population)

        if False:
            for i in range(4):
                plt.plot(self.times, result.expect[i], label="%d" % i)
            plt.legend()
            plt.show()
            plt.close()

        self.assertEqual(round(result.expect[0][-1] + result.expect[2][-1], 6), 1.0)
        self.assertEqual(round(result.expect[0][-1], 6), 0.75)
        self.assertEqual(round(result.expect[2][-1], 6), 0.25)

    def test_sigma_plus_pumping(self):
        psi0 = self.FLS.basis[0]

        self.FLS.sat = 1.0
        self.FLS.polarization = (0, 1, 0)

        result = qutip.mesolve(self.FLS.H, psi0, self.times, self.FLS.decay, self.population)

        if False:
            for i in range(4):
                plt.plot(self.times, result.expect[i], label="%d" % i)
            plt.legend()
            plt.show()
            plt.close()

        self.assertTrue(result.expect[0][-1] < 1e-6)
        self.assertTrue(result.expect[2][-1] < 1e-6)
        self.assertTrue(result.expect[3][-1] < 1e-6)
        self.assertTrue(result.expect[1][-1] > 1 - 1e-6)

    def test_sigma_driving(self):
        psi0 = self.FLS.basis[0]

        self.FLS.sat = 1.0
        self.FLS.polarization = (0, 1, 1)

        result = qutip.mesolve(self.FLS.H, psi0, self.times, self.FLS.decay, self.population)

        if False:
            for i in range(4):
                plt.plot(self.times, result.expect[i], label="%d" % i)
            plt.legend()
            plt.show()
            plt.close()

        self.assertTrue(result.expect[0][-1] - result.expect[1][-1] < 1e-6)
        self.assertTrue(result.expect[2][-1] - result.expect[3][-1] < 1e-6)

    def test_mixed_saturation(self):
        psi0 = self.FLS.basis[0]

        self.FLS.sat = 1.0
        self.FLS.polarization = (1, 1, 1)

        result = qutip.mesolve(self.FLS.H, psi0, self.times, self.FLS.decay, self.population)

        if False:
            for i in range(4):
                plt.plot(self.times, result.expect[i], label="%d" % i)
            plt.legend()
            plt.show()
            plt.close()

        self.assertTrue(result.expect[0][-1] - result.expect[1][-1] < 1e-6)
        self.assertTrue(result.expect[2][-1] - result.expect[3][-1] < 1e-6)

    def test_sigma_saturation(self):
        psi0 = self.FLS.basis[1]

        self.FLS.sat = 1.0
        self.FLS.polarization = (0, 1, 1)

        result = qutip.mesolve(self.FLS.H, psi0, self.times, self.FLS.decay, self.population)

        if False:
            for i in range(4):
                plt.plot(self.times, result.expect[i], label="%d" % i)
            plt.legend()
            plt.show()
            plt.close()

        self.assertTrue(result.expect[0][-1] - result.expect[1][-1] < 1e-6)
        self.assertTrue(result.expect[2][-1] - result.expect[3][-1] < 1e-6)


class TestMagneticField(unittest.TestCase):

    def setUp(self):
        self.FLS = FourLevelSystem(sat=0.5)

        self.population = [state * state.dag() for state in self.FLS.basis]

        self.times = np.linspace(0, 2 * 10 ** -6, num=500)

    def test_sigma_plus_is_sigma_plus(self):
        """If the laser is blue-detuned, increasing the magnetic field
        brings the sigma-plus transition into resonance.
        """
        psi0 = self.FLS.basis[0]

        fields = np.linspace(0, 10.0, num=30)

        self.FLS.delta = 5 * 1.4 * (1/2 * 2 + 1/2 * 3/2)
        self.FLS.polarization = (0, 1, 0)

        excited = []
        excited_max = []
        for field in fields:
            self.FLS.B = field
            result = qutip.mesolve(self.FLS.H, psi0,
                                   self.times, self.FLS.decay, self.population)
            excited.append((result.expect[3], "%0.2f G" % field))
            excited_max.append(np.max(result.expect[3]))

        if False:
            plt.plot(fields, excited_max, "o")
            plt.show()
            plt.close()

        if False:
            for trace, label in excited:
                plt.plot(self.times, trace, label=label)
            plt.legend()
            plt.show()
            plt.close()

        max_index = np.argmax(excited_max)
        max_field = fields[max_index]
        field_step = abs(max_field - fields[max_index - 1])

        self.assertTrue(4 < max_field < 6)
        self.assertTrue(max_field - field_step <= 5 <= max_field + field_step)

    def test_pi_transition_detuning(self):
        """For a positive magnetic field, the pi transition with negative
        mJ coefficients is blue-detuned, the pi transition with positive
        mJ coefficients is red-detuned from the bare atomic resonance.
        """
        fields = np.linspace(8.0, 16.0, num=30)

        self.FLS.polarization = (1, 0, 0)

        for i in (0, 1):
            psi0 = self.FLS.basis[i]

            self.FLS.delta = (-1) ** i * 12 * 1.4 * (1 / 2 * 2 - 1 / 2 * 3 / 2)

            excited = []
            excited_max = []
            for field in fields:
                self.FLS.B = field
                result = qutip.mesolve(self.FLS.H, psi0, self.times,
                                       self.FLS.decay, self.population)
                excited.append((result.expect[2 + i], "%0.2f G" % field))
                excited_max.append(np.max(result.expect[2 + i]))

            if False:
                plt.plot(fields, excited_max, "o")
                plt.show()
                plt.close()

            if False:
                for trace, label in excited:
                    plt.plot(self.times, trace, label=label)
                plt.legend()
                plt.show()
                plt.close()

            max_index = np.argmax(excited_max)
            max_field = fields[max_index]
            field_step = abs(max_field - fields[max_index - 1])

            self.assertTrue(11 < max_field < 13)
            self.assertTrue(max_field - field_step <= 12
                            <= max_field + field_step)


class TestSpectroscopy(unittest.TestCase):

    def setUp(self):
        self.FLS = FourLevelSystem(delta=0.0, sat=0.0, polarization=(1, 0, 0),
                                   B=0)
        self.population = [state * state.dag() for state in self.FLS.basis]
        self.times = np.linspace(0, 2.0 * 10 ** -6, num=1000)

    def test_linewidth_pi(self):
        psi0 = 1 / np.sqrt(2) * (self.FLS.basis[0] + self.FLS.basis[1])

        self.FLS.sat = 0.0001
        self.FLS.polarization = (1, 0, 0)

        self.times = np.linspace(0, 0.2 * 10 ** -6, num=100)

        detuning = np.linspace(-81.0, 79.0, num=80)
        transfer = []

        for delta in detuning:
            self.FLS.delta = delta
            result = qutip.mesolve(self.FLS.H, psi0, self.times, self.FLS.decay, self.population)
            transfer.append(result.expect[2][-1] + result.expect[3][-1])

            if False:
                plt.plot(result.times, result.expect[2], "--", label="%0.2f" % delta)

        if False:
            plt.legend()
            plt.show()
            plt.close()

        popt, pcov = curve_fit(lorentzian, detuning, transfer, p0=[1.0, 20.0])
        fit_detuning = np.linspace(min(detuning), max(detuning), num=1000)

        if False:
            plt.plot(detuning, transfer, "o")
            plt.plot(fit_detuning, lorentzian(fit_detuning, *popt), "--", label="$\Gamma$ = %0.2f MHz" % popt[0])
            plt.legend()
            plt.show()
            plt.close()

        self.assertTrue(abs(popt[0] - self.FLS.linewidth / 10 ** 6) < 0.1)


if __name__ == "__main__":
    unittest.main()
