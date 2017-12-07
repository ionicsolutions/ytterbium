import unittest

import matplotlib.pyplot as plt
import numpy as np
import qutip
from scipy.optimize import curve_fit

from Yb174.tenlevel import TenLevelSystem


def exponential_decay(t, tau):
    return np.exp(-t / tau)


class TestRepumping(unittest.TestCase):
    def setUp(self):
        self.TLS = TenLevelSystem(delta_SP=0.0, sat_SP=0.0,
                                  polarization_SP=(1, 0, 0),
                                  delta_D=0.0, sat_D=0.0,
                                  polarization_D=(1, 0, 0),
                                  B=0.0)
        self.population = [state * state.dag() for state in self.TLS.basis]

        self.times = np.linspace(0, 500 * 10 ** -6, num=1000)

    def test_dark_pumping(self):
        psi0 = 1 / np.sqrt(2) * (self.TLS.basis[8] + self.TLS.basis[9])

        self.TLS.sat_SP = 5.0
        self.TLS.delta_SP = -15.0

        result = qutip.mesolve(self.TLS.H, psi0, self.times, self.TLS.decay, self.population)

        if False:
            for i in range(10):
                plt.plot(self.times * 10 ** 6, result.expect[i], label="%d" % i)
            plt.legend()
            plt.show()
            plt.close()

        self.assertTrue(result.expect[6][-1] < 1e-6)
        self.assertTrue(result.expect[7][-1] < 1e-6)
        final_d_population = result.expect[0][-1] + result.expect[2][-1] + result.expect[4][-1] + result.expect[5][-1]
        self.assertTrue(final_d_population > 1 - 1e-6)

    def test_branching_ratio(self):
        psi0 = 1 / np.sqrt(2) * (self.TLS.basis[8] + self.TLS.basis[9])

        times = np.linspace(0, 0.1 * 10 ** -6, num=10000)

        result = qutip.mesolve(self.TLS.H, psi0, times, self.TLS.decay, self.population)

        if False:
            for i in range(10):
                plt.plot(times * 10 ** 6, result.expect[i], label="%d" % i)
            plt.legend()
            plt.show()
            plt.close()

        final_s_population = result.expect[6][-1] + result.expect[7][-1]
        final_p_population = result.expect[8][-1] + result.expect[9][-1]
        final_d_population = result.expect[0][-1] + result.expect[2][-1] + result.expect[4][-1] + result.expect[5][-1]

        self.assertTrue(final_s_population + final_d_population > 0.9999)
        self.assertTrue(abs(final_s_population - 0.995) < 0.001)
        self.assertTrue(abs(final_d_population - 0.005) < 0.001)

    def test_dark_pumping_rate(self):
        psi0 = 1 / np.sqrt(2) * (self.TLS.basis[8] + self.TLS.basis[9])

        self.TLS.delta_SP = 0.0

        sat_param = [0.01, 0.1, 0.5, 1.0, 10.0, 100.0]
        decay_time = []

        for sat in sat_param:
            self.TLS.sat_SP = sat
            result = qutip.mesolve(self.TLS.H, psi0, self.times, self.TLS.decay, self.population)

            sp_population = result.expect[6] + result.expect[7] + result.expect[8] + result.expect[9]
            d_population = result.expect[0] + result.expect[1] + result.expect[2] + result.expect[3]

            popt, pcov = curve_fit(exponential_decay, self.times, sp_population, p0=[100.0 * 10 ** -6])

            decay_time.append(popt[0] * 10 ** 6)

            if False:
                plt.plot(self.times * 10 ** 6, sp_population,
                         label="SP, $\\tau$ = %0.2f us, s = %0.2f" % (popt[0] * 10 ** 6, sat))
                plt.plot(self.times * 10 ** 6, d_population, label="D, s = %0.2f" % (sat))

        if False:
            plt.legend()
            plt.show()
            plt.close()

        if False:
            plt.plot(sat_param, decay_time, "o")
            plt.show()
            plt.close()


class TestPulsedSixLevelSystem(unittest.TestCase):
    def setUp(self):
        self.TLS = TenLevelSystem(delta_SP=0.0, sat_SP=0.0,
                                  polarization_SP=(1, 0, 0),
                                  delta_D=0.0, sat_D=0.0,
                                  polarization_D=(1, 0, 0),
                                  B=0.0)
        self.population = [state * state.dag() for state in self.TLS.basis]

        self.TLS.sat_D = 0.05

        self.detunings = np.linspace(-20.0, 20.0, num=200)
        self.times = np.linspace(0, 10 * 10 ** -6, num=1000)

    def test_pi_pulses(self):
        self.TLS.polarization_D = (1, 0, 0)

        psi0 = 1 / np.sqrt(4) * (self.TLS.basis[0] + self.TLS.basis[2] +
                                 self.TLS.basis[4] + self.TLS.basis[5])

        s_population = {}

        for field in [0.5, 2.0, 4.0, 8.0]:
            s_population[field] = []
            self.TLS.B = field

            for delta in self.detunings:
                self.TLS.delta_D = delta
                result = qutip.mesolve(self.TLS.H, psi0, self.times,
                                       self.TLS.decay, self.population)

                s_population[field].append(result.expect[6][-1] +
                                           result.expect[7][-1])

        if True:
            for field, s_pop in s_population.items():
                plt.plot(self.detunings, s_pop, "o", label="%0.2f G" % field)

            plt.legend()
            plt.title("10 us $\pi$-pulses for different magnetic fields")
            plt.xlabel("935 nm laser detuning from resonance [MHz]")
            plt.ylabel("Total S-state population")
            plt.tight_layout()
            plt.savefig("test_tenlevel_test_pi_pulses.png")
            plt.close()

    def test_sigma_pulses(self):
        self.TLS.polarization_D = (0, 1, 1)

        psi0 = 1 / np.sqrt(4) * (self.TLS.basis[0] + self.TLS.basis[2] +
                                 self.TLS.basis[4] + self.TLS.basis[5])

        s_population = {}

        for field in [0.5, 2.0, 4.0, 8.0]:
            s_population[field] = []
            self.TLS.B = field

            for delta in self.detunings:
                self.TLS.delta_D = delta
                result = qutip.mesolve(self.TLS.H, psi0, self.times,
                                       self.TLS.decay, self.population)

                s_population[field].append(result.expect[6][-1] +
                                           result.expect[7][-1])

        if True:
            for field, s_pop in s_population.items():
                plt.plot(self.detunings, s_pop, "o", label="%0.2f G" % field)

            plt.legend()
            plt.title("10 us $\sigma$-pulses for different magnetic fields")
            plt.xlabel("935 nm laser detuning from resonance [MHz]")
            plt.ylabel("Total S-state population")
            plt.savefig("test_tenlevel_test_sigma_pulses.png")
            plt.close()

    def test_mixed_pulses(self):
        self.TLS.polarization_D = (1, 1, 1)

        psi0 = 1 / np.sqrt(4) * (self.TLS.basis[0] + self.TLS.basis[2] +
                                 self.TLS.basis[4] + self.TLS.basis[5])

        s_population = {}

        for field in [0.5, 2.0, 4.0, 8.0]:
            s_population[field] = []
            self.TLS.B = field

            for delta in self.detunings:
                self.TLS.delta_D = delta
                result = qutip.mesolve(self.TLS.H, psi0, self.times,
                                       self.TLS.decay, self.population)

                s_population[field].append(result.expect[6][-1] +
                                           result.expect[7][-1])

        if True:
            for field, s_pop in s_population.items():
                plt.plot(self.detunings, s_pop, "o", label="%0.2f G" % field)

            plt.legend()
            plt.title("10 us equally mixed $\pi$- and $\sigma$-pulses "
                      "for different magnetic fields")
            plt.xlabel("935 nm laser detuning from resonance [MHz]")
            plt.ylabel("Total S-state population")
            plt.savefig("test_tenlevel_test_mixed_pulses.png")
            plt.close()
