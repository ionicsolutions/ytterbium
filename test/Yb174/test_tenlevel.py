import unittest

import matplotlib.pyplot as plt
import numpy as np
import qutip

from Yb174.tenlevel import TenLevelSystem


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
