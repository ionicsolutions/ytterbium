import unittest

import matplotlib.pyplot as plt
import numpy as np
import qutip
from scipy.optimize import curve_fit

from ...Yb174.tenlevel import TenLevelSystem


def exponential_decay(t, tau):
    return np.exp(-t / tau)


class TestIntersystemCoupling(unittest.TestCase):
    def setUp(self):
        self.TLS = TenLevelSystem(delta_SP=0.0, sat_SP=0.0,
                                  polarization_SP=(1, 0, 0),
                                  delta_D=0.0, sat_D=0.0,
                                  polarization_D=(1, 0, 0),
                                  B=0.0)
        self.population = [state * state.dag() for state in self.TLS.basis]

        self.times = np.linspace(0, 0.5 * 10 ** -6, num=10000)

    def test_branching_ratio_P_to_D(self):
        """Decay from the 2P1/2 state results in the expected ratio of 2S1/2 and 2D3/2 population."""
        psi0 = 1 / np.sqrt(2) * (self.TLS.basis[8] + self.TLS.basis[9])

        result = qutip.mesolve(self.TLS.H, psi0, self.times, self.TLS.decay, self.population)

        if False:
            for i in range(10):
                plt.plot(self.times * 10 ** 6, result.expect[i], label="%d" % i)
            plt.legend()
            plt.show()
            plt.close()

        final_s_population = result.expect[6][-1] + result.expect[7][-1]
        final_p_population = result.expect[8][-1] + result.expect[9][-1]
        final_d_population = result.expect[0][-1] + result.expect[2][-1] + result.expect[4][-1] + result.expect[5][-1]

        self.assertTrue(final_s_population + final_d_population > 0.9999)
        self.assertTrue(abs(final_s_population - 0.995) < 0.001)
        self.assertTrue(abs(final_d_population - 0.005) < 0.001)

    def test_branching_ratio_D_to_S(self):
        """Decay from the 2D[3/2]1/2 state results in the expected ratio of 2S1/2 and 2D3/2 population."""
        psi0 = 1 / np.sqrt(2) * (self.TLS.basis[1] + self.TLS.basis[3])

        result = qutip.mesolve(self.TLS.H, psi0, self.times, self.TLS.decay, self.population)

        if False:
            for i in range(10):
                plt.plot(self.times * 10 ** 6, result.expect[i], label="%d" % i)
            plt.legend()
            plt.show()
            plt.close()

        final_s_population = result.expect[6][-1] + result.expect[7][-1]
        final_d_population = result.expect[0][-1] + result.expect[2][-1] + result.expect[4][-1] + result.expect[5][-1]

        self.assertTrue(final_s_population + final_d_population > 0.9999)
        self.assertTrue(abs(final_s_population - 0.982) < 0.001)
        self.assertTrue(abs(final_d_population - 0.018) < 0.001)


class TestPumpingFromSPtoD(unittest.TestCase):
    def setUp(self):
        self.TLS = TenLevelSystem(delta_SP=0.0, sat_SP=0.0,
                                  polarization_SP=(1, 0, 0),
                                  delta_D=0.0, sat_D=0.0,
                                  polarization_D=(1, 0, 0),
                                  B=0.0)
        self.population = [state * state.dag() for state in self.TLS.basis]

        self.times = np.linspace(0, 500 * 10 ** -6, num=1000)

    def test_dark_pumping(self):
        """Pumping on the S-P transition without light on the D transition results in a complete population transfer
        from the S-P system into 2D3/2."""
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

    def test_sigma_polarized_dark_pumping(self):
        """Pumping on resonance with a large sigma+ component results in unevenly distributed 2D3/2 population
        when a magnetic field is present."""
        psi0 = 1 / np.sqrt(2) * (self.TLS.basis[8] + self.TLS.basis[9])

        self.TLS.sat_SP = 1.0
        self.TLS.delta_SP = 0.0
        self.TLS.polarization_SP = (1.0, 2.0, 1.0)

        self.TLS.B = 1.0

        result = qutip.mesolve(self.TLS.H, psi0, self.times, self.TLS.decay, self.population)

        if False:
            for i in (0, 2, 4, 5):
                plt.plot(self.times * 10 ** 6, result.expect[i], label="%d" % i)
            plt.legend()
            plt.show()
            plt.close()

        self.assertTrue(result.expect[5][-1] > result.expect[4][-1])
        self.assertTrue(result.expect[4][-1] > result.expect[2][-1])
        self.assertTrue(result.expect[2][-1] > result.expect[0][-1])

    def test_pi_polarized_dark_pumping_with_small_sigma(self):
        """Pumping red-detuned from resonance with pi-light results in unevenly distributed 2D3/2 population when
         both a small sigma-component (can be either sigma+, sigma-, or both) and a magnetic field are present."""
        psi0 = 1 / np.sqrt(2) * (self.TLS.basis[8] + self.TLS.basis[9])

        self.TLS.sat_SP = 1.0
        self.TLS.delta_SP = -15.0
        self.TLS.polarization_SP = (0.9, 0.1, 0.0)

        self.TLS.B = 5.0

        result = qutip.mesolve(self.TLS.H, psi0, self.times, self.TLS.decay, self.population)

        if False:
            for i in (0, 2, 4, 5):
                plt.plot(self.times * 10 ** 6, result.expect[i], label="%d" % i)
            plt.legend()
            plt.show()
            plt.close()

        self.assertTrue(result.expect[5][-1] > result.expect[4][-1])
        self.assertTrue(result.expect[4][-1] > result.expect[2][-1])
        self.assertTrue(result.expect[2][-1] > result.expect[0][-1])

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


if __name__ == "__main__":
    unittest.main()