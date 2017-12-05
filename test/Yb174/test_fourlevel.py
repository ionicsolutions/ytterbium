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
        self.times = np.linspace(0, 0.1*10**-6, num=1000)

    def test_initial_condition(self):
        psi0 = self.FLS.basis[2]

        result = qutip.mesolve(self.FLS.H, psi0, self.times, self.FLS.decay,
                               self.population)

        self.assertEqual(result.expect[0][0], 0.0)
        self.assertEqual(result.expect[1][0], 0.0)
        self.assertEqual(result.expect[2][0], 1.0)
        self.assertEqual(result.expect[3][0], 0.0)

    def test_P_state_decay(self):

        psi0 = self.FLS.basis[2]

        result = qutip.mesolve(self.FLS.H, psi0, self.times, self.FLS.decay, self.population)

        popt, pcov = curve_fit(exponential_decay, self.times, result.expect[2])

        #plt.plot(self.times, result.expect[2], "--")
        #plt.plot(self.times, exponential_decay(self.times, *popt), "o")
        #plt.show()
        #plt.close()

        print(1/popt[0], self.FLS.linewidth, result.expect[2][-1])


    def test_P_state_symmetry(self):
        psi0 = self.FLS.basis[2]


if __name__ == "__main__":
    unittest.main()