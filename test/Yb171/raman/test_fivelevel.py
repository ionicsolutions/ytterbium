import unittest

import matplotlib.pyplot as plt
import numpy as np
import qutip
from scipy.optimize import curve_fit

from ....parallelize import vary, mesolve
from ....Yb171.raman.fivelevel import FourLevelSystem


class TestRamanTransitions(unittest.TestCase):

    def setUp(self):
        self.FLS = FourLevelSystem()

        self.population = [state * state.dag() for state in self.FLS.basis]
        self.times = np.linspace(0, 1.0 * 10 ** -6, num=1000)

        self.FLS.sat_M = 1.0
        self.FLS.sat_S = 1.0

    def test_near_resonant_rotation(self):
        """Verify that Rabi oscillations occur when the system is driven with
        two strong lasers at two-photon resonance close to resonance."""
        psi0 = self.FLS.basis[0]

        self.FLS.delta = 0
        self.FLS.detuning = 5
        self.FLS.sat_M = 100.0
        self.FLS.sat_S = 100.0

        self.FLS.polarization_M = (1, 1, 1)
        self.FLS.polarization_S = (1, 1, 1)

        result = qutip.mesolve(self.FLS.H, psi0, self.times, self.FLS.decay,
                               self.population)

        if True:
            for i in range(4):
                plt.plot(self.times, result.expect[i], label="%d" % i)
            plt.legend()
            plt.show()
            plt.close()


        return

        result = qutip.mesolve(self.FLS.H, psi0, self.times, self.FLS.decay,
                               self.population)

        if True:
            for i in range(4):
                plt.plot(self.times, result.expect[i], label="%d" % i)
            plt.legend()
            plt.show()
            plt.close()

        self.FLS.polarization_S = (0, 1, 0)
        self.FLS.polarization_M = (0, 0, 1)

        result = qutip.mesolve(self.FLS.H, psi0, self.times, self.FLS.decay,
                               self.population)

        if True:
            for i in range(4):
                plt.plot(self.times, result.expect[i], label="%d" % i)
            plt.legend()
            plt.show()
            plt.close()

        self.FLS.polarization_S = (1, 1, 0)
        self.FLS.polarization_M = (1, 0, 1)

        result = qutip.mesolve(self.FLS.H, psi0, self.times, self.FLS.decay,
                               self.population)

        if True:
            for i in range(4):
                plt.plot(self.times, result.expect[i], label="%d" % i)
            plt.legend()
            plt.show()
            plt.close()

        self.FLS.polarization_S = (1, 0, 1)
        self.FLS.polarization_M = (1, 0, 1)

        result = qutip.mesolve(self.FLS.H, psi0, self.times, self.FLS.decay,
                               self.population)

        if True:
            for i in range(4):
                plt.plot(self.times, result.expect[i], label="%d" % i)
            plt.legend()
            plt.show()
            plt.close()




if __name__ == "__main__":
    unittest.main()