import unittest

import matplotlib.pyplot as plt
import numpy as np
import qutip

from ytterbium.Yb171 import EightLevelSystem


class TestDrivenSystem(unittest.TestCase):

    def setUp(self):
        self.ELS = EightLevelSystem()

        self.population = [state * state.dag() for state in self.ELS.basis]

        self.times = np.linspace(0.0, 1.0*10**-6, num=1000)

    def test_drive_it(self):
        psi0 = self.ELS.basis[2]

        self.ELS.polarization = (1, 1, 1)
        self.ELS.sat = 1.0

        result = qutip.mesolve(self.ELS.H, psi0, self.times, self.ELS.decay, self.population)

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

        result = qutip.mesolve(self.ELS.H, psi0, self.times, self.ELS.decay, self.population)

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