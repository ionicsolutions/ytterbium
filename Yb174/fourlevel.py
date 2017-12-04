# -*- coding: utf-8 -*-
#
#   (c) 2017 Kilian Kluge
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
import itertools

import matplotlib.pyplot as plt
import numpy as np
import qutip

# Clebsch-Gordon coefficients
cg = np.zeros((4, 4))
cg[0][2] = cg[2][0] = cg[1][3] = cg[3][1] = 1 / np.sqrt(3)
cg[0][3] = cg[3][0] = cg[1][2] = cg[2][1] = np.sqrt(2 / 3)

# gJ factors
gJ = np.zeros(4)
gJ[0] = gJ[1] = 2
gJ[2] = gJ[3] = 3 / 2

# mJ factors
mJ = np.zeros(4)
mJ[0] = mJ[2] = -1 / 2
mJ[1] = mJ[3] = 1 / 2


class FourLevelSystem:
    linewidth = 19.6 * 10 ** 6  # Hz

    def __init__(self, delta=0.0, sat=1.0, polarization=(1, 0, 0), B=0.0):
        """Model of the 2S1/2-2P1/2 transition in 174Yb+ as a four-level system.

        :param delta: Laser detuning from resonance in MHz.
        :param sat: Laser intensity expressed as a multiple of the saturation intensity.
        :param polarization: Laser polarization as a 3-tuple (pi, sigma+, sigma-).
        :param B: Magnetic field in Gauss.
        """
        self.delta = delta
        self.sat = sat
        self.polarization = polarization
        self.B = B

        self.basis = [qutip.states.basis(4, i) for i in range(4)]

    @property
    def H(self):
        """Full Hamiltonian of the system."""
        laser_field = [2 * np.pi * self.delta * 10 ** 6 * self.basis[i] * self.basis[i].dag()
                       for i in range(4)]

        magnetic_field = [2 * np.pi * mJ[i] * gJ[i] * 1.4 * 10 ** 6 * self.B * self.basis[i] * self.basis[i].dag()
                          for i in range(4)]

        off_diagonal_elements = [self.omega[i][j] * self.basis[i] * self.basis[j].dag() * cg[i][j] ** 2
                                 for i, j in itertools.product(range(4), range(4))]

        return sum(laser_field) + sum(magnetic_field) + sum(off_diagonal_elements)

    @property
    def decay(self):
        """Decay terms prepared for use in `qutip.mesolve`."""
        return list(itertools.chain(*self.raw_decay))

    @property
    def raw_decay(self):
        """All decay terms."""
        decay = [[] for _ in range(4)]
        decay[2] = [np.sqrt(2 * np.pi * self.linewidth) * cg[i][2] * self.basis[i] * self.basis[2].dag()
                    for i in (0, 1)]
        decay[3] = [np.sqrt(2 * np.pi * self.linewidth) * cg[i][3] * self.basis[i] * self.basis[3].dag()
                    for i in (0, 1)]
        return decay

    @property
    def omega(self):
        """Rabi frequencies."""
        pi, sigma_plus, sigma_minus = self.polarization
        _omega = 2 * np.pi * self.linewidth * np.sqrt(self.sat / 2)

        omega = np.zeros((4, 4))
        omega[0][2] = omega[2][0] = omega[1][3] = omega[3][1] = _omega * pi
        omega[0][3] = omega[3][0] = _omega * sigma_plus
        omega[1][2] = omega[2][1] = _omega * sigma_minus
        return omega


if __name__ == "__main__":

    FLS = FourLevelSystem(-10.0, 0.05, (1, 2, 0), 0.5)

    psi0 = 1 / np.sqrt(2) * (FLS.basis[0] + FLS.basis[1])

    times = np.linspace(0.0, 10.0 * 10 ** -6, num=1000)

    populations = [state * state.dag() for state in FLS.basis]

    result = qutip.mesolve(FLS.H, psi0, times, FLS.decay, populations)
    print("Done calculating")

    for i in range(4):
        plt.plot(result.times * 10 ** 6, result.expect[i], label="%d" % i)

    plt.xlabel("Time in us")
    plt.ylabel("Population")
    plt.legend()
    plt.show()
    plt.close()
