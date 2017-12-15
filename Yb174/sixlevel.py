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

__all__ = ["SixLevelSystem"]

import itertools

import numpy as np
import qutip

from ..polarization import normalize

# Clebsch-Gordan coefficients
cg = np.zeros((6, 6))
cg[0][1] = cg[1][0] = cg[5][3] = cg[3][5] = 1 / np.sqrt(2)
cg[2][1] = cg[1][2] = cg[4][3] = cg[3][4] = 1 / np.sqrt(3)
cg[2][3] = cg[3][2] = cg[4][1] = cg[1][4] = 1 / np.sqrt(6)

# gJ factors
gJ = np.zeros(6)
gJ[0] = gJ[2] = gJ[4] = gJ[5] = 4 / 5
gJ[1] = gJ[3] = -7 / 6

# mJ factors
mJ = np.zeros(6)
mJ[0] = -3 / 2
mJ[1] = mJ[2] = -1 / 2
mJ[3] = mJ[4] = 1 / 2
mJ[5] = 3 / 2


# noinspection PyPep8Naming
class SixLevelSystem:
    linewidth = 4.11 * 10 ** 6  # Hz

    def __init__(self, delta=0.0, sat=0.0, polarization=(1, 0, 0), B=0.0):
        """Model of the 2D3/2-3D[3/2]1/2 transition in 174Yb+ as a six-level
        system.

        Based on the model described in H. Meyer (2014), the level numbering
        is identical.

        :param delta: Laser detuning from resonance in MHz.
        :param sat: Laser intensity expressed as a multiple of the saturation
                    intensity.
        :param polarization: Laser polarization as a 3-tuple
                             (pi, sigma+, sigma-).
        :param B: Magnetic field in Gauss.
        """
        self._polarization = (1, 0, 0)

        self.delta = delta
        self.sat = sat
        self.polarization = polarization
        self.B = B

        self.basis = [qutip.states.basis(6, i) for i in range(6)]

    @property
    def polarization(self):
        return self._polarization

    @polarization.setter
    def polarization(self, vector):
        self._polarization = normalize(vector)

    @property
    def H(self):
        """Full Hamiltonian of the system."""
        laser_field = [2 * np.pi * -self.delta * 10 ** 6
                       * self.basis[i] * self.basis[i].dag()
                       for i in (1, 3)]

        magnetic_field = [2 * np.pi * mJ[i] * gJ[i] * 1.4 * 10 ** 6 * self.B
                          * self.basis[i] * self.basis[i].dag()
                          for i in range(6)]

        off_diagonal_elements = [self.omega[i][j] / 2 * cg[i][j]
                                 * self.basis[i] * self.basis[j].dag()
                                 for i, j in itertools.product(range(6),
                                                               range(6))]

        H = sum(laser_field) + sum(magnetic_field) + sum(off_diagonal_elements)

        return H

    @property
    def decay(self):
        """Decay terms prepared for use in `qutip.mesolve`."""
        return list(itertools.chain(*self.raw_decay))

    @property
    def raw_decay(self):
        """All decay terms."""
        decay = [[] for _ in range(6)]
        decay[1] = [np.sqrt(2 * np.pi * self.linewidth) * cg[i][1]
                    * self.basis[i] * self.basis[1].dag()
                    for i in range(6)]
        decay[3] = [np.sqrt(2 * np.pi * self.linewidth) * cg[i][3]
                    * self.basis[i] * self.basis[3].dag()
                    for i in range(6)]

        return decay

    @property
    def omega(self):
        """Rabi frequencies."""
        pi, sigma_plus, sigma_minus = self.polarization
        _omega = 2 * np.pi * self.linewidth * np.sqrt(self.sat / 2)

        omega = np.zeros((6, 6))
        omega[1][0] = omega[0][1] = \
            omega[2][3] = omega[3][2] = _omega * sigma_plus
        omega[1][4] = omega[4][1] = \
            omega[3][5] = omega[5][3] = _omega * sigma_minus
        omega[1][2] = omega[2][1] = \
            omega[3][4] = omega[4][3] = _omega * pi

        return omega
