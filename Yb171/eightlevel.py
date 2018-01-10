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

__all__ = ["EightLevelSystem"]

import itertools

import numpy as np
import qutip

from ..polarization import normalize

# Clebsch-Gordan coefficients (as calculated with ytterbium.clebsch_gordan)
cg = np.zeros((8, 8))

# pi transitions
cg[0][6] = cg[6][0] = 1 / np.sqrt(3)
cg[1][5] = cg[5][1] = -1 / np.sqrt(3)
cg[2][4] = cg[4][2] = 1 / np.sqrt(3)
cg[3][7] = cg[7][3] = 1 / np.sqrt(3)

# sigma+ transitions
cg[0][7] = cg[7][0] = -1 / np.sqrt(3)
cg[1][4] = cg[4][1] = 1 / np.sqrt(3)
cg[1][6] = cg[6][1] = 1 / np.sqrt(3)
cg[2][7] = cg[7][2] = 1 / np.sqrt(3)

# sigma- transitions
cg[0][5] = cg[5][0] = -1 / np.sqrt(3)
cg[2][5] = cg[5][2] = -1 / np.sqrt(3)
cg[3][4] = cg[4][3] = 1 / np.sqrt(3)
cg[3][6] = cg[6][3] = -1 / np.sqrt(3)

# mF factors
mF = np.zeros(8)
mF[1] = mF[5] = -1
mF[3] = mF[7] = 1


# noinspection PyPep8Naming
class EightLevelSystem:
    linewidth = 19.7 * 10 ** 6  # Hz

    # hyperfine splitting
    s_splitting = 12642.812118466  # MHz
    p_splitting = 2105  # MHz

    quadratic_shift = 310.8  # B**2 Hz (B in Gauss)

    def __init__(self, sat=0.0, delta=0.0, polarization=(1, 0, 0), B=0.0):
        """Model of the 2S1/2-2P1/2 transition in 171Yb+ as an eight-level system.

        :param delta: Laser detuning from resonance in MHz.
        :param sat: Laser intensity expressed as a multiple of the saturation
                    intensity.
        :param polarization: Laser polarization as a 3-tuple
                             (pi, sigma+, sigma-).
        :param B: Magnetic field in Gauss.
        """
        self._polarization = (1, 0, 0)

        self.sat = sat
        self.delta = delta
        self.polarization = polarization
        self.B = B

        self.basis = [qutip.states.basis(8, i) for i in range(8)]

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
                       * self.basis[4] * self.basis[4].dag()]
        laser_field += [2 * np.pi * (-self.delta + self.p_splitting) * 10 ** 6
                        * self.basis[i] * self.basis[i].dag()
                        for i in (5, 6, 7)]
        laser_field += [2 * np.pi * -self.s_splitting * 10 ** 6
                        * self.basis[0] * self.basis[0].dag()]

        magnetic_field = [2 * np.pi * mF[i] * 1.4 * 10 ** 6 * self.B
                          * self.basis[i] * self.basis[i].dag()
                          for i in (1, 3)]
        magnetic_field += [2 * np.pi * mF[i] * 0.47 * 10 ** 6 * self.B
                           * self.basis[i] * self.basis[i].dag()
                           for i in (5, 7)]
        magnetic_field += [2 * np.pi * self.quadratic_shift * self.B ** 2
                           * self.basis[0] * self.basis[0].dag()]

        off_diagonal_elements = [self.omega[i][j] / 2 * cg[i][j]
                                 * self.basis[i] * self.basis[j].dag()
                                 for i, j in itertools.product(range(8),
                                                               range(8))]

        H = sum(laser_field) + sum(magnetic_field) + sum(off_diagonal_elements)

        return H

    @property
    def decay(self):
        """Decay terms prepared for use in `qutip.mesolve`."""
        return list(itertools.chain(*self.raw_decay))

    @property
    def raw_decay(self):
        """All decay terms."""
        decay = [[] for _ in range(8)]
        decay[4] = [np.sqrt(2 * np.pi * self.linewidth) * cg[i][4]
                    * self.basis[i] * self.basis[4].dag()
                    for i in range(4)]
        decay[5] = [np.sqrt(2 * np.pi * self.linewidth) * cg[i][5]
                    * self.basis[i] * self.basis[5].dag()
                    for i in range(4)]
        decay[6] = [np.sqrt(2 * np.pi * self.linewidth) * cg[i][6]
                    * self.basis[i] * self.basis[6].dag()
                    for i in range(4)]
        decay[7] = [np.sqrt(2 * np.pi * self.linewidth) * cg[i][7]
                    * self.basis[i] * self.basis[7].dag()
                    for i in range(4)]

        return decay

    @property
    def omega(self):
        """Rabi frequencies."""
        pi, sigma_plus, sigma_minus = self.polarization
        _omega = 2 * np.pi * self.linewidth * np.sqrt(self.sat / 2)

        omega = np.zeros((8, 8))
        omega[0][7] = omega[7][0] = \
            omega[1][4] = omega[4][1] = \
            omega[1][6] = omega[6][1] = \
            omega[2][7] = omega[7][2] = _omega * sigma_plus

        omega[0][5] = omega[5][0] = \
            omega[2][5] = omega[5][2] = \
            omega[3][4] = omega[4][3] = \
            omega[3][6] = omega[6][3] = _omega * sigma_minus

        omega[0][6] = omega[6][0] = \
            omega[1][5] = omega[5][1] = \
            omega[2][4] = omega[4][2] = \
            omega[3][7] = omega[7][3] = _omega * pi

        return omega
