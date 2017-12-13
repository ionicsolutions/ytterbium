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

__all__ = ["FourLevelSystem"]

import itertools

import numpy as np
import qutip

from ...polarization import normalize

# Clebsch-Gordan coefficients
cg = np.zeros((4, 4))
cg[0][2] = cg[2][0] = -1 / np.sqrt(2)
cg[0][3] = cg[3][0] = 1 / np.sqrt(2)

cg[1][2] = cg[2][1] = 1 / np.sqrt(2)
cg[1][3] = cg[3][1] = 1 / np.sqrt(2)


# mF factors
mF = np.zeros(4)
mF[0] = mF[3] = 0
mF[1] = mF[2] = -1


# noinspection PyPep8Naming
class FourLevelSystem:
    linewidth = 19.7 * 10 ** 6  # Hz

    def __init__(self, sat_M=0.0, polarization_M=(1, 0, 0), sat_S=0.0, polarization_S=(0, 1, 0),
                 delta=0.0, detuning=0.0, B=0.0):
        """Model of the 2S1/2-2P1/2 transition in 171Yb+ as a four-level system
        to simulate Raman transitions from (S, F=0, mF=0) to (S, F=1, mF=+1) via
        (P, F=1, mF=0).

        :param sat_pi: Intensity of the pi-polarized laser in multiples of its
                       on-resonance, two-level saturation intensity.
        :param sat_sig: Intensity of the sigma-polarized laser in multiples of its
                        on-resonance, two-level saturation intensity.
        :param delta: Frequency difference between the two lasers.
        :param detuning: Detuning from the excited state in GHz.
        :param B: Magnetic field in Gauss.
        """
        self._polarization_M = (1, 0, 0)
        self._polarization_S = (0, 1, 0)

        self.sat_M = sat_M
        self.polarization_M = polarization_M
        self.sat_S = sat_S
        self.polarization_S = polarization_S

        self.delta = delta
        self.detuning = detuning
        self.B = B

        self.basis = [qutip.states.basis(4, i) for i in range(4)]

    @property
    def polarization_M(self):
        return self._polarization_M

    @polarization_M.setter
    def polarization_M(self, vector):
        self._polarization_M = normalize(vector)

    @property
    def polarization_S(self):
        return self._polarization_S

    @polarization_S.setter
    def polarization_S(self, vector):
        self._polarization_S = normalize(vector)

    @property
    def H(self):
        """Full Hamiltonian of the system."""
        laser_field = [2 * np.pi * self.detuning * 10 ** 9 * self.basis[i] * self.basis[i].dag()
                       for i in (0, 1)]
        laser_field.append(2 * np.pi * self.delta * 10 ** 6 * self.basis[0] * self.basis[0].dag())

        magnetic_field = [2 * np.pi * mF[i] * 1.4 * 10 ** 6 * self.B * self.basis[i] * self.basis[i].dag()
                          for i in range(4)]

        off_diagonal_elements = [self.omega[i][j] / 2 * self.basis[i] * self.basis[j].dag() * cg[i][j]
                                 for i, j in itertools.product(range(4), range(4))]

        return sum(laser_field) + sum(magnetic_field) + sum(off_diagonal_elements)

    @property
    def decay(self):
        """Decay terms prepared for use in `qutip.mesolve`."""
        return list(itertools.chain(*self.raw_decay))

    @property
    def raw_decay(self):
        """All decay terms."""
        decay = [[] for _ in range(5)]
        return decay

    @property
    def omega(self):
        pi_M, sigma_plus_M, sigma_minus_M = self.polarization_M
        pi_S, sigma_plus_S, sigma_minus_S = self.polarization_S

        _omega_M = 2 * np.pi * self.linewidth * np.sqrt(self.sat_M / 2)
        _omega_S = 2 * np.pi * self.linewidth * np.sqrt(self.sat_S / 2)

        omega = np.zeros((4, 4))
        omega[0][2] = omega[2][0] = _omega_M * sigma_minus_M
        omega[0][3] = omega[3][0] = _omega_M * pi_M
        omega[1][2] = omega[2][1] = _omega_S * pi_S
        omega[1][3] = omega[3][1] = _omega_S * sigma_plus_S
        return omega
