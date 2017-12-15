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

__all__ = ["TwoLevelSystem"]

import itertools

import numpy as np
import qutip

# Clebsch-Gordan coefficients
cg = np.zeros((2, 2))
cg[0][1] = cg[1][0] = 1


# noinspection PyPep8Naming
class TwoLevelSystem:
    linewidth = 19.6 * 10 ** 6  # Hz

    def __init__(self, delta, sat):
        """Model of the 2S1/2-2P1/2 transition in 174Yb+ as a two-level system.

        :param delta: Laser detuning from resonance in MHz.
        :param sat: Laser intensity expressed as a multiple of the
                    saturation intensity.
        """
        self.delta = delta
        self.sat = sat

        self.basis = [qutip.states.basis(2, i) for i in range(2)]

    @property
    def H(self):
        """Full Hamiltonian of the system."""
        laser_field = [2 * np.pi * -self.delta * 10 ** 6
                       * self.basis[1] * self.basis[1].dag()]

        off_diagonal_elements = [self.omega[i][j] / 2 * cg[i][j]
                                 * self.basis[i] * self.basis[j].dag()
                                 for i, j in itertools.product(range(2),
                                                               range(2))]

        H = sum(laser_field) + sum(off_diagonal_elements)

        return H

    @property
    def decay(self):
        """Decay terms prepared for use in `qutip.mesolve`."""
        return list(itertools.chain(*self.raw_decay))

    @property
    def raw_decay(self):
        """All decay terms."""
        decay = [[] for _ in range(2)]
        decay[1] = [np.sqrt(2 * np.pi * self.linewidth) * cg[0][1]
                    * self.basis[0] * self.basis[1].dag()]

        return decay

    @property
    def omega(self):
        """Rabi frequencies."""
        omega = np.zeros((2, 2))
        omega[1][0] = \
            omega[0][1] = 2 * np.pi * self.linewidth * np.sqrt(self.sat / 2)

        return omega
