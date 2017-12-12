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

__all__ = ["TenLevelSystem"]

import itertools

import numpy as np
import qutip

from . import FourLevelSystem, SixLevelSystem

# inter-system Clebsch-Gordan coefficients
cg = np.zeros((10, 10))
cg[1][6] = cg[6][1] = cg[3][7] = cg[7][3] = 1 / np.sqrt(3)
cg[1][7] = cg[7][1] = cg[3][6] = cg[6][3] = np.sqrt(2 / 3)
cg[0][8] = cg[8][0] = cg[5][9] = cg[9][5] = 1 / np.sqrt(2)
cg[2][8] = cg[8][2] = cg[4][9] = cg[9][4] = 1 / np.sqrt(3)
cg[2][9] = cg[9][2] = cg[4][8] = cg[8][4] = 1 / np.sqrt(6)


# noinspection PyPep8Naming
class TenLevelSystem:
    def __init__(self, delta_SP=0.0, sat_SP=0.0, polarization_SP=(1, 0, 0),
                 delta_D=0.0, sat_D=0.0, polarization_D=(1, 0, 0), B=0.0):
        """Combined model of both 2S1/2-2P1/2 and 2D3/2-3D[3/2]1/2
        transitions in 174Yb+ as a ten-level system.

        Based on the model described in H. Meyer (2014), the level numbering
        is identical.

        :param delta_(SP/D): Laser detuning from resonance in MHz.
        :param sat_(SP/D): Laser intensity expressed as a multiple
                           of the saturation intensity.
        :param polarization_(SP/D): Laser polarization as a
                                    3-tuple (pi, sigma+, sigma-).
        :param B: Magnetic field in Gauss.
        """

        self.delta_SP = delta_SP
        self.sat_SP = sat_SP
        self.polarization_SP = polarization_SP

        self.delta_D = delta_D
        self.sat_D = sat_D
        self.polarization_D = polarization_D

        self.B = B

        self.basis = [qutip.states.basis(10, i) for i in range(10)]

    @property
    def FLS(self):
        """2S1/2-2P1/2 transition as a four-level system."""
        FLS = FourLevelSystem(delta=self.delta_SP,
                              sat=self.sat_SP,
                              polarization=self.polarization_SP,
                              B=self.B)
        FLS.basis = self.basis[6:]
        return FLS

    @property
    def SLS(self):
        """2D3/2-3D[3/2]1/2 as a six-level system."""
        SLS = SixLevelSystem(delta=self.delta_D,
                             sat=self.sat_D,
                             polarization=self.polarization_D,
                             B=self.B)
        SLS.basis = self.basis[:6]
        return SLS

    @property
    def H(self):
        """Full Hamiltonian of the system."""
        return self.SLS.H + self.FLS.H

    @property
    def decay(self):
        """Decay terms prepared for use in `qutip.mesolve`."""
        return list(itertools.chain(*self.raw_decay))

    @property
    def raw_decay(self):
        """All decay terms."""
        # Inter-system decay terms
        _decay = [[] for _ in range(10)]
        _decay[1] = [np.sqrt(2 * np.pi * self.SLS.linewidth) * cg[i][1]
                     * self.basis[i] * self.basis[1].dag()
                     for i in (6, 7)]
        _decay[3] = [np.sqrt(2 * np.pi * self.SLS.linewidth) * cg[i][3]
                     * self.basis[i] * self.basis[3].dag()
                     for i in (6, 7)]
        _decay[8] = [np.sqrt(2 * np.pi * self.FLS.linewidth) * cg[i][8]
                     * self.basis[i] * self.basis[8].dag()
                     for i in (0, 2, 4, 5)]
        _decay[9] = [np.sqrt(2 * np.pi * self.FLS.linewidth) * cg[i][9]
                     * self.basis[i] * self.basis[9].dag()
                     for i in (0, 2, 4, 5)]

        # decay from the meta-stable 2D3/2 state to 2S1/2 is neglected
        # in this model (lifetime > 50 ms)

        # Combined decay terms

        # branching ratio 2D[3/2]1/2 to 2S1/2 as quoted in
        # H. Meyer et al. (2012)
        decay = [[] for _ in range(10)]
        decay[1] = [np.sqrt(0.982) * term for term in _decay[1]] + \
                   [np.sqrt(0.018) * term for term in self.SLS.raw_decay[1]]

        decay[3] = [np.sqrt(0.982) * term for term in _decay[3]] + \
                   [np.sqrt(0.018) * term for term in self.SLS.raw_decay[3]]

        # branching ratio 2P1/2 to 2D3/2 from S. Olmschenk et al. (2007)
        decay[8] = [np.sqrt(0.00501) * term for term in _decay[8]] + \
                   [np.sqrt(0.99499) * term for term in self.FLS.raw_decay[2]]

        decay[9] = [np.sqrt(0.00501) * term for term in _decay[9]] + \
                   [np.sqrt(0.99499) * term for term in self.FLS.raw_decay[3]]

        return decay
