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

__all__ = ["NineLevelSystem"]

import itertools

import numpy as np
import qutip

from ytterbium.Yb171 import EightLevelSystem
from ytterbium.polarization import normalize


# noinspection PyPep8Naming
class NineLevelSystem:

    def __init__(self, detuning=0.0, delta=0.0, B=0.0,
                 sat_M=0.0, polarization_M=(1, 0, 0),
                 sat_S=0.0, polarization_S=(1, 0, 0)):
        """Model of 171Yb+ as a
        nine-level system to simulate Raman transitions driven
        by two lasers labelled *M*aster and *S*lave.

        The model includes the 2S1/2-2P1/2 states as a
        full eight-level system and the 2D3/2 states as a single
        state.

        :param detuning: Detuning of the Master from
        :param delta: Frequency difference between Master and Slave in MHz.
        :param B: Magnetic field in Gauss.
        :param sat_M/S:
        :param polarization_M/S:
        """
        self._polarization_S = (1, 0, 0)

        self.detuning = detuning
        self.delta = delta
        self.B = B

        self.polarization_M = polarization_M
        self.sat_M = sat_M
        self.polarization_S = polarization_S
        self.sat_S = sat_S

        self.basis = [qutip.states.basis(9, i) for i in range(9)]

    @property
    def polarization_S(self):
        return self._polarization_S

    @polarization_S.setter
    def polarization_S(self, vector):
        self._polarization_S = normalize(vector)

    @property
    def ELS(self):
        ELS = EightLevelSystem(delta=self.detuning,
                               sat=self.sat_M,
                               polarization=self.polarization_M,
                               B=self.B)
        ELS.basis = self.basis[:9]
        return ELS

    @property
    def H(self):
        laser_field = [2 * np.pi
                       * (self.detuning * 10 ** 9 + self.delta * 10 ** 6)
                       * self.basis[i] * self.basis[i].dag()
                       for i in (0, 2)]

        off_diagonal_elements = []

        H = self.ELS.H + sum(laser_field) + sum(off_diagonal_elements)

        return H

    @property
    def decay(self):
        """Decay terms prepared for use in `qutip.mesolve`."""
        return list(itertools.chain(*self.raw_decay))

    @property
    def raw_decay(self):
        # Decay to 2D3/2
        _decay = [np.sqrt(2 * np.pi * self.ELS.linewidth)
                  * self.basis[8] * self.basis[i].dag()
                  for i in (4, 5, 6, 7)]

        # Combined decay terms

        decay = [np.sqrt(0.005) * term for term in _decay] + \
                [np.sqrt(0.995) * term for term in self.ELS.decay]

        return decay

    @property
    def omega(self):
        return None
