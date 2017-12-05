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
import numpy as np


class System:

    def __init__(self):
        """Common base class for multi-level systems."""
        self._polarization = (1, 0, 0)

    @property
    def polarization(self):
        """Normalized polarization vector (pi, sigma+, sigma-)."""
        return self._polarization

    @polarization.setter
    def polarization(self, vector):
        length = np.sqrt(np.sum([vector[i] ** 2 for i in range(3)]))
        if length <= 0.0:
            raise ValueError("Polarization vector has to be of non-zero length.")
        self._polarization = tuple(vector[i] / length for i in range(3))
