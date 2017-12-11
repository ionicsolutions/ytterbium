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

__all__ = ["normalize"]

import numpy as np


def normalize(vector):
    """Normalize polarization *vector*."""
    if len(vector) != 3:
        raise ValueError(
            "Polarization vector must have exactly 3 components.")
    _vector = np.array(vector)
    length = np.sqrt(np.sum(np.power(_vector, 2)))
    if length <= 0.0 or not np.all(np.isreal(_vector)):
        raise ValueError(
            "Polarization vector has to be real and of non-zero length.")
    return tuple(_vector/length)
