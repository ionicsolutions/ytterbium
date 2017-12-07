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
import os
from multiprocessing import Pool

import numpy as np
import qutip


def mesolve(hamiltonians, psi0, times, c_ops, e_ops):
    """Wrapper for parallel evaluation of `qutip.mesolve`.

    Instead of a single Hamiltonian, a list of Hamiltonians to be
    evaluated is passed to the method.

    All other arguments are identical to `qutip.mesolve`.
    """
    worklist = [(H, psi0, times, c_ops, e_ops) for H in hamiltonians]
    with Pool(processes=os.cpu_count()) as p:
        results = p.starmap(qutip.mesolve, worklist)
    return results


def vary(system, parameters=(), **kwargs):
    """Generate a list of Hamiltonians for use in `parallelize.mesolve`."""
    if not parameters:
        if len(kwargs) > 1:
            raise ValueError(
                "Multiple parameters need to be passed as a list to guarantee a non-ambiguous order.")
        elif len(kwargs) == 0:
            raise ValueError("No parameter list and no single parameter given.")
        else:
            parameters = [(parameter, range_) for parameter, range_ in kwargs.items()]

    for parameter, range_ in parameters:
        try:
            _ = getattr(system, parameter)
        except AttributeError:
            raise AttributeError("System has no parameter '%s'." % parameter)

    expanded_parameters = [itertools.product([parameter], range_) for parameter, range_ in parameters]
    parameter_space = itertools.product(*expanded_parameters)

    hamiltonians = []
    system_parameters = []

    for parameter_set in parameter_space:
        for parameter, value in parameter_set:
            try:
                setattr(system, parameter, value)
            except AttributeError:
                raise AttributeError("System has no settable parameter '%s'." % parameter)
            else:
                hamiltonians.append(system.H)
        system_parameters.append([value for parameter, value in parameter_set])

    if np.all([np.isclose(hamiltonians[0].full(), H.full()) for H in hamiltonians[1:]]):
        raise ValueError("All generated Hamiltonians are identical.")

    return hamiltonians, system_parameters
