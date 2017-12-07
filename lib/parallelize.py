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
    """Generate a list of Hamiltonians for use in `parallelize.mesolve`.

    If only one parameter is to be varied, the parameter and the values it
    should take on can be provided as a keyword argument:

    >>> vary(TwoLevelSystem, delta=np.linspace(-10.0, 10.0, num=30))

    When several parameters vary, they need to be provided as a list or
    tuple which defines the order of the Hamiltonians:

    >>> vary(TwoLevelSystem, parameters=(("delta", [-100.0, -10.0, 0.0]), ("sat", [0.1, 1.0])))

    The function returns two values:
    The first (*hamiltonians*) is the list of Hamiltonians which can be passed to `parallelize.mesolve`.
    The second (*system_parameters*) is a list of the parameter values used to generate the Hamiltonians:

    >>> hamiltonians, system_parameters = vary(...)
    >>> system_parameters
    [[-100.0, 0.1], [-100.0, 1.0], [-10.0, 0.1], [-10.0, 1.0], [0.0, 0.1], [0.0, 1.0]]
    """
    if not parameters:
        if len(kwargs) > 1:
            raise ValueError(
                "Multiple parameters need to be passed as a list to guarantee a non-ambiguous order.")
        elif len(kwargs) == 0:
            raise ValueError("No parameter list and no single parameter given.")
        else:
            parameters = [(parameter, range_) for parameter, range_ in kwargs.items()]

    initial = {}  # check that all parameters exist and record their initial value
    for parameter, range_ in parameters:
        try:
            initial[parameter] = getattr(system, parameter)
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
                raise AttributeError("Parameter '%s' cannot be set." % parameter)
            else:
                hamiltonians.append(system.H)
        system_parameters.append([value for parameter, value in parameter_set])

    if np.all([np.isclose(hamiltonians[0].full(), H.full()) for H in hamiltonians[1:]]):
        raise ValueError("All generated Hamiltonians are identical.")

    for parameter, value in initial.items():  # reset system to avoid confusing behavior
        setattr(system, parameter, value)

    return hamiltonians, system_parameters
