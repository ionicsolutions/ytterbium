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

__all__ = ["mesolve", "vary"]

import itertools
from multiprocessing import Pool, cpu_count

import numpy as np
import qutip


def mesolve(hamiltonians, rho0, tlist, c_ops=[], e_ops=[],
            args={}, options=None, progress_bar=None, _safe_mode=True):
    """Wrapper for parallel evaluation of `qutip.mesolve`.

    Instead of a single Hamiltonian, a list of Hamiltonians to be
    evaluated is passed to the method.

    All other arguments are identical to `qutip.mesolve`.
    """
    worklist = [(H, rho0, tlist, c_ops, e_ops,
                 args, options, None, _safe_mode) for H in hamiltonians]
    with Pool(processes=cpu_count()) as p:
        results = p.starmap(qutip.mesolve, worklist)
    return results


def vary(system, **kwargs):
    """Generate a list of Hamiltonians for use in `parallelize.mesolve`.

    The parameters and their values are provided as keyword arguments:

    .. code-block: python

       hamiltonians, parameters = vary(TwoLevelSystem(),
                                       delta=np.linspace(-10.0, 10.0, num=30),
                                       sat=[0.1, 1.0])

    The function returns two values:
    - *hamiltonians* is the list of Hamiltonians which can be passed to
       `parallelize.mesolve`.
    - *system_parameters* is a list of the same length where each entry
       contains the parameter values of the Hamiltonian.

    The parameters are processed in the order in which they are passed
    to the function, and Hamiltonians for all possible combinations are
    generated.
    """
    if not kwargs:
        raise ValueError("No parameter to vary.")
    else:
        variable_parameters = [(parameter, range_)
                               for parameter, range_ in kwargs.items()]

    # check the existence and record the initial value of all parameters
    initial = {}
    for parameter, range_ in variable_parameters:
        try:
            initial[parameter] = getattr(system, parameter)
        except AttributeError:
            raise AttributeError("System has no parameter '%s'." % parameter)

    # generate the parameter space
    expanded_parameters = [itertools.product([parameter], range_)
                           for parameter, range_ in variable_parameters]
    parameter_space = itertools.product(*expanded_parameters)

    hamiltonians = []
    parameters = []

    for parameter_set in parameter_space:
        for parameter, value in parameter_set:
            try:
                setattr(system, parameter, value)
            except AttributeError:
                raise AttributeError(
                    "Parameter '%s' cannot be set." % parameter)
            
        hamiltonians.append(system.H)
        parameters.append([value for parameter, value in parameter_set])

    # verify that the generated Hamiltonians differ
    identical = [np.isclose(hamiltonians[0].full(), H.full())
                 for H in hamiltonians[1:]]
    if np.all(identical):
        raise ValueError("All generated Hamiltonians are identical.")

    # reset system to initial parameter values to avoid confusing behavior
    for parameter, value in initial.items():
        setattr(system, parameter, value)

    return hamiltonians, parameters
