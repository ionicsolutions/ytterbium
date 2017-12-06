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
import os
from multiprocessing import Pool

import qutip


def mesolve_worker(arguments):
    H, psi0, times, c_ops, e_ops = arguments
    return qutip.mesolve(H, psi0, times, c_ops, e_ops)


def mesolve(list_of_H, psi0, times, c_ops, e_ops):
    """Wrapper for parallel evaluation of `qutip.mesolve`.

    Instead of a single Hamiltonian, a list of Hamiltonians to be
    evaluated is passed to the method.

    All other arguments are identical to `qutip.mesolve`.
    """
    worklist = [(H, psi0, times, c_ops, e_ops) for H in list_of_H]
    with Pool(processes=os.cpu_count()) as p:
        it = p.imap(mesolve_worker, worklist)
        results = [result for result in it]
    return results
