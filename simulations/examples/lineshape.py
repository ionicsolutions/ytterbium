import numpy as np
import matplotlib.pyplot as plt

import ytterbium as yb
from ytterbium.Yb174 import FourLevelSystem

# initialize the S-P transition in 174Yb+ as a four-level system
FLS = FourLevelSystem(sat=0.5)

# to measure the lineshape, we drive the system at different laser detunings,
# which are defined in MHz across ytterbium
laser_detuning = np.linspace(-40.0, 40.0, num=41)

# for each detuning, we generate a Hamiltonian
hamiltonians, _ = yb.vary(FLS, delta=laser_detuning)

# initially, all population is in the ground state
psi0 = 1/np.sqrt(2) * (FLS.basis[0] + FLS.basis[1])

# we prepare population operators |i><i| for all states
population = [state * state.dag() for state in FLS.basis]

# to use Python's multiprocessing module for parallel evaluation,
# the call to yb.mesolve() must not be executed unless the script
# is invoked directly
if __name__ == "__main__":
    # solve the system for each Hamiltonian for 15 us
    results = yb.mesolve(hamiltonians, psi0,
                         np.linspace(0, 15*10**-6, num=500),
                         FLS.decay, population)

    # extract the steady-state excited-state population from the results
    excited_state_population = [result.expect[2][-1] + result.expect[3][-1]
                                for result in results]

    plt.plot(laser_detuning, excited_state_population, "o")
    plt.xlabel("Laser detuning from resonance [MHz]")
    plt.ylabel("Total excited-state population")
    plt.show()
