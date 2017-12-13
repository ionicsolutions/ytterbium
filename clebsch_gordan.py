import numpy as np
import qutip
from scipy.misc import factorial
import itertools

from sympy import N
from sympy.physics.wigner import wigner_3j, wigner_6j

"""Generate the Clebsch-Gordan coefficients for S-P in Ytterbium 171."""

# nuclear spin
I = 1 / 2

# electron spin
S = 1 / 2

# 2 S 1/2
Jg = 1 / 2

# 2 P 1/2
Je = 1 / 2


def eq35(F, mF, Fprime, mFprime):
    q = mF - mFprime
    return np.power(-1.0, Fprime - 1 + mF) * np.sqrt(2*F + 1)\
           * N(wigner_3j(Fprime, 1, F,
                         mFprime, q, -mF))


def eq35mc(F, mF, Fprime, mFprime):
    q = mF - mFprime
    return np.power(-1.0, F - 1 + mFprime) * np.sqrt(2*Fprime + 1)\
           * N(wigner_3j(F, 1, Fprime,
                         mF, -q, -mFprime))


def eq36mc(F, Fprime, J, Jprime):
    return np.power(-1.0, F + Jprime + 1 + I)\
           * np.sqrt((2*F + 1)*(2*Jprime + 1)) \
           * N(wigner_6j(Jprime, Fprime, I,
                         F, J, 1))


def eq36(F, Fprime, J, Jprime):
    return np.power(-1.0, Fprime + J + 1 + I)\
           * np.sqrt((2*Fprime + 1)*(2*J + 1)) \
           * N(wigner_6j(J, Jprime, 1,
                         Fprime, F, I))


def factor(F, mF, Fprime, mFprime, J, Jprime):
    return eq35(F, mF, Fprime, mFprime) * eq36(F, Fprime, J, Jprime)


def factormc(F, mF, Fprime, mFprime, J, Jprime):
    return eq35mc(F, mF, Fprime, mFprime) * eq36mc(F, Fprime, J, Jprime)



transitions = [
    (0, 0, 1, 0, Jg, Je),
    (0, 0, 1, 1, Jg, Je),
    (0, 0, 1, -1, Jg, Je),

    (1, 0, 0, 0, Jg, Je),
    (1, 0, 1, 1, Jg, Je),
    (1, 0, 1, -1, Jg, Je),

    (1, 1, 1, 0, Jg, Je),
    (1, 1, 0, 0, Jg, Je),
    (1, 1, 1, 1, Jg, Je),

    (1, -1, 1, 0, Jg, Je),
    (1, -1, 0, 0, Jg, Je),
    (1, -1, 1, -1, Jg, Je)
]

for transition in transitions:
    print(transition[0:4], factor(*transition) * np.sqrt(3), factormc(*transition) * np.sqrt(3))


def clebsch_gordan(Fg, mFg, Fe, mFe, q):
    exponent = (2 * Fe + I + Jg + Je + Lg + S + mFg + 1)
    root = np.sqrt((2 * Fg + 1) * (2 * Fe + 1) * (2 * Jg + 1) * (2 * Je + 1) * (2 * Lg + 1))

    w3 = wigner_3j(Fe, 1, Fg, mFe, -q, -mFg)
    w61 = wigner_6j(Jg, Je, 1, Fe, Fg, I, prec=32)
    w62 = wigner_6j(Lg, Le, 1, Je, Jg, S, prec=32)

    return (-1) ** exponent * root * N(w3) * w61 * w62


def clebsch_g(Fg, mFg, Fe, mFe, q):
    return qutip.utilities.clebsch(j1=Fe, j2=1, j3=Fg, m1=mFe, m2=-q, m3=-mFg)




