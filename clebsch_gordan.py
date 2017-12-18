"""Calculate Clebsch-Gordan coefficients symbolically.

This small script follows Steck, Cesium Numbers v1.6 and the identical
Steck, Rubidium Numbers v . See one of these documents for details.
"""
import itertools

import sympy
from sympy.physics.wigner import wigner_3j, wigner_6j

# nuclear and electron spin
sI, sS = sympy.symbols("I S")
# ground and excited state angular momentum
sJg, sJe = sympy.symbols("Jg Je")

# Defaults for 171Yb+ S-P transition
Jg = sympy.S("1/2")
Je = sympy.S("1/2")
I = sympy.S("1/2")


# From Steck, Cesium Numbers v1.6
def eq35(F, mF, Fprime, mFprime):
    sF, smF, sFprime, smFprime = sympy.symbols("F mF Fprime mFprime")
    q = mF - mFprime
    expr = (-1)**(sFprime - 1 + smF) * sympy.sqrt(2 * sF + 1) \
           * wigner_3j(Fprime, 1, F, mFprime, q, -mF)
    return expr.subs({sF: F, smF: mF, sFprime: Fprime, smFprime: mFprime})


def eq36(F, Fprime, J, Jprime, I):
    sF, sFprime, sJ, sJprime = sympy.symbols("F Fprime J Jprime")
    expr = (-1)**(sFprime + sJ + 1 + sI) \
           * sympy.sqrt((2 * sFprime + 1) * (2 * sJ + 1)) \
           * wigner_6j(J, Jprime, 1, Fprime, F, I)
    return expr.subs({sF: F, sFprime: Fprime, sJ: J, sJprime: Jprime, sI: I})


def factor(F, mF, Fprime, mFprime, J=Jg, Jprime=Je, I=I):
    return eq35(F, mF, Fprime, mFprime) * eq36(F, Fprime, J, Jprime, I)


def generate(F_min, F_max):
    Fs = [F_min]
    while Fs[-1] < F_max:
        Fs.append(Fs[-1] + 1)
    return Fs


if __name__ == "__main__":
    F_min_g, F_max_g = abs(Jg - I), abs(Jg + I)
    F_min_e, F_max_e = abs(Je - I), abs(Je + I)

    F_gs = generate(F_min_g, F_max_g)
    F_es = generate(F_min_e, F_max_e)

    print("Hyperfine states:")
    print("Ground: %s" % F_gs)
    print("Excited: %s" % F_es)
    print("")

    for F_g in F_gs:
        for F_e in F_es:
            if abs(F_e - F_g) <= 1:
                print("F = %s to F' = %s" % (F_g, F_e))
                for mF_g, mF_e in itertools.product(generate(-F_g, F_g),
                                                    generate(-F_e, F_e)):
                    if abs(mF_e - mF_g) <= 1:
                        cg = factor(F_g, mF_g, F_e, mF_e)
                        # trick to display coefficient in usual form
                        sign = "-" if cg < 0 else ""
                        print_cg = "%ssqrt(%s)" % (sign, cg**2)
                        print("mF=%s to mF'=%s: %s" % (mF_g, mF_e, print_cg))
                print("")
