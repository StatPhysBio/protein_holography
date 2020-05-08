#
# clebsch.py module for protein holographic machine learning
#

from sympy.physics.quantum.cg import CG
import sympy

# this class and function implements the Clebsch Gordan coefficients
# as a function that takes the parameters of a Clebsch Gordan coefficient
# and returns that coefficient
class Memoize:
    # https://stackoverflow.com/a/1988826/1142217
    def __init__(self, f):
        self.f = f
        self.memo = {}
    def __call__(self, *args):
        if not args in self.memo:
            self.memo[args] = self.f(*args)
        return self.memo[args]

def clebsch(j1,m1,j2,m2,j3,m3):
    """
    Computes <j1 m1 j2 m2 | j3 m3>, where all spins are given as double their values.
    """
    # https://docs.sympy.org/latest/modules/physics/quantum/cg.html
    # https://mattpap.github.io/scipy-2011-tutorial/html/numerics.html
    return CG(sympy.S(j1),sympy.S(m1),sympy.S(j2),sympy.S(m2),sympy.S(j3),sympy.S(m3)).doit().evalf()

clebsch = Memoize(clebsch)
