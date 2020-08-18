#
# Implementation of the Wigner D-matrices via sympy.
#

import sympy
import numpy as np

class Memoize:
    # https://stackoverflow.com/a/1988826/1142217
    def __init__(self, f):
        self.f = f
        self.memo = {}
    def __call__(self, *args):
        if not args in self.memo:
          self.memo[args] = self.f(*args)
        return self.memo[args]

from sympy.physics.quantum.spin import WignerD
def wignerD(ll,mm,nn,aa,bb,gg):
    """
    Computes <j1 m1 j2 m2 | j3 m3>, where all spins are given as double their values.
    """
    # https://docs.sympy.org/latest/modules/physics/quantum/cg.html
    # htt`ps://mattpap.github.io/scipy-2011-tutorial/html/numerics.html
    return WignerD(ll,mm,nn,aa,bb,gg).doit().evalf()

wignerD = Memoize(wignerD)

def wigner_d_matrix(l,a,b,g):
    wigner_d_mat = np.zeros([2*l+1,2*l+1],dtype=complex)
    for m in range(-l,l+1):
        for n in range(-l,l+1):
            wigner_d_mat[m+l,n+l] = wignerD(l,m,n,a,b,g)
    return wigner_d_mat
