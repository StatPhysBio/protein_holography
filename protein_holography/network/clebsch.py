#
# clebsch.py module for protein holographic machine learning
#

import numpy as np
import sympy
from sympy.physics.quantum.cg import CG
import tensorflow as tf

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

def load_clebsch(cg_file,L_MAX):
    
    # load clebsch gordan coefficients
    cg_matrices = np.load(cg_file, allow_pickle=True).item()

    tf_cg_matrices = {}
    for l in range(L_MAX + 1):
        for l1 in range(L_MAX + 1):
            for l2 in range(0,l1+1):
                tf_cg_matrices[(l,l1,l2)] = tf.convert_to_tensor(cg_matrices[(l,l1,l2)],
                                                                 dtype=tf.complex64)
    return tf_cg_matrices
