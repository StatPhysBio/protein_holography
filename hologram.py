#
# hologram.py file for holographic machine learning program
#


import scipy as sp
import numpy as np
from scipy import special

# spherical Hankel functions of the first kind
def spherical_hn1(n,z,derivative=False):
    """ Spherical Hankel Function of the First Kind """
    return sp.special.spherical_jn(n,z,derivative=False) + 1j*sp.special.spherical_yn(n,z,derivative=False)

# given a set of points parametrized by r,t,p, this function returns the
# coefficient of Ylm in the spherical harmonic expansion of the electric
# field caused by these points at the surface of a sphere of radius r_h
def hologram_coeff_lm(r,t,p,r_h,k,l,m):

    # check input dimensions of arrays
    if (np.array(r).shape != np.array(t).shape or
        np.array(t).shape != np.array(p).shape):
        print('Error: input arrays do not have same shape')

    # determine r_lesser and r_greater
    r_lesser = np.minimum(r,r_h)
    r_greater = np.maximum(r,r_h)
#    print(r_lesser)
#    print(r_greater)
    # compute factors in the spherical wave expansion
    kTimesRlesser = np.array(k*r_lesser,dtype=float)
    kTimesRgreater = np.array(k*r_greater,dtype=float)
    j = sp.special.spherical_jn(l,kTimesRlesser)
    h = spherical_hn1(l,kTimesRgreater)
    y = np.conj(sp.special.sph_harm(m,l,p,t))

    # assemble coefficients
    coeffs = np.exp(1j*np.array(r)) * k * 1j * j * h * y

    return np.sum(coeffs,axis=-1)
    
def hologram_coeff_l(r,t,p,r_h,k,l):
    if (np.array(r).shape != np.array(t).shape or
        np.array(p).shape != np.array(t).shape):
        print('Error: input arrays do not have same shape')
        return None

    fourier_coeff_l = []
    for m in range(0,2*l+1):
        fourier_coeff_l.append(hologram_coeff_lm(r,t,p,r_h,k,l,m-l))
    return np.array(fourier_coeff_l)
