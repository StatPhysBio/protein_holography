#
# hologram.py -- Michael Pun -- 28 January 2021
#
# This module contains the functions used to project point clouds 
# characterized by arrays of spherical coordinates into spherical
# coefficients. The functions also allow for a channel dimensions
# to encode any information that is not spatial in the data.
#

# import modules necessary for this code
import scipy as sp
import numpy as np
from scipy import special
import os

#----------------------
# Holographic functions
#----------------------

# spherical Hankel functions of the first kind for use in the 
# hologram projections (i.e. Bessel function projections)
def spherical_hn1(n,z,derivative=False):
    return (sp.special.spherical_jn(n,z,derivative=False) + 
            1j*sp.special.spherical_yn(n,z,derivative=False))

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

    # compute factors in the spherical wave expansion
    kTimesRlesser = np.array(np.real(k)*r_lesser)
    kTimesRgreater = np.array(np.real(k)*r_greater)
    j = sp.special.spherical_jn(l,kTimesRlesser)
    h = spherical_hn1(l,kTimesRgreater)
    y = np.conj(sp.special.sph_harm(m,l,p,t))

    # assemble coefficients
    coeffs = np.exp(1j*np.array(r)) * k * 1j * j * h * y
    
    # sum over the points in the point cloud and return
    return np.sum(coeffs,axis=-1)
    
# Using the previous function, this method returns the coefficients 
# of a given l in an array of dimension (ch_num X 2l+1)
def hologram_coeff_l(r,t,p,r_h,k,l):
    if (np.array(r).shape != np.array(t).shape or
        np.array(p).shape != np.array(t).shape):
        print('Error: input arrays do not have same shape')
        return None

    fourier_coeff_l = []
    for m in range(0,2*l+1):
        fourier_coeff_l.append(hologram_coeff_lm(r,t,p,r_h,k,l,m-l))
    return np.array(fourier_coeff_l)

#-------------------------------
# Functions for delta-projection
#-------------------------------

# A cartesian to spherical projection with a delta function 
# transform in the radial component.
def delta(r,t,p,l):
    if (np.array(r).shape != np.array(t).shape or
        np.array(p).shape != np.array(t).shape):
        print('Error: input arrays do not have same shape')
        return None

    fourier_coeff_l = []
    for m in range(-l,l+1):
        fourier_coeff_l.append(np.sum(sp.special.sph_harm(m,l,p,t),axis=-1))
    return np.array(fourier_coeff_l)


#---------------------------------
# Functions for Zernike projection
#---------------------------------

# Inputs r,t,p in the shapes (ch X n) and give coefficients
# in output shape (ch)
def zernike_coeff_lm(r,t,p,n,r_max,l,m,weights):
    # zernike coefficient is zero if n-l odd
    n = int(np.real(n))
    if n < l:
        zeros = np.zeros(shape=r.shape,dtype=complex)
        return 0. + 0j

    if (n-l) % 2 == 1:
        zeros = np.zeros(shape=r.shape,dtype=complex)
        return 0.+0j

    # check input dimensions of arrays
    if (np.array(r).shape != np.array(t).shape or
        np.array(t).shape != np.array(p).shape):
        print('Error: input arrays do not have same shape')

    # dimension of the Zernike polynomial
    D = 3.
    # constituent terms in the polynomial
    A = np.power(-1,(n-l)/2.) 
    B = np.sqrt(2.*n + D)
    C = sp.special.binom(int((n+l+D)/2. - 1),
                         int((n-l)/2.))
    E = sp.special.hyp2f1(-(n-l)/2.,
                           (n+l+D)/2.,
                           l+D/2.,
                           np.array(r)/r_max*np.array(r)/r_max)
    F = np.power(np.array(r)/r_max,l)

    # spherical harmonic component
    y = np.conj(sp.special.sph_harm(m,l,p,t))
    if True in np.isinf(E):
        print('Error: E is inf')
        print('E',E)
        print('n',n,
              'l',l,
              'D',D,
              'r',np.array(r),
              'rmax',r_max)

    # assemble coefficients
    coeffs = A * B * C * E * F * y

    return np.sum(weights*coeffs,axis=-1)
    
# Inputs r,t,p in the shapes (ch X n) and give coefficients
# in output shape (ch) 
def zernike_coeff_l(r,t,p,n,r_max,l,weights=None):
    if weights is None:
        weights = np.ones(shape=r.shape[-1])
    if (np.array(r).shape != np.array(t).shape or
        np.array(p).shape != np.array(t).shape):
        print('Error: input arrays do not have same shape')
        return None

    fourier_coeff_l = []
    for m in range(0,2*l+1):
        fourier_coeff_l.append(zernike_coeff_lm(r,t,p,n,r_max,l,m-l,weights))
    return np.array(fourier_coeff_l)


#-----------------------------
# Saving and loading functions
#-----------------------------

# function to load premade holograms
# needs to be updated as naming convention changes for holograms
# maybe also just remove completely since loading is straightforward
def load_holograms(k_,d,cutoff_l,examples_per_aa):
    file_workspace = '/gscratch/spe/mpun/holograms'
    os.chdir(file_workspace)

    training_f_coeffs_real = np.load(
        'train_hgram_real_example_examplesPerAA=' + 
        str(examples_per_aa) + '_k=' + str(k_) + '_d=' + 
        str(d) + '_l=' + str(cutoff_l) + '.npy',
        allow_pickle=True,encoding='latin1')[()]
    training_f_coeffs_imag = np.load(
        'train_hgram_imag_example_examplesPerAA=' + 
        str(examples_per_aa) + '_k='+str(k_) + '_d=' + 
        str(d) + '_l=' + str(cutoff_l) + '.npy',
        allow_pickle=True,encoding='latin1')[()]   
    training_labels = np.load(
        'train_labels_examplesPerAA=' + 
        str(examples_per_aa) + '_k=' + str(k_) + '_d=' +
        str(d) + '_l=' + str(cutoff_l) + '.npy',
        allow_pickle=True,encoding='latin1')

    return (training_f_coeffs_real,training_f_coeffs_imag,training_labels)

#
# function to save holograms
#
def save(holograms,filename,directory):
    np.save(directory+'/'+filename,holograms,allow_pickle=True)




