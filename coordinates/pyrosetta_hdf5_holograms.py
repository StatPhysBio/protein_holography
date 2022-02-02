import h5py
from functools import partial
import numpy as np
import os
os.sys.path.append('/gscratch/spe/mpun/protein_holography/fourier')
from hologram import zernike_coeff_l

# slice array along given indices
def slice_array(arr,inds):
    return arr[inds]


def get_hologram(nh,L_max,ks,num_combi_channels,r_max):
    dt = np.dtype([(str(l),'complex64',(num_combi_channels,2*l+1)) for l in range(L_max + 1)])
    arr = np.zeros(shape=(1,),dtype=dt)
    channels = ['C','N','O','S','H','SASA','charge']
    num_channels = len(channels)
    atom_names = nh['atom_names']
    real_locs = atom_names != b''
    elements = nh['elements'][real_locs]
    padded_coords = nh['coords']
    curr_SASA = nh['SASAs'][real_locs]
    curr_charge = nh['charges'][real_locs]
    atom_coords = padded_coords[real_locs]
    r,t,p = np.einsum('ij->ji',atom_coords)
    nmax = len(ks)
    for i_ch,ch in enumerate(channels):
        weights=None
        if ch == 'C':
            r,t,p = *atom_coords[elements == b'C'].T,
        if ch == 'N':
            r,t,p = *atom_coords[elements == b'N'].T,
        if ch == 'O':
            r,t,p = *atom_coords[elements == b'O'].T, 
        if ch == 'S':
            r,t,p = *atom_coords[elements == b'S'].T, 
        if ch == 'H':
            r,t,p = *atom_coords[elements == b'H'].T, 
        if ch == 'SASA':
            weights = curr_SASA
            r,t,p = np.einsum('ij->ji',atom_coords)
            #print(weights)
        if ch == 'charge':
            r,t,p = np.einsum('ij->ji',atom_coords)
            weights = curr_charge
        for n in np.arange(len(ks)):
            for l in range(L_max +1):
                arr[0][l][i_ch*nmax + n,:] = zernike_coeff_l(r,t,p,n,r_max,l,weights)

    return arr[0]
    
