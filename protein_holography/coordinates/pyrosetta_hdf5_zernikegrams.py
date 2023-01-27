"""Zenrikegram projection"""
from functools import partial
import os

import  h5py
import numpy as np
import scipy as sp
from scipy import special
os.sys.path.append('/gscratch/spe/mpun/protein_holography/fourier')

def zernike_coeff_lm_new(
        r: np.ndarray,
        t: np.ndarray,
        p: np.ndarray,
        n: np.ndarray,
        r_max: np.float64,
        l: np.ndarray,
        m: np.ndarray,
        weights: np.ndarray
) -> np.ndarray:
    """
    Compute Zerkinke coefficients.

    This implementation uses vectorized operations and avoids unnecessary 
    computation when possible.

    Parameters
    ----------
    r : np.ndarray
        Radii magnitudes.
    t : np.ndarray
        Theta values.
    p : np.ndarray
        Phi values.
    n : np.ndarray

    r_max : np.float64

    l : np.ndarray

    m :  np.ndarray

    weights : np.ndarray


    Returns
    -------
    coeffs : np.ndarray
        Zerkine coefficients.
    """
    # Dimension of the Zernike polynomial.
    D = 3.

    # Constituent terms in the polynomial.
    A = np.power(-1.0 + 0j, (n - l) / 2.)

    B = np.sqrt(2. * n + D)
    C = sp.special.binom((n + l + D) // 2 - 1, (n - l) // 2)

    nl_unique_combs, nl_inv_map = np.unique(np.vstack([n, l]).T, axis=0,
                                            return_inverse=True)

    num_nl_combs = nl_unique_combs.shape[0]
    n_hyp2f1_tile = np.tile(nl_unique_combs[:, 0], (r.shape[1], 1)).T
    l_hyp2f1_tile = np.tile(nl_unique_combs[:, 1], (r.shape[1], 1)).T

    E_unique = sp.special.hyp2f1(-(n_hyp2f1_tile - l_hyp2f1_tile) / 2.,
                                 (n_hyp2f1_tile + l_hyp2f1_tile + D) /2.,
                                 l_hyp2f1_tile + D / 2.,
                                 r[:num_nl_combs, :]**2 / r_max**2)
    E = E_unique[nl_inv_map]

    l_unique, l_inv_map = np.unique(l, return_inverse=True)
    l_power_tile = np.tile(l_unique, (r.shape[1], 1)).T
    F_unique = np.power(r[:l_unique.shape[0]] / r_max, l_power_tile)
    F = F_unique[l_inv_map]

    # Spherical harmonic component.
    lm_unique_combs, lm_inv_map = np.unique(np.vstack([l, m]).T, axis=0,
                                            return_inverse=True)
    num_lm_combs = lm_unique_combs.shape[0]
    l_sph_harm_tile = np.tile(lm_unique_combs[:, 0], (p.shape[1], 1)).T
    m_sph_harm_tile = np.tile(lm_unique_combs[:, 1], (p.shape[1], 1)).T

    y_unique = np.conj(sp.special.sph_harm(m_sph_harm_tile, l_sph_harm_tile,
                                           p[:num_lm_combs], t[:num_lm_combs]))
    y = y_unique[lm_inv_map]

    if True in np.isinf(E):
        print('Error: E is inf')
        print(f'E={E}, n={n}, l={l}, D={D}, r={np.array(r)}, rmax={r_max}')

    # n indexes the combinations of n, l, m and N indexes the points in the point cloud
    coeffs = A * B * C * np.einsum('cN,nN,nN,nN->cn', weights, E, F, y)

    return coeffs



def get_hologram(nh, L_max: int, ks, num_combi_channels, r_max: np.float64):
    dt = np.dtype([(str(l),'complex64',(num_combi_channels,2*l+1)) for l in range(L_max + 1)])
    arr = np.zeros(shape=(1,),dtype=dt)

    # get info from nh
    channels = ['C','N','O','S','H','SASA','charge']
    num_channels = len(channels)
    atom_names = nh['atom_names']
    real_locs = np.logical_and(atom_names != b'',nh['coords'][:,0] <= r_max)
    #real_locs = atom_names != b''
    elements = nh['elements'][real_locs]
    padded_coords = nh['coords']
    curr_SASA = nh['SASAs'][real_locs]
    curr_charge = nh['charges'][real_locs]
    atom_coords = padded_coords[real_locs]
    r,t,p = np.einsum('ij->ji',atom_coords)


    ns = []
    ls = []
    ms = []
    for l in range(L_max + 1):
        for n in ks:
            m_to_append = np.arange(-l, l + 1)

            ns.append(np.zeros(shape=(2*l + 1), dtype=int) + n)
            ls.append(np.zeros(shape=(2*l + 1), dtype=int) + l)
            ms.append(m_to_append)

    ns = np.concatenate(ns)
    ls = np.concatenate(ls)
    ms = np.concatenate(ms)
    l_greater_n = ns < ls
    odds = ((ns - ls) % 2 == 1)
    nonzero_idxs = ~(l_greater_n | odds)
    nonzero_len = np.count_nonzero(nonzero_idxs)
    nmax = len(ks)

    arr_weights = np.empty(shape=(7,r.shape[-1],))
    # for i_ch,ch in enumerate(channels):


    #     r,t,p = np.einsum('ij->ji',atom_coords)
    #     if ch == 'C':
    #         weights=np.array(elements == b'C',dtype=float)
    #     if ch == 'N':
    #         weights=np.array(elements == b'N',dtype=float)
    #     if ch == 'O':
    #         weights=np.array(elements == b'O',dtype=float)
    #     if ch == 'S':
    #         weights=np.array(elements == b'S',dtype=float)
    #     if ch == 'H':
    #         weights=np.array(elements == b'H',dtype=float)
    #     if ch == 'SASA':
    #         weights = curr_SASA
    #     if ch == 'charge':
    #         weights = curr_charge

    #    arr_weights[i_ch] = weights
    arr_weights[0] = np.array(elements == b'C', dtype=float)
    arr_weights[1] = np.array(elements == b'N', dtype=float)
    arr_weights[2] = np.array(elements == b'O', dtype=float)
    arr_weights[3] = np.array(elements == b'S', dtype=float)
    arr_weights[4] = np.array(elements == b'H', dtype=float)
    arr_weights[5] = curr_SASA
    arr_weights[6] = curr_charge

    ch_num = len(channels)
    out_z = np.zeros(shape=(ch_num,ns.shape[0]), dtype=np.complex64)

    rs = np.tile(r, (nonzero_len, 1))
    ts = np.tile(t, (nonzero_len, 1))
    ps = np.tile(p, (nonzero_len, 1))

    out_z[:,nonzero_idxs] = zernike_coeff_lm_new(rs, ts, ps, ns[nonzero_idxs],
                                                   10.0, ls[nonzero_idxs], ms[nonzero_idxs],
                                                   arr_weights)
    #return out_z
    low_idx = 0
    for l in range(L_max + 1):
        num_m = (2 * l + 1)
        high_idx = (nmax) * num_m  + low_idx
        arr[0][l][:,:] = out_z[:,low_idx:high_idx].reshape(nmax*ch_num, num_m, )
        low_idx = high_idx

    return arr[0]

def get_sparse_hologram(nh,L_max,ks,num_combi_channels,r_max):
    dt = np.dtype([(str(l),'complex64',(num_combi_channels,2*l+1)) for l in range(L_max + 1)])
    arr = np.zeros(shape=(1,),dtype=dt)

    # get info from nh
    channels = ['C','CA','N','O','CB']
    num_channels = len(channels)
    atom_names = nh['atom_names']
    real_locs = np.logical_and(atom_names != b'',nh['coords'][:,0] <= r_max)
    #real_locs = atom_names != b''
    atom_names = nh['atom_names'][real_locs]
    elements = nh['elements'][real_locs]
    padded_coords = nh['coords']
    curr_SASA = nh['SASAs'][real_locs]
    curr_charge = nh['charges'][real_locs]
    atom_coords = padded_coords[real_locs]
    r,t,p = np.einsum('ij->ji',atom_coords)


    ns = []
    ls = []
    ms = []
    for l in range(L_max + 1):
        for n in ks:
            m_to_append = np.arange(-l, l + 1)

            ns.append(np.zeros(shape=(2*l + 1), dtype=int) + n)
            ls.append(np.zeros(shape=(2*l + 1), dtype=int) + l)
            ms.append(m_to_append)

    ns = np.concatenate(ns)
    ls = np.concatenate(ls)
    ms = np.concatenate(ms)
    l_greater_n = ns < ls
    odds = ((ns - ls) % 2 == 1)
    nonzero_idxs = ~(l_greater_n | odds)
    nonzero_len = np.count_nonzero(nonzero_idxs)
    nmax = len(ks)

    arr_weights = np.empty(shape=(5,r.shape[-1],))
    # for i_ch,ch in enumerate(channels):


    #     r,t,p = np.einsum('ij->ji',atom_coords)
    #     if ch == 'C':
    #         weights=np.array(elements == b'C',dtype=float)
    #     if ch == 'N':
    #         weights=np.array(elements == b'N',dtype=float)
    #     if ch == 'O':
    #         weights=np.array(elements == b'O',dtype=float)
    #     if ch == 'S':
    #         weights=np.array(elements == b'S',dtype=float)
    #     if ch == 'H':
    #         weights=np.array(elements == b'H',dtype=float)
    #     if ch == 'SASA':
    #         weights = curr_SASA
    #     if ch == 'charge':
    #         weights = curr_charge
            
    #    arr_weights[i_ch] = weights
    arr_weights[0] = np.array(atom_names == b'C',dtype=float)
    arr_weights[1] = np.array(atom_names == b'CA',dtype=float)
    arr_weights[2] = np.array(atom_names == b'N',dtype=float)
    arr_weights[3] = np.array(atom_names == b'O',dtype=float)
    arr_weights[4] = np.array(atom_names == b'CB',dtype=float)

    
    ch_num = len(channels)
    out_z = np.zeros(shape=(ch_num,ns.shape[0]), dtype=np.complex64)

    rs = np.tile(r, (nonzero_len, 1))
    ts = np.tile(t, (nonzero_len, 1))
    ps = np.tile(p, (nonzero_len, 1))

        
    out_z[:,nonzero_idxs] = zernike_coeff_lm_new(rs, ts, ps, ns[nonzero_idxs],
                                                   10.0, ls[nonzero_idxs], ms[nonzero_idxs],
                                                   arr_weights)
    #return out_z
        
        
    low_idx = 0
    for l in range(L_max + 1):
        num_m = (2 * l + 1)
        high_idx = (nmax) * num_m  + low_idx
        arr[0][l][:,:] = out_z[:,low_idx:high_idx].reshape(nmax*ch_num, num_m, )
        low_idx = high_idx
        
        
            
    return arr[0]

def get_backbone_hologram(nh,L_max,ks,num_combi_channels,r_max):
    dt = np.dtype([(str(l),'complex64',(num_combi_channels,2*l+1)) for l in range(L_max + 1)])
    arr = np.zeros(shape=(1,),dtype=dt)

    # get info from nh
    channels = ['C','N','O','S','H']
    num_channels = len(channels)
    atom_names = nh['atom_names']
    real_locs = np.logical_and(atom_names != b'',nh['coords'][:,0] <= r_max)
    #real_locs = atom_names != b''
    atom_names = nh['atom_names'][real_locs]
    elements = nh['elements'][real_locs]
    padded_coords = nh['coords']
    curr_SASA = nh['SASAs'][real_locs]
    curr_charge = nh['charges'][real_locs]
    atom_coords = padded_coords[real_locs]
    r,t,p = np.einsum('ij->ji',atom_coords)


    ns = []
    ls = []
    ms = []
    for l in range(L_max + 1):
        for n in ks:
            m_to_append = np.arange(-l, l + 1)

            ns.append(np.zeros(shape=(2*l + 1), dtype=int) + n)
            ls.append(np.zeros(shape=(2*l + 1), dtype=int) + l)
            ms.append(m_to_append)

    ns = np.concatenate(ns)
    ls = np.concatenate(ls)
    ms = np.concatenate(ms)
    l_greater_n = ns < ls
    odds = ((ns - ls) % 2 == 1)
    nonzero_idxs = ~(l_greater_n | odds)
    nonzero_len = np.count_nonzero(nonzero_idxs)
    nmax = len(ks)

    arr_weights = np.empty(shape=(5,r.shape[-1],))
    # for i_ch,ch in enumerate(channels):


    #     r,t,p = np.einsum('ij->ji',atom_coords)
    #     if ch == 'C':
    #         weights=np.array(elements == b'C',dtype=float)
    #     if ch == 'N':
    #         weights=np.array(elements == b'N',dtype=float)
    #     if ch == 'O':
    #         weights=np.array(elements == b'O',dtype=float)
    #     if ch == 'S':
    #         weights=np.array(elements == b'S',dtype=float)
    #     if ch == 'H':
    #         weights=np.array(elements == b'H',dtype=float)
    #     if ch == 'SASA':
    #         weights = curr_SASA
    #     if ch == 'charge':
    #         weights = curr_charge
            
    #    arr_weights[i_ch] = weights
    arr_weights[0] = np.array(np.logical_or.reduce([
        atom_names == b'C',
        atom_names == b'CA',
        atom_names == b'CB',
    ]),dtype=float)
    arr_weights[1] = np.array(atom_names == b'N',dtype=float)
    arr_weights[2] = np.array(atom_names == b'O',dtype=float)
    arr_weights[3] = np.array(atom_names == b'S',dtype=float)
    arr_weights[4] = np.array(np.logical_or.reduce([
        atom_names == b'H',
        atom_names == b'HA',
        atom_names == b'HB',
        atom_names == b'1HB',
        atom_names == b'2HB',
        atom_names == b'3HB',
    ]),dtype=float)

    
    ch_num = len(channels)
    out_z = np.zeros(shape=(ch_num,ns.shape[0]), dtype=np.complex64)

    rs = np.tile(r, (nonzero_len, 1))
    ts = np.tile(t, (nonzero_len, 1))
    ps = np.tile(p, (nonzero_len, 1))

        
    out_z[:,nonzero_idxs] = zernike_coeff_lm_new(rs, ts, ps, ns[nonzero_idxs],
                                                   10.0, ls[nonzero_idxs], ms[nonzero_idxs],
                                                   arr_weights)
    #return out_z
        
        
    low_idx = 0
    for l in range(L_max + 1):
        num_m = (2 * l + 1)
        high_idx = (nmax) * num_m  + low_idx
        arr[0][l][:,:] = out_z[:,low_idx:high_idx].reshape(nmax*ch_num, num_m, )
        low_idx = high_idx
        
        
            
    return arr[0]


def rbf_coeff_lm_new(
        r: np.ndarray,
        t: np.ndarray,
        p: np.ndarray,
        n: np.ndarray,
        r_max: np.float64,
        l: np.ndarray,
        m: np.ndarray,
        weights: np.ndarray
) -> np.ndarray:
    """
    Compute Zerkinke coefficients.

    This implementation uses vectorized operations and avoids unnecessary 
    computation when possible.

    Parameters
    ----------
    r : np.ndarray
        Radii magnitudes.
    t : np.ndarray
        Theta values.
    p : np.ndarray
        Phi values.
    n : np.ndarray

    r_max : np.float64

    l : np.ndarray

    m :  np.ndarray

    weights : np.ndarray


    Returns
    -------
    coeffs : np.ndarray
        Zerkine coefficients.
    """
    # Dimension of the Zernike polynomial.
    D = 3.

    # Constituent terms in the polynomial.
    A = np.power(-1.0 + 0j, (n - l) / 2.)

    B = np.sqrt(2. * n + D)
    C = sp.special.binom((n + l + D) // 2 - 1, (n - l) // 2)

    nl_unique_combs, nl_inv_map = np.unique(np.vstack([n, l]).T, axis=0,
                                            return_inverse=True)

    num_nl_combs = nl_unique_combs.shape[0]
    n_hyp2f1_tile = np.tile(nl_unique_combs[:, 0], (r.shape[1], 1)).T
    l_hyp2f1_tile = np.tile(nl_unique_combs[:, 1], (r.shape[1], 1)).T

    E_unique = sp.special.hyp2f1(-(n_hyp2f1_tile - l_hyp2f1_tile) / 2.,
                                 (n_hyp2f1_tile + l_hyp2f1_tile + D) /2.,
                                 l_hyp2f1_tile + D / 2.,
                                 r[:num_nl_combs, :]**2 / r_max**2)
    E = E_unique[nl_inv_map]

    l_unique, l_inv_map = np.unique(l, return_inverse=True)
    l_power_tile = np.tile(l_unique, (r.shape[1], 1)).T
    F_unique = np.power(r[:l_unique.shape[0]] / r_max, l_power_tile)
    F = F_unique[l_inv_map]

    # Spherical harmonic component.
    lm_unique_combs, lm_inv_map = np.unique(np.vstack([l, m]).T, axis=0,
                                            return_inverse=True)
    num_lm_combs = lm_unique_combs.shape[0]
    l_sph_harm_tile = np.tile(lm_unique_combs[:, 0], (p.shape[1], 1)).T
    m_sph_harm_tile = np.tile(lm_unique_combs[:, 1], (p.shape[1], 1)).T

    y_unique = np.conj(sp.special.sph_harm(m_sph_harm_tile, l_sph_harm_tile,
                                           p[:num_lm_combs], t[:num_lm_combs]))
    y = y_unique[lm_inv_map]

    if True in np.isinf(E):
        print('Error: E is inf')
        print(f'E={E}, n={n}, l={l}, D={D}, r={np.array(r)}, rmax={r_max}')

    # n indexes the combinations of n, l, m and N indexes the points in the point cloud
    coeffs = A * B * C * np.einsum('cN,nN,nN,nN->cn', weights, E, F, y)

    return coeffs
