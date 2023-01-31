"""Parsing module for HCNN predictions"""

import logging
import os
from typing import Dict, List

import numpy as np
from tensorflow.data import Dataset
from tensorflow.keras import Model

from protein_holography.utils.protein import (
    ol_to_ind_size)

L_MAX = 5

def get_energy(
    seq: str,
    energies: np.ndarray
) -> float:
    """Calculate pseudoenergy given a prediction matrix and a sequence"""
    seq_idxs = np.array([ol_to_ind_size[x] for x in seq])
    network_energy = np.sum(energies[np.arange(len(seq_idxs)),seq_idxs])
    return network_energy

def get_res_locs(
    nh_ids: List,
    loc_ids: List
) -> np.ndarray:
    """
    Get residue locations in order for a list of residues specified by
    their residues ids
    """
    nh_string_ids = np.array([b''.join(x) for x in nh_ids])
    loc_string_ids = np.array([b''.join(x) for x in loc_ids])
    return np.squeeze(np.argwhere(
        np.logical_or.reduce(
            nh_string_ids[None,:] == loc_string_ids[:,None])))

def make_string_from_tup(x):
    return (x[0] + str(x[1]) + x[2]).encode()

def get_res_locs_from_tups(
    nh_ids: List,
    loc_tups: List
) -> np.ndarray:
    """Get indices of specific residues based on their residue ids"""
    nh_string_ids = np.array([b''.join(x) for x in nh_ids[:,2:5]])
    loc_string_ids = np.array([make_string_from_tup(x) for x in loc_tups])
    return np.squeeze(np.argwhere(
        np.logical_or.reduce(
            nh_string_ids[None,:] == loc_string_ids[:,None])))

def get_protein_network_energy(
    seq: str,
    pseudoenergies: np.ndarray
) -> float:
    """
    Get protein network energy for a given sequence over a matrix of 
    predictions
    """
    idx_seq = [ol_to_ind_size[x] for x in seq]
    return np.sum(pseudoenergies[np.arange(len(seq)),idx_seq])

def get_region_metrics(
    np_zgrams: np.ndarray,
    nh_ids: np.ndarray,
    region_name: str,
    region_tups: Dict,
    network: Model,
    bs: int=64,    
) -> Dict:
    """
    Get pseudoenergies and protein network energies for a region 
    specified by pdb res_ids
    """
    region_idxs = get_res_locs_from_tups(nh_ids, region_tups)
    region_size = len(region_idxs)
    region_zgrams = {l:np_zgrams[region_idxs][str(l)] for l in range(L_MAX + 1)}
    region_aas = nh_ids[:,0][region_idxs]
    region_one_hots = np.zeros(shape=(region_size,20))
    region_one_hots[
        np.arange(region_size),
        np.array(
            [ol_to_ind_size[x.decode('utf-8')]
             for x in region_aas])
    ] = 1.
    region_ds = Dataset.from_tensor_slices(
        (region_zgrams, region_one_hots))
    
    region_pes = network.predict(region_ds.batch(bs))
    region_seq = b''.join(region_aas).decode('utf-8')
    region_pnE = get_protein_network_energy(region_seq, region_pes)
    
    metric_dict = {}
    metric_dict = {
        f'{region_name} idxs': region_idxs,
        f'{region_name} pes': region_pes,
        f'{region_name} seq': region_seq,
        f'{region_name} pnE': region_pnE
    }
    return metric_dict

