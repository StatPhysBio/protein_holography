"""Unit test for cooridnate gathering routine"""

import os
import sys
import numpy as np

import h5py
import pyrosetta
import pytest


from protein_holography.coordinates.pyrosetta_hdf5_proteins import (
    get_structural_info, pad_structural_info)
from protein_holography.coordinates.pyrosetta_hdf5_neighborhoods import (
    get_neighborhoods_from_protein,pad_neighborhoods)
from protein_holography.coordinates.pyrosetta_hdf5_zernikegrams import (
    get_hologram)

test_pdb = '1PGA'
padded_length = 200000
r_max = 10.
L_max = 5
ks = np.arange(21)
num_combi_channels = 147


    
init_flags = ('-ignore_unrecognized_res 1 -include_current -ex1 -ex2 -mute all'
              '-ignore_zero_occupancy false -obey_ENDMDL 1')
pyrosetta.init(init_flags)

with h5py.File('proteinG_structural_info.hdf5','r') as f: 
    true_structural_info = f['proteinG'][0]
with h5py.File('proteinG_neighborhoods.hdf5','r') as f:
    true_neighborhoods = f['proteinG'][:]
with h5py.File('proteinG_zernikegrams.hdf5','r') as f:
    true_zernikegrams = f['proteinG'][0]

test_pose = pyrosetta.pose_from_pdb(test_pdb + '.pdb')
pdb,test_structural_info = get_structural_info(test_pose)
padded_test_structural_info = (
    pdb,
    *pad_structural_info(
        test_structural_info,
        padded_length=padded_length
    )
)
dt = true_structural_info.dtype

test_structural_info_arr = np.array([padded_test_structural_info],dtype=dt)

print(test_structural_info_arr.dtype,test_structural_info_arr.shape)

test_structural_info_arr[0]

neighborhoods = get_neighborhoods_from_protein(
    test_structural_info_arr[0],
    #true_structural_info,
    r_max,
    uc=True
)

test_padded_neighborhoods = pad_neighborhoods(
    neighborhoods,
    padded_length=1700
)


         
def test_structural_info_simple():
    assert test_structural_info_arr == true_structural_info

            
def test_neighborhoods():
    assert (test_padded_neighborhoods == true_neighborhoods).all()

def test_zernikegrams():
    testing_zernikegrams = get_hologram(
        test_padded_neighborhoods[0],
        L_max,
        ks,
        num_combi_channels,
        r_max
    )
    assert (testing_zernikegrams == true_zernikegrams)
