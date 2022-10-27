"""Unit test for parellel cooridnate-gathering routine"""

import os
import sys
import numpy as np

import h5py
#import pyrosetta
import pytest


from protein_holography.coordinates.get_structural_info import (
    get_structural_info_from_dataset)
from protein_holography.coordinates.get_neighborhoods import (
    get_neighborhoods_from_dataset)
from protein_holography.coordinates.pyrosetta_hdf5_zernikegrams import get_hologram


padded_length = 200000
r_max = 10.
L_max = 5
ks = np.arange(21)
num_combi_channels = 147


    
# init_flags = ('-ignore_unrecognized_res 1 -include_current -ex1 -ex2 -mute all'
#               '-ignore_zero_occupancy false -obey_ENDMDL 1')
# pyrosetta.init(init_flags)

true_dataset = 'parallel_proteinG'
with h5py.File('parallel_proteinG_true_proteins.hdf5','r') as f: 
    true_structural_info = f[true_dataset][:]
with h5py.File('parallel_proteinG_true_neighborhoods.hdf5','r') as f:
    true_neighborhoods = f[true_dataset][:]
with h5py.File('parallel_proteinG_true_zernikegrams.hdf5','r') as f:
    true_zernikegrams = f[true_dataset][:]


#
# structural info test
#
pdb_hdf5= 'parallel_proteinG_test_pdbs.hdf5'
proteins_hdf5 = 'parallel_proteinG_test_proteins.hdf5'
pdb_list = 'parallel_proteinG'
pdb_dir = '.'
max_atoms = padded_length
parallelism = 4
get_structural_info_from_dataset(
    pdb_hdf5, pdb_list, pdb_dir, max_atoms, proteins_hdf5, parallelism
)

         
def test_structural_info():
    with h5py.File(proteins_hdf5,'r') as f:
        for i,test_structural_info in enumerate(f[pdb_list]):
            assert test_structural_info == true_structural_info[i]


#
# neighborhoods test
#

neighborhoods_hdf5 = 'parallel_protein_G_test_neighborhoods.hdf5'
protein_list = 'parallel_proteinG'
num_nhs = 280
r_max = 10.
unique_chains = True
parallelism = 4
get_neighborhoods_from_dataset(
    proteins_hdf5, protein_list, num_nhs, r_max, neighborhoods_hdf5,
    unique_chains, parallelism
)
def test_neighborhoods():
    with h5py.File(neighborhoods_hdf5,'r') as f:
        test_padded_neighborhoods = f[protein_list][:]
        print('shape1:',test_padded_neighborhoods.shape)
        print('shape2:',true_neighborhoods.shape)
        assert (test_padded_neighborhoods == true_neighborhoods).all()

# def test_zernikegrams():

#     testing_zernikegrams = get_hologram(
#         test_padded_neighborhoods[0],
#         L_max,
#         ks,
#         num_combi_channels,
#         r_max
#     )

#     assert (testing_zernikegrams == true_zernikegrams)
