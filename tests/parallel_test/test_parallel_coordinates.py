"""Unit test for parellel cooridnate-gathering routine"""

import os
import sys
import numpy as np
from pathlib import Path

import h5py
#import pyrosetta
import logging
import pytest

import protein_holography
from protein_holography.coordinates.get_structural_info import (
    get_structural_info_from_dataset)
from protein_holography.coordinates.get_neighborhoods import (
    get_neighborhoods_from_dataset)
from protein_holography.coordinates.get_zernikegrams import (
    get_zernikegrams_from_dataset)


padded_length = 200000

phdir = Path(protein_holography.__file__).parents[1]
    
# init_flags = ('-ignore_unrecognized_res 1 -include_current -ex1 -ex2 -mute all'
#               '-ignore_zero_occupancy false -obey_ENDMDL 1')
# pyrosetta.init(init_flags)

true_dataset = 'parallel_proteinG'
with h5py.File(
    os.path.join(
        phdir, 
        "tests/parallel_test",
        'parallel_proteinG_true_proteins.hdf5'),
    'r') as f: 
    true_structural_info = f[true_dataset][:]
with h5py.File(
    os.path.join(
        phdir, 
        "tests/parallel_test",
        'parallel_proteinG_true_neighborhoods.hdf5'),
    'r') as f:
    true_neighborhoods = f[true_dataset][:]
with h5py.File(
    os.path.join(
        phdir, 
        "tests/parallel_test",
        'parallel_proteinG_true_zernikegrams.hdf5'),
    'r') as f:
    true_zernikegrams = f[true_dataset][:]


#
# structural info test
#
pdb_hdf5= os.path.join(
    phdir, "tests/parallel_test", 'parallel_proteinG_pdbs.hdf5')
proteins_hdf5 = os.path.join(
    phdir, "tests/parallel_test", 'parallel_proteinG_test_proteins.hdf5')
dataset = 'parallel_proteinG'
pdbdir = os.path.join(phdir, "tests/parallel_test")
max_atoms = padded_length
parallelism = 4
get_structural_info_from_dataset(
    pdb_hdf5, dataset, pdbdir, max_atoms, proteins_hdf5, parallelism
)
         
def test_structural_info():
    with h5py.File(proteins_hdf5,'r') as f:
        for i,test_structural_info in enumerate(f[dataset]):
            assert test_structural_info == true_structural_info[i]
    print(f"Size of protein file:{os.path.getsize(proteins_hdf5) / 1e6:.2f} MB")
    os.system(f"rm {proteins_hdf5}")
#
# neighborhoods test
#

neighborhoods_hdf5 = 'parallel_proteinG_test_neighborhoods.hdf5'
num_nhs = 280
r_max = 10.
unique_chains = True
parallelism = 4
get_neighborhoods_from_dataset(
    proteins_hdf5, dataset, num_nhs, r_max, neighborhoods_hdf5,
    unique_chains, parallelism
)

def test_neighborhoods():
    with h5py.File(neighborhoods_hdf5,'r') as f:
        test_padded_neighborhoods = f[dataset][:]
        if (test_padded_neighborhoods == true_neighborhoods).all():
            assert True
        else:
            for field in true_neighborhoods.dtype.names:
                if field == "coords":
                    assert np.mean(
                        true_neighborhoods[field] - 
                        test_padded_neighborhoods[field] < 1e-14)
                else:
                    assert (
                        true_neighborhoods[field] ==
                         test_padded_neighborhoods[field]
                    ).all()
    print(f"Size of nh file:{os.path.getsize(neighborhoods_hdf5) / 1e6:.2f} MB")
    os.system(f"rm {neighborhoods_hdf5}")
        
#
# zernikegrams test
#

L_max = 5
L_max = 5
ks = np.arange(21)
zgram_hdf5 = 'parallel_proteinG_test_zernikegrams.hdf5'
#num_combi_channels = 147
get_zernikegrams_from_dataset(
    neighborhoods_hdf5,
    dataset,
    num_nhs,
    r_max,
    L_max,
    ks,
    zgram_hdf5,
    parallelism
)
    
def test_zernikegrams():
    with h5py.File(zgram_hdf5,'r') as f:
        testing_zernikegrams = f[dataset][:]
    assert (testing_zernikegrams == true_zernikegrams).all()
    print(f"Size of zgram file:{os.path.getsize(zgram_hdf5) / 1e6:.2f} MB")
    os.system(f"rm {zgram_hdf5}")



       
