#
# This file computes the atomic spherical coordinates in a given set of
# neighborhoods and outputs a file with these coordinates.
#
# It takes as arguments:
#  - The name of the ouput file
#  - Name of central residue dataset
#  - Number of threads
#  - The neighborhood radius
#  - "easy" flag to include central res
#
from argparse import ArgumentParser
import logging
import sys

import h5py
import numpy as np
from progress.bar import Bar

from protein_holography.coordinates.pyrosetta_hdf5_proteins import get_structural_info, pad_structural_info
from protein_holography.coordinates.preprocessor_pdbs import PDBPreprocessor
from protein_holography.utils.posterity import get_metadata,record_metadata

def c(pose,padded_length=200000):

    if pose is None:
        print('pose is none')
        return (None,)

    try:
        pdb,ragged_structural_info = get_structural_info(pose)
        mat_structural_info = pad_structural_info(
            ragged_structural_info,padded_length=padded_length
        )
    except Exception as e:
        print(e)
        print('Error with',pose.pdb_info().name())
        return (None,)


    
    return (pdb,*mat_structural_info)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--hdf5_out', dest='hdf5_out', type=str, help='ouptut hdf5 filename', required=True)
    parser.add_argument('--pdb_list', dest='pdb_list', type=str, help='pdb list within hdf5_in file', required=True)
    parser.add_argument('--parallelism', dest='parallelism', type=int, help='ouptput file name', default=4)
    parser.add_argument('--hdf5_in', dest='hdf5_in', type=str, help='hdf5 filename', required=True)
    parser.add_argument('--pdb_dir', dest='pdb_dir', type=str, help='directory of pb files', required=True)
    
    args = parser.parse_args()
    
    # get metadata
    metadata = get_metadata()

    
    logging.basicConfig(level=logging.DEBUG)
    ds = PDBPreprocessor(args.hdf5_in,args.pdb_list,args.pdb_dir)
    bad_neighborhoods = []
    n = 0


    max_atoms = 200000
    dt = np.dtype([
        ('pdb','S4',()),
        ('atom_names', 'S4', (max_atoms)),
        ('elements', 'S3', (max_atoms)),
        ('res_ids', 'S5', (max_atoms,6)),
        ('coords', 'f8', (max_atoms,3)),
        ('SASAs', 'f8', (max_atoms)),
        ('charges', 'f8', (max_atoms)),
    ])
    with h5py.File(args.hdf5_out,'w') as f:
        f.create_dataset(args.pdb_list,
                         shape=(ds.size,),
                         dtype=dt)
    with Bar('Processing', max = ds.count(), suffix='%(percent).1f%%') as bar:
        with h5py.File(args.hdf5_out,'r+') as f:
            for i,structural_info in enumerate(ds.execute(
                    c,
                    limit = None,
                    params = {'padded_length': max_atoms},
                    parallelism = args.parallelism)):
                if structural_info[0] is None:
                    bar.next()
                    n+=1
                    continue
                pdb,atom_names,elements,res_ids,coords,sasas,charges = (*structural_info,)
                #print(pdb)
                #print(max_atoms - np.sum(atom_names == b''),'atoms in array')
                #print('wrting to hdf5')
                try:
                    f[args.pdb_list][i] = (pdb,atom_names,elements,res_ids,coords,sasas,charges)
                    #print('success')
                except Exception as e:
                    print(e)
                #print('done writing. \n moving to next entry')
                n+=1
                bar.next()
    
