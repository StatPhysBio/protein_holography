"""Module for parallel processing of pdb files into structural info"""

from argparse import ArgumentParser
import logging
import sys
from typing import Tuple

import h5py
import numpy as np
from progress.bar import Bar
from pyrosetta.rosetta.core.pose import Pose

from protein_holography.coordinates.pyrosetta_hdf5_proteins import (
    get_structural_info, pad_structural_info
)
from protein_holography.coordinates.preprocessor_pdbs import PDBPreprocessor
from protein_holography.utils.posterity import get_metadata,record_metadata


def get_padded_structural_info(
    pose: Pose, padded_length: int=200000) -> Tuple[
    bytes,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    """
    Extract structural info used for holographic projection from PyRosetta pose.
    
    Parameters
    ----------
    pose : pyrosetta.rosetta.core.pose.Pose
        Pose created by PyRosetta from pdb file
        
    Returns
    -------
    tuple of (bytes, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, 
              np.ndarray)
        The entries in the tuple are
            bytes encoding the pdb name string
            bytes array encoding the atom names of shape [max_atoms]
            bytes array encoding the elements of shape [max_atoms]
            bytes array encoding the residue ids of shape [max_atoms,6]
            float array of shape [max_atoms,3] representing the 3D Cartesian 
              coordinates of each atom
            float array of shape [max_atoms] storing the SASA of each atom
            float array of shape [max_atoms] storing the partial charge of each atom
    """
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


def get_structural_info_from_dataset(
    hdf5_in: str,
    pdb_list: str,
    pdb_dir: str,
    max_atoms: int,
    hdf5_out: str,
    parllelism: int    
):
    """
    Parallel processing of pdbs into structural info
    
    Parameters
    ---------
    hdf5_in : str
        Path to hdf5 file containing pdb ids to process
    pdb_list : str
        Name of the dataset within hdf5_in to process
    pdb_dir : str
        Path where the pdb files are stored
    max_atoms : int
        Max number of atoms in a protein for padding purposes
    hdf5_out : str
        Path to hdf5 file to write
    parlellism : int
        Number of workers to use
    """
    metadata = get_metadata()
    
    logging.basicConfig(level=logging.DEBUG)
    
    ds = PDBPreprocessor(args.hdf5_in,args.pdb_list,args.pdb_dir)
    bad_neighborhoods = []
    n = 0

    
    max_atoms = 200000
    dt = np.dtype([
        ('pdb','S4',()),
        ('atom_names', 'S4', (max_atoms)),
        ('elements', 'S1', (max_atoms)),
        ('res_ids', 'S5', (max_atoms,6)),
        ('coords', 'f8', (max_atoms,3)),
        ('SASAs', 'f8', (max_atoms)),
        ('charges', 'f8', (max_atoms)),
    ])
    
    with h5py.File(args.hdf5_out,'w') as f:
        f.create_dataset(args.pdb_list,
                         shape=(ds.size,),
                         dtype=dt)
    print("beginning data gathering process")    
    with Bar('Processing', max = ds.count(), suffix='%(percent).1f%%') as bar:
        with h5py.File(args.hdf5_out,'r+') as f:
            for i,structural_info in enumerate(ds.execute(
                    get_padded_structural_info,
                    limit = None,
                    params = {'padded_length': max_atoms},
                    parallelism = args.parallelism)):
                if structural_info[0] is None:
                    bar.next()
                    n+=1
                    continue
                (pdb,atom_names,elements,
                 res_ids,coords,sasas,charges) = (*structural_info,)

                try:
                    f[args.pdb_list][i] = (
                        pdb, atom_names, elements,
                        res_ids, coords, sasas, charges
                    )
                except Exception as e:
                    print(e)
                n+=1
                bar.next()

def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--hdf5_in', dest='hdf5_in', type=str,
        help='hdf5 filename', required=True
    )
    parser.add_argument(
        '--hdf5_out', dest='hdf5_out', type=str,
        help='ouptut hdf5 filename', required=True
    )
    parser.add_argument(
        '--pdb_list', dest='pdb_list', type=str,
        help='dataset containing pdb list within hdf5_in file',
        required=True
    )
    parser.add_argument(
        '--pdb_dir', dest='pdb_dir', type=str,
        help='directory of pb files', required=True
    )    
    parser.add_argument(
        '--parallelism', dest='parallelism', type=int,
        help='ouptput file name', default=4
    )
    parser.add_argument(
        '--max_atoms', dest='max_atoms', type=int,
        help='max number of atoms per protein for padding purposes',
        default=200000
    )
    
    args = parser.parse_args()

    get_structural_info_from_dataset(
        args.hdf5_in,
        args.pdb_list,
        args.pdb_dir,
        args.max_atoms,
        args.hdf5_out,
        args.parallelism,
    )

if __name__ == "__main__":
    main()
