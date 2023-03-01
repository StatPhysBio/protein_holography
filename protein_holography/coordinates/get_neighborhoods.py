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
"""Gather neighborhoods from structural infos"""
from argparse import ArgumentParser
import logging
import sys
import traceback

import h5py
from hdf5plugin import LZ4
import numpy as np
from progress.bar import Bar
from tqdm import tqdm

#sys.path.append('/gscratch/spe/mpun/protein_holography/utils')
from protein_holography.coordinates.pyrosetta_hdf5_neighborhoods import (
    get_neighborhoods_from_protein,
    pad_neighborhoods
)
from protein_holography.coordinates.preprocessor_hdf5_proteins import (
    PDBPreprocessor)
from protein_holography.utils.posterity import get_metadata,record_metadata

def get_padded_neighborhoods(np_protein,r_max,padded_length,unique_chains):
    """
    Gets padded neighborhoods associated with one structural info unit
    
    Parameters:
    np_protein : np.ndarray
        Array representation of a protein
    r_max : float
        Radius of the neighborhood
    padded_length : int
        Total length including padding
    unique_chains : bool
        Flag indicating whether chains with identical sequences should 
        contribute unique neoighborhoods
    """
    try:
        neighborhoods = get_neighborhoods_from_protein(
            np_protein,r_max,uc=unique_chains)
        padded_neighborhoods = pad_neighborhoods(
            neighborhoods,padded_length=padded_length)
        del neighborhoods
    except Exception as e:
        print(e)
        logging.error(f"Error with{np_protein[0]}")
        #print(traceback.format_exc())
        return (None,)
    
    return (padded_neighborhoods)

def get_neighborhoods_from_dataset(
        hdf5_in,
        protein_list,
        num_nhs,
        r_max,
        hdf5_out,
        unique_chains,
        parallelism,
        compression=LZ4()
):
    """
    Parallel retrieval of neighborhoods from structural info file and writing
    to neighborhods hdf5_out file
    
    Parameters
    ----------
    hdf5_in : str
        Path to hdf5 file containing structural info
    protein_list : str
        Name of the dataset within the hdf5 file to process
    num_nhs : int
        Number of neighborhoods to expect in total
    r_max : float
        Radius of the neighborhood
    hdf5_out : str
        Path to write the output file 
    unique_chains : bool
        Flag indicating whether or not chains with identical sequences should each
        contribute neighborhoods
    parallelism : int
        Number of workers to use
    """
    metadata = get_metadata()


    logging.basicConfig(level=logging.DEBUG)
    ds = PDBPreprocessor(hdf5_in,protein_list)
    bad_neighborhoods = []
    n = 0
    L = np.max([ds.pdb_name_length, 5])

    max_atoms = 1300
    dt = np.dtype([
        ('res_id', f'S{L}',(6)),
        ('atom_names', 'S4', (max_atoms)),
        ('elements', 'S1', (max_atoms)),
        ('res_ids', f'S{L}', (max_atoms,6)),
        ('coords', 'f8', (max_atoms,3)),
        ('SASAs', 'f8', (max_atoms)),
        ('charges', 'f8', (max_atoms)),
    ])
    
    logging.info(f"Extracting {num_nhs} neighborhoods")
    logging.info("Writing hdf5 file")
    with h5py.File(hdf5_out,'w') as f:
        f.create_dataset(protein_list,
                         shape=(num_nhs,),
                         dtype=dt,
                         compression=compression)
        record_metadata(metadata, f[protein_list])

    logging.debug(f"Gathering unique chains {unique_chains}")
    nhs = np.empty(shape=num_nhs,dtype=('S5',(6)))
    with Bar('Processing', max = ds.count(), suffix='%(percent).1f%%') as bar:
        with h5py.File(hdf5_out,'r+') as f:
            for i,neighborhoods in enumerate(ds.execute(
                    get_padded_neighborhoods,
                    limit = None,
                    params = {
                        'r_max': r_max,
                        'padded_length' : max_atoms,
                        'unique_chains': unique_chains
                    },
                    parallelism = parallelism)):
                if neighborhoods[0] is None:
                    del neighborhoods
                    bar.next()
                    continue
                    
                neighborhoods_per_protein = neighborhoods.shape[0]
                f[protein_list][n:n+neighborhoods_per_protein] = neighborhoods
                nhs[n:n+neighborhoods_per_protein] = neighborhoods['res_id']
                n+=neighborhoods_per_protein
                
                # attempt to address memory issues. currently unsuccessfully
                del neighborhoods
                bar.next()

    with h5py.File(hdf5_out,'r+') as f:
        f.create_dataset('nh_list',
                         data=nhs)
        record_metadata(metadata, f["nh_list"])
    
    print('Done with parallel computing')


def main():
    parser = ArgumentParser()
    
    parser.add_argument('--hdf5_in', dest='hdf5_in', type=str,
                        help='hdf5 filename', default=False)
    parser.add_argument('--hdf5_out', dest='hdf5_out', type=str,
                        help='ouptut hdf5 filename', required=True)
    parser.add_argument('--protein_list', dest='protein_list', type=str,
                        help='protein list within hdf5_in file', required=True)
    parser.add_argument('--num_nhs', dest='num_nhs', type=int,
                        help='number of neighborhoods in protein set')
    parser.add_argument('--r_max', dest='r_max', type=float,
                        help='radius of neighborhood')
    parser.add_argument('--unique_chains', dest='unique_chains',
                        action='store_true',default=False, 
                        help='Only take one neighborhood'
                        'per residue per unique chain')
    parser.add_argument('--parallelism', dest='parallelism', type=int,
                        help='ouptput file name', default=4)
    
    args = parser.parse_args()

    get_neighborhoods_from_dataset(
        args.hdf5_in,
        args.protein_list,
        args.num_nhs,
        args.r_max,
        args.hdf5_out,
        args.unique_chains,
        args.parallelism
    )
    
    
if __name__ == "__main__":
    main()
