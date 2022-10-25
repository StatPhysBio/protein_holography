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
import traceback

import h5py
import numpy as np
from progress.bar import Bar
from tqdm import tqdm


sys.path.append('/gscratch/spe/mpun/protein_holography/utils')
from protein_holography.coordinates.pyrosetta_hdf5_neighborhoods import (
    get_neighborhoods_from_protein,
    pad_neighborhoods
)
from protein_holography.coordinates.preprocessor_hdf5_proteins import PDBPreprocessor
from protein_holography.utils. posterity import get_metadata,record_metadata

def c(np_protein,r_max,padded_length,unique_chains):

    try:
        neighborhoods = get_neighborhoods_from_protein(np_protein,r_max,uc=unique_chains)
        padded_neighborhoods = pad_neighborhoods(neighborhoods,padded_length=padded_length)
        del neighborhoods
    except Exception as e:
        print(e)
        print('Error with',np_protein[0])
        #print(traceback.format_exc())
        return (None,)
    #padded_neighborhoods = None
 


    
    return (padded_neighborhoods)

def get_neighborhoods(
        hdf5_in,
        protein_list,
        num_nhs,
        r_max,
        hdf5_out,
        unique_chains,
        parallelism
):

    # get metadata
    metadata = get_metadata()


    logging.basicConfig(level=logging.DEBUG)
    ds = PDBPreprocessor(hdf5_in,protein_list)
    bad_neighborhoods = []
    n = 0


    max_atoms = 1300
    dt = np.dtype([
        ('res_id','S5',(6)),
        ('atom_names', 'S4', (max_atoms)),
        ('elements', 'S1', (max_atoms)),
        ('res_ids', 'S5', (max_atoms,6)),
        ('coords', 'f8', (max_atoms,3)),
        ('SASAs', 'f8', (max_atoms)),
        ('charges', 'f8', (max_atoms)),
    ])
    print(dt)
    print(num_nhs)
    print('writing hdf5 file')
    with h5py.File(hdf5_out,'w') as f:
        f.create_dataset(protein_list,
                         shape=(num_nhs,),
                         dtype=dt)
    print('calling parallel process')
    print('Value of unique_chains = ',unique_chains)
    nhs = np.empty(shape=num_nhs,dtype=('S5',(6)))
    with Bar('Processing', max = ds.count(), suffix='%(percent).1f%%') as bar:
        with h5py.File(hdf5_out,'r+') as f:
            for i,neighborhoods in enumerate(ds.execute(
                    c,
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
                    #n+=1
                    continue
                
                
                neighborhoods_per_protein = neighborhoods.shape[0]
                
                f[protein_list][n:n+neighborhoods_per_protein] = neighborhoods
                nhs[n:n+neighborhoods_per_protein] = neighborhoods['res_id']
                n+=neighborhoods_per_protein
                
                del neighborhoods
                bar.next()

                
    print(len(nhs))
    with h5py.File(hdf5_out,'r+') as f:
        f.create_dataset('nh_list',
                         data=nhs)
    
    print('Done with parallel computing')


def main():
    parser = ArgumentParser()
    parser.add_argument('--hdf5_out', dest='hdf5_out', type=str, help='ouptut hdf5 filename', required=True)
    parser.add_argument('--protein_list', dest='protein_list', type=str, help='protein list within hdf5_in file', required=True)
    parser.add_argument('--parallelism', dest='parallelism', type=int, help='ouptput file name', default=4)
    parser.add_argument('--hdf5_in', dest='hdf5_in', type=str, help='hdf5 filename', default=False)
    parser.add_argument('--num_nhs', dest='num_nhs', type=int, help='number of neighborhoods in protein set')
    parser.add_argument('--r_max', dest='r_max', type=float, help='radius of neighborhood')
    parser.add_argument('--unique_chains', dest='unique_chains', action='store_true',default=False,help='Only take one neighborhood per residue per unique chain')
    
    args = parser.parse_args()

    print('First value of unique_chains',args.unique_chains)
    get_neighborhoods(
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
