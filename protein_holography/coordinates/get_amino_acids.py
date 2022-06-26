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

from pyrosetta_hdf5_amino_acids import get_neighborhoods_from_protein,pad_neighborhoods
from preprocessor_hdf5_proteins import PDBPreprocessor
from argparse import ArgumentParser
import numpy as np
import h5py
import sys
sys.path.append('/gscratch/spe/mpun/protein_holography/utils')
from posterity import get_metadata,record_metadata
import logging
from progress.bar import Bar
import traceback

def c(np_protein):

    try:
        neighborhoods = get_neighborhoods_from_protein(np_protein)
        padded_neighborhoods = pad_neighborhoods(neighborhoods,padded_length=50)
    except Exception as e:
        print(e)
        print('Error with',np_protein[0])
        #print(traceback.format_exc())
        return (None,)

 


    
    return (padded_neighborhoods)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--hdf5_out', dest='hdf5_out', type=str, help='ouptut hdf5 filename', required=True)
    parser.add_argument('--protein_list', dest='protein_list', type=str, help='protein list within hdf5_in file', required=True)
    parser.add_argument('--parallelism', dest='parallelism', type=int, help='ouptput file name', default=4)
    parser.add_argument('--hdf5_in', dest='hdf5_in', type=str, help='hdf5 filename', default=False)
    parser.add_argument('--num_nhs', dest='num_nhs', type=int, help='number of neighborhoods in protein set')
    
    args = parser.parse_args()
    
    # get metadata
    metadata = get_metadata()

    
    logging.basicConfig(level=logging.DEBUG)
    ds = PDBPreprocessor(args.hdf5_in,args.protein_list)
    bad_neighborhoods = []
    n = 0


    max_atoms = 50
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
    print(args.num_nhs)
    print('writing hdf5 file')
    with h5py.File(args.hdf5_out,'w') as f:
        f.create_dataset(args.protein_list,
                         shape=(args.num_nhs,),
                         dtype=dt)
    print('calling parallel process')
    nhs = np.empty(shape=args.num_nhs,dtype=('S5',(6)))
    with Bar('Processing', max = ds.count(), suffix='%(percent).1f%%') as bar:
        with h5py.File(args.hdf5_out,'r+') as f:
            for i,neighborhoods in enumerate(ds.execute(
                    c,
                    limit = None,
                    params = {},
                    parallelism = args.parallelism)):
                if neighborhoods[0] is None:
                    bar.next()
                    #n+=1
                    continue
                
                #print([neighborhoods[j][0] for j in range(10)])
                for neighborhood in neighborhoods:
                    nhs[n] = neighborhood[0]
                    f[args.protein_list][n] = (*neighborhood,)
                    n+=1
                #print(neighborhoods[0][0])
                #print('done writing. \n moving to next entry')
                bar.next()

    print(len(nhs))
    with h5py.File(args.hdf5_out,'r+') as f:
        f.create_dataset('nh_list',
                         data=nhs)
    
    print('Done with parallel computing')
