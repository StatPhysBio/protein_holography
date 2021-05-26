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

from preprocessor import PDBPreprocessor
import pandas as pd
import numpy as np
import logging
import itertools
import os
from Bio.PDB import Selection, NeighborSearch
from argparse import ArgumentParser
from progress.bar import Bar
import coordinates as co
import h5py
import sys
sys.path.append('/gscratch/spe/mpun/protein_holography/utils')
from posterity import get_metadata,record_metadata
import protein


def c(struct, res, d, easy, l_max):

    EL_CHANNEL_NUM = 4
    DIMENSION = 3
    ca_coord = res['CA'].get_coord()
    model_tag = res.get_full_id()[1]
    model = struct[model_tag]

    if easy:
        curr_coords = co.get_res_atomic_coords(res,COA=True,ignore_alpha=False)
    else:
        curr_coords = [[[] for i in range(EL_CHANNEL_NUM)] for j in range(DIMENSION)]
    if d > 0:
        neighbor_coords = co.get_res_neighbor_atomic_coords(res,d,struct)

        for i in range(DIMENSION):
            for j in range(EL_CHANNEL_NUM):
                curr_coords[i][j] = curr_coords[i][j] + neighbor_coords[i][j]


    curr_r,curr_t,curr_p = curr_coords
    full_id = res.get_full_id()

    return res.get_resname(), full_id, curr_coords


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--output', dest='output', type=str, help='ouptput file name', required=True)
    parser.add_argument('--dataset', dest='dataset', type=str, help='dataset file name', required=True)
    parser.add_argument('--parallelism', dest='parallelism', type=int, help='ouptput file name', default=4)
    parser.add_argument('--invariants', dest='invariants', type=str, help='invariants file name', default='curated_invariants.txt')
    parser.add_argument('-d', dest='d', type=float, help='radius', default=5.0)

    parser.add_argument('--easy', action='store_true', help='easy', default=False)
    parser.add_argument('--hdf5', dest='hdf5', type=str, help='hdf5 filename', default=False)
    
    args = parser.parse_args()
    
    # get metadata
    metadata = get_metadata()

    
    logging.basicConfig(level=logging.DEBUG)
    ds = PDBPreprocessor(args.hdf5,args.dataset)
 
    n = 0
    with Bar('Processing', max = ds.count(), suffix='%(percent).1f%%') as bar:
        for res, fid, invs in ds.execute(
                c, limit = None, params = {'d': args.d, 'easy' : args.easy, 'l_max': 5}, 
                parallelism = args.parallelism):
            for i in range(4):
                with h5py.File('/gscratch/spe/mpun/protein_holography/data/coordinates/casp11_training30.hdf5','r+') as f:
                    try:
                        coords = np.array([x[i] for x in invs])
                        print(coords)
                        pdb_name = fid[0]
                        res_id = (fid[1],fid[2],fid[3])
                        ch = protein.ind_to_el[i]
                        dset = f.create_dataset('{}/{}/{}/{}'.format(
                            pdb_name,
                            res_id,
                            args.d,
                            ch
                        )
                                                ,data=coords) 
                        record_metadata(metadata,dset)
                    except:
                        print("Unexpected error:", sys.exc_info()[0])
                        
                    
                    #o.write(str(n) + ',' + res + ',' + str(fid)+ ',' + str(i) +',{},{},{}'.format(invs[0][i][j],invs[1][i][j],invs[2][i][j]) + '\n')
            n+=1
            bar.next()
