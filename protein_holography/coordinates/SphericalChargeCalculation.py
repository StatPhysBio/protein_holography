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
import itertools
import logging
import os
import sys

from Bio.PDB import Selection, NeighborSearch
import h5py
import numpy as np
import pandas as pd
from progress.bar import Bar

import protein_holography.coordinates.charge as charge
import protein_holography.coordinates.coordinates as co
from protein_holography.coordinates.preprocessorRosetta import PDBPreprocessor
from protein_holography.utils.posterity import get_metadata, record_metadata
import protein_holography.utils.protein as protein # Is this even used?

def c(nb_list, pose, d, easy, COA=False):
    if pose is None:
        return None,nb_list,None,None
    #print('\n\n current struct is {}'.format(struct))
    full_id_list = []
    charges_list = []
    charge_coords_list = []
    #print('c called')
    #print(len(nb_list))
    i = 0
    for nb in nb_list:
        #print(nb)
        i += 1
        res = (
            nb[3].decode('utf-8'),
            int(nb[5].decode('utf-8')),
            nb[6].decode('utf-8')
        )

        #print(nb)
        if d > 0:
            try:
                #neighbor_coords,SASA_weights = co.get_res_neighbor_atomic_coords(res,d,struct)
                #print('Getting charge')
                charge_coords,charges = charge.get_res_neighbor_charge_coords(res,pose,d)
                
            except Exception as e:
                print(e)
                print(nb,' returned exception')
                #print(e)
                #print('Error: get_res_neighbor_atomic_coords error\n'
                #      'Exception raised in neighbor coord gathering')
                #print('The problematic struct is: {}'.format(struct))
                p = 0
                charge_coords,charges = (None,None)
        #curr_r,curr_t,curr_p = curr_coords
        full_id = (
            nb[1].decode('utf-8'), # pdb
            int(nb[2].decode('utf-8')), # model number
            nb[3].decode('utf-8'), # chain
            (
                nb[4].decode('utf-8'), # hetero tag
                int(nb[5].decode('utf-8')), # id num
                nb[6].decode('utf-8') # insertion code
            )
        )
        charges_list.append(charges)
        charge_coords_list.append(charge_coords)
        full_id_list.append(full_id)
    return full_id_list, nb_list, charge_coords_list, charges_list


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--output', dest='output', type=str, help='ouptut hdf5 filename', required=True)
    parser.add_argument('--dataset', dest='dataset', type=str, help='dataset file name', required=True)
    parser.add_argument('--parallelism', dest='parallelism', type=int, help='ouptput file name', default=4)
    parser.add_argument('-d', dest='d', type=float, help='radius', default=5.0)
    parser.add_argument('--easy', action='store_true', help='easy', default=False)
    parser.add_argument('--COA', action='store_true', help='aligned axes', default=False)
    parser.add_argument('--hdf5', dest='hdf5', type=str, help='hdf5 filename', default=False)
    
    args = parser.parse_args()
    
    # get metadata
    metadata = get_metadata()

    
    logging.basicConfig(level=logging.DEBUG)
    ds = PDBPreprocessor(args.hdf5,args.dataset)
    bad_neighborhoods = []
    n = 0
    with Bar('Processing', max = ds.count(), suffix='%(percent).1f%%') as bar:
        with h5py.File('/gscratch/spe/mpun/protein_holography/data/coordinates/' +
                       args.output,'r+') as f:
            for fid_list, nb_list, coords_list, charges_list in ds.execute(
                    c, limit = None, params = {'d': args.d, 'easy' : args.easy, 'COA' : args.COA}, 
                    parallelism = args.parallelism):

                print(nb_list[0][1],' complete')
                if fid_list is None:
                    bar.next()
                    continue
                for fid,nb,coords,charges in zip(fid_list,nb_list,coords_list,charges_list):
                    #print(fid,nb,coords.shape,charges.shape)
                    if coords is None:
                        continue
                    try:
                        pdb_name = fid[0]
                        res_id = (fid[1],fid[2],fid[3])    
                        dset = f.create_dataset('{}/{}/{}/{}'.format(
                            pdb_name,
                            res_id,
                            args.d,
                            'charges'
                        )
                                                ,data=charges) 
                        record_metadata(metadata,dset)
                    except Exception as e:
                        i=1
                        #print(e)
                        #except:
                        #print("Unexpected error:", sys.exc_info()[0])
                
                
                    try:
                        pdb_name = fid[0]
                        res_id = (fid[1],fid[2],fid[3])
                        dset = f.create_dataset('{}/{}/{}/{}'.format(
                            pdb_name,
                            res_id,
                            args.d,
                            'charge_coords'
                        )
                                                ,data=coords) 
                        record_metadata(metadata,dset)
                    except Exception as e:
                        i=1
                        #print(e)
                        #except:
                        #print("Unexpected error:", sys.exc_info()[0])
                
                
                        #o.write(str(n) + ',' + res + ',' + str(fid)+ ',' + str(i) +',{},{},{}'.format(invs[0][i][j],invs[1][i][j],invs[2][i][j]) + '\n')
                n+=1
                bar.next()
    print('Bad Neighborhoods')
    #print(bad_neighborhoods)
    np.save('bad_neighborhoods.npy',
            bad_neighborhoods,
            allow_pickle=True)
    
