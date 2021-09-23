#
# This file takes in a list of pdbs and retrieves all residues from them.
# It then outputs a list of all residues into an hdf5 file in the format
#   (<aa>, <pdb>, <model>, <chain>, <insertion>, <seq id>, <hetero>)
#

from preprocessor import PDBPreprocessor
from argparse import ArgumentParser
import logging
import sys
sys.path.append('/gscratch/spe/mpun/protein_holography/utils')
from posterity import get_metadata,record_metadata
from progress.bar import Bar
import protein
import h5py
import numpy as np

# function to be applied to each pdb in list
# take a pdb and retrieve all residues from it
def c(struct):
    
    # only use model 0
    model = struct[0]

    res_ids = []
    for chain in model:
        for res in chain:
            if 'CA' not in res:
                continue
            resname = res.get_resname()
            if resname not in protein.aa_to_ind.keys():
                continue
            new_res_id = []
            full_id = res.get_full_id()
            new_res_id.append(resname.encode())
            new_res_id.append(full_id[0].encode())
            new_res_id.append(full_id[1])
            new_res_id.append(full_id[2].encode())
            new_res_id.append(full_id[3][0].encode())
            new_res_id.append(full_id[3][1])
            new_res_id.append(full_id[3][2].encode())
            res_ids.append(new_res_id)
            

    return res_ids

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '--hdf5',
        dest='hdf5',
        type=str,
        help='hdf5 file for pdb list to sample from',
        default=None
    )
    parser.add_argument(
        '--dataset',
        dest='dataset',
        type=str,
        help='dataset within hdf5 to sample from',
        default=None
    )
    parser.add_argument(
        '--parallelism',
        dest='parallelism',
        type=int,
        help='number threads',
        default=4
    )

    args = parser.parse_args()

    # get metadata for the recording
    metadata = get_metadata()

    logging.basicConfig(level=logging.DEBUG)
    # load the dataset from the hdf5 file
    ds = PDBPreprocessor(args.hdf5,args.dataset)
    master_list = []    
    with Bar('Processing', max = ds.count(), suffix='%(percent).1f%%') as bar:
        for res_list in ds.execute(
                c, limit = None, params = {}, 
                parallelism = args.parallelism):

            for res in res_list:
                master_list.append(res)
            
                
            bar.next()
    master_list = np.array(master_list)
    print(master_list[:10])
    with h5py.File(args.hdf5,'r+') as f:
        ds = f.create_dataset(args.dataset[:-4] + '/complete_sample_1',
                              data = master_list)

    print(len(master_list))
    print('Terminated succesfully')
