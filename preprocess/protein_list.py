#
# protein_list.py -- Michael Pun -- 12 May 2021
# 
# Take a directory of pdb files as an argument and output an hdf5
# file with a list of the protein names as the dataset.
#
# This hdf5 file will serve as the foundation for the dataset
# and will later contain lists of high resolution structures
# and splits for training and test sets.
#

from argparse import ArgumentParser
import os
import h5py
import sys
sys.path.append('/gscratch/spe/mpun/protein_holography/utils')
from posterity import get_metadata,record_metadata


parser = ArgumentParser()

parser.add_argument(
    '--pdb_dir',
    dest='pdb_dir',
    type=str,
    help='Directory for pdb files'
)
parser.add_argument(
    '--filename',
    dest='filename',
    type=str,
    help='Name for the dataset'
)
parser.add_argument(
    '--data_dir',
    dest='data_dir',
    type=str,
    default='/gscratch/spe/mpun/protein_holography/data',
    help='Directory to save data'
)

args = parser.parse_args()

# get metadata
metadata = get_metadata()

# gather pdb names from pdb directory
pdb_files = [x for x in os.listdir(args.pdb_dir) if '.pdb' in x]

# get names from pdbs
pdb_names = [str.encode(x[:-4]) for x in pdb_files]

# filepath 
filepath = args.data_dir + '/' + args.filename

# write names to dataset file
try:
    with h5py.File(filepath + '.hdf5','w-') as f:
        dset = f.create_dataset('pdb_list',
                                data=pdb_names)
        record_metadata(metadata,dset)

except:
    with h5py.File(filepath + '.hdf5','r+') as f:
        try:
            dset = f.create_dataset('pdb_list',
                                    data=pdb_names)
            record_metadata(metadata,dset)
        except:
            print('Trying to create dataset that already exists')
    
