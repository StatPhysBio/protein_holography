#
# File to split a protein list into training, validation, and testing data
#
    
from argparse import ArgumentParser
import sys

import Bio.PDB as pdb
import Bio.PDB.mmtf as mmtf
import h5py
import numpy as np

from protein_holography.utils.posterity import get_metadata,record_metadata
import protein_holography.preprocess.sample.sample_functions as sf

parser = ArgumentParser()

parser.add_argument(
    '--pdb_list_file',
    dest='plf',
    type=str,
    help='hdf5 file with pdb list contained within'
)
parser.add_argument(
    '--resolution',
    dest='res',
    type=float,
    help='Resolution cutoff for structures'
)
parser.add_argument(
    '--image_type',
    dest='img',
    type=str,
    nargs='+',
    help='image types allowed'
)
parser.add_argument(
    '--pdb_dir',
    dest='pdb_dir',
    type=str,
    help='Directory for pdb files'
)
parser.add_argument(
    '--train_frac',
    dest='x_train',
    type=float,
    help='fraction of training proteins'
)
parser.add_argument(
    '--val_frac',
    dest='x_val',
    type=float,
    help='fraction of validation proteins'
)
parser.add_argument(
    '--test_frac',
    dest='x_test',
    type=float,
    help='fraction of testing proteins'
) 

args = parser.parse_args()
metadata = get_metadata()

f = h5py.File(args.plf,'r+')

name = 'img={}_max_res={}'.format(
    '+'.join(args.img)
    ,args.res)

num_sample = {
    'train': 10000,
    'val': 128,
    'test': 128
}

for x in ['train','val','test']:
    print('pdb_subsets/{}/split_{}_{}_{}/{}/pdbs'.format(name,
                                                                args.x_train,
                                                                args.x_val,
                                                                args.x_test,
                                                                x))
    pdb_list = f['pdb_subsets/{}/split_{}_{}_{}/{}/pdbs'.format(name,
                                                                args.x_train,
                                                                args.x_val,
                                                                args.x_test,
                                                                x)]
    pdb_list = [x.decode('utf-8') for x in pdb_list]
    print(pdb_list[:10])
    sample = sf.sample_equally(
        pdb_list,
        args.pdb_dir,
        num_sample[x]
    )
    dset = f.create_dataset(
        'pdb_subsets/{}/split_{}_{}_{}/{}/neighborhoods_equally_sampled_e={}'.format(name,
                                                                      args.x_train,
                                                                      args.x_val,
                                                                      args.x_test,
                                                                      x,num_sample[x]),
        data=sample
    )


f.close()
