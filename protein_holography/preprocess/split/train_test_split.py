#
# File to split a protein list into training, validation, and testing data
#

from argparse import ArgumentParser
import sys


import Bio.PDB as pdb
import Bio.PDB.mmtf as mmtf
import h5py
import numpy as np

from protein_holography.utils.posterity import get_metadata, record_metadata

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

x_train = args.x_train
x_val = args.x_val
x_test = args.x_test

f = h5py.File(args.plf,'r+')

name = 'img={}_max_res={}'.format(
    '+'.join(args.img)
    ,args.res)

pdb_list = np.array(f['pdb_subsets/{}/list'.format(name)])
print(pdb_list)
np.random.shuffle(pdb_list)
N = len(pdb_list)
n_train = int(x_train*N)
n_val = int((x_train + x_val)*N)

train_pdbs = pdb_list[:n_train]
val_pdbs = pdb_list[n_train:n_val]
test_pdbs = pdb_list[n_val:]
try:
    g = f.create_group('pdb_subsets/{}/split_{}_{}_{}'.format(name,
                                                              x_train,
                                                              x_val,
                                                              x_test))
except:
    g = f['pdb_subsets/{}/split'.format(name)]
    print('Could not create group pdb_subsets/{}/split_{}_{}_{}'.format(name,
                                                                        x_train,
                                                                        x_val,
                                                                        x_test))
record_metadata(metadata,g)
dsets = []
dsets.append(
    f.create_dataset(
        'pdb_subsets/{}/split_{}_{}_{}/train/pdbs'.format(
            name,
            args.x_train,
            args.x_val,
            args.x_test
        ),
        data=train_pdbs
    )
)
dsets.append(
    f.create_dataset(
        'pdb_subsets/{}/split_{}_{}_{}/val/pdbs'.format(
            name,
            args.x_train,
            args.x_val,
            args.x_test
        ),
        data=val_pdbs
    )
)
dsets.append(
    f.create_dataset(
        'pdb_subsets/{}/split_{}_{}_{}/test/pdbs'.format(
            name,
            args.x_train,
            args.x_val,
            args.x_test
        ),
        data=test_pdbs
    )
)

for dset in dsets:
    record_metadata(metadata,dset)


f.close()
