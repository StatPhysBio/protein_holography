#
# File to select proteins from pdb files and store them in hdf5 file
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

args = parser.parse_args()
metadata = get_metadata()

for i,x in enumerate(args.img):
    args.img[i] = x.encode()

# get list of pdbs from hdf5 file
f = h5py.File(args.plf,'r+') 
pdb_list = np.array(f['pdb_list'])
pdb_metadata = dict(f['pdb_metadata'])

good_pdbs = []

for pdb,res,img in zip(
        pdb_list,
        pdb_metadata['resolution'],
        pdb_metadata['structure_method']
):
    if img not in args.img:
        continue
    if res > args.res:
        continue
    good_pdbs.append(pdb)

# record protein list that meets specifications
dset = f.create_dataset(
    'pdb_subsets/img={}_max_res={}/list'.format(
        '+'.join(map(lambda x : x.decode('utf-8'),args.img))
        ,args.res),
    data=good_pdbs
)
record_metadata(metadata,dset)

# close hdf5 file
f.close()
