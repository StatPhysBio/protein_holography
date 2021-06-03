#
# filer_pdbs.py -- Michael Pun -- 12 May 2021
#
# Take a pdb list and produce subset of pdbs that meet the given
# conditions of the pdb files.
#

from argparse import ArgumentParser
import sys
sys.path.append('/gscratch/spe/mpun/protein_holography/utils')
from posterity import get_metadata,record_metadata
import h5py
import Bio.PDB as pdb
import numpy as np

parser = ArgumentParser()

parser.add_argument(
    '--pdb_list_file',
    dest='plf',
    type=str,
    help='hdf5 file with pdb list contained within'
)
parser.add_argument(
    '--resolution',
    dest='resolution',
    type=float,
    default=2.5,
    help='Resolution cutoff for pdb files'
)
parser.add_argument(
    '--pdb_dir',
    dest='pdb_dir',
    type=str,
    help='Directory for pdb files'
)


args = parser.parse_args()
metadata = get_metadata()

# get list of pdbs from hdf5 file
with h5py.File(args.plf,'r') as f:
    pdb_list = np.array(f['pdb_list'])

filtered_pdbs = []

# establish pdb parser
parser = pdb.PDBParser(QUIET=True)

for pdb in pdb_list[:100]:
    pdb = pdb.decode('utf-8')
    struct = parser.get_structure(pdb + 'struct',args.pdb_dir + pdb + '.pdb')
    curr_reso = struct.header['resolution']
    if curr_reso == None:
        continue
    if curr_reso > args.resolution:
        continue

    filtered_pdbs.append(pdb)


print(len(pdb_list))
print(len(filtered_pdbs))
