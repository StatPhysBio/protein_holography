#
# File to get metadata from pdbs files and store them in the hdf5 file
#


from argparse import ArgumentParser
import sys
sys.path.append('/gscratch/spe/mpun/protein_holography/utils')
from posterity import get_metadata,record_metadata
import h5py
import Bio.PDB as pdb
import numpy as np
import Bio.PDB.mmtf as mmtf

parser = ArgumentParser()

parser.add_argument(
    '--pdb_list_file',
    dest='plf',
    type=str,
    help='hdf5 file with pdb list contained within'
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

pdb_metadata = {}
keys = ['deposition_date',
        'release_date',
        'structure_method',
        'resolution',
        'has_missing_residues',
        'missing_residues']
for k in keys:
    pdb_metadata[k] = []


# establish pdb parser
parser = pdb.PDBParser(QUIET=True)
#parser = mmtf.MMTFParser()

for i,pdb in enumerate(pdb_list):
    pdb = pdb.decode('utf-8')
    struct = parser.get_structure(pdb + 'struct',args.pdb_dir + pdb + '.pdb')
    #struct = parser.get_structure(args.pdb_dir + pdb + '.mmtf')

    if i % 1000 == 0:
        print(i)
    #print('{} -- {}'.format(struct.header['resolution'],len(struct.header['missing_residues'])))
    for k in pdb_metadata.keys():
        info = struct.header[k]
        if k == 'missing_residues':
            info = len(info)
        if type(info) == str:
            info = info.encode()
        if k == 'resolution' and info == None:
            info = 10.
        #print(k,info)
        pdb_metadata[k].append(info)

# check that all metadata is same length as pdb_list
pdb_list_len = len(pdb_list)
for k in pdb_metadata.keys():
    new_len = len(pdb_metadata[k])
    if new_len != pdb_list_len:
        print('{} metadata information not same length as pdb list'.format(new_len))

# get list of pdbs from hdf5 file
with h5py.File(args.plf,'r+') as f:
    for k in pdb_metadata.keys():
        try:
            del f['pdb_metadata/{}'.format(k)]
        except:
            print('Could not be deleted')
        dset = f.create_dataset('pdb_metadata/{}'.format(k),data=np.array(pdb_metadata[k]))
        record_metadata(metadata,dset)
