import coordinates as co
import Bio.PDB as pdb
import protein
import numpy as np

print('Program starting')

parser = pdb.PDBParser(QUIET=True)

struct = parser.get_structure(
    '1PGA',
    '/gscratch/stf/mpun/data/1PGA.pdb'
)

print('Testing get_coords() on 1PGA')
atom_list = pdb.Selection.unfold_entities(
    struct,
    'A'
)


coords = co.get_coords(
    atom_list,
    [0.,0.,0.],
    co.el_channel,
    6,
    protein.ch_to_ind_encoding
)

print('Number of points in each channel:')
print([np.array(x).shape for x in coords[0]])
print('\n')

print('Testing get_res_neighbor_atomic_coords on 1PGA')
res_list = pdb.Selection.unfold_entities(struct,'R')
coords,SASA_weights = co.get_res_neighbor_atomic_coords(
    res_list[0],
    10.,
    struct
)
print('Shape of coords in each channels')
print([np.array(x).shape for x in coords[0]])
print('\nSASA weights in SASA channel')
print(SASA_weights)
print('\nNumber of points with associated SASA weights: ',len(SASA_weights))


print('Terminating')
