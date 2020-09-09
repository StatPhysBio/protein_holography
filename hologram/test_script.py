


import os

os.chdir('/gscratch/spe/mpun/protein_holography/hologram')

import protein
from protein_dir_parse import get_proteins_from_dir
from sample_protein import sample_amino_acid_from_protein as sample 
import pdb_interface as pit
import Bio.PDB as pdb
import imp
imp.reload(pit)
proteins = get_proteins_from_dir('/gscratch/stf/mpun/data/casp11/training30')
p = proteins[0]

parser = pdb.PDBParser(QUIET=True)

curr_struct = parser.get_structure(p + 'struct', p + '.pdb')



aa = sample(p,'GLY')

sph_c = pit.get_res_atomic_coords(aa)
                                
sph_nc = pit.get_res_neighbor_aa_coords(aa,5.,curr_struct)
