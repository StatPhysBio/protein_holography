import imp
import os

#os.chdir('/gscratch/spe/mpun/protein_holography/hologram')
import Bio.PDB as pdb

from protein_holography.hologram.protein_dir_parse import get_proteins_from_dir
from protein_holography.hologram.sample_protein import sample_amino_acid_from_protein as sample
import protein_holography.hologram.pdb_interface as pit
import protein_holography.utils.protein as protein

imp.reload(pit)
proteins = get_proteins_from_dir('/gscratch/stf/mpun/data/casp11/training30')
p = proteins[0]

parser = pdb.PDBParser(QUIET=True)

curr_struct = parser.get_structure(p + 'struct', p + '.pdb')



aa = sample(p,'GLY')

sph_c = pit.get_res_atomic_coords(aa)
                                
sph_nc = pit.get_res_neighbor_aa_coords(aa,5.,curr_struct)
