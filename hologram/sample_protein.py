#
# module for sampling residues from proteins
#

import Bio.PDB as pdb
import numpy as np

# function to sample a random residue of a given amino acid type from a protein structure
def sample_amino_acid_from_protein(protein,aa):
    parser = pdb.PDBParser(QUIET=True)
    struct = parser.get_structure(protein + 'struct', protein + '.pdb')
    residues = [x for x in pdb.Selection.unfold_entities(struct,'R') if (x.resname == aa and 'CA' in x)]

    # if the given amino acid is not found print an error and return None
    if len(residues) == 0:
        print('ERROR: Amino acid ' + str(aa) + ' not found in protein ' + str(protein))
        return None

    # sample a random residue from the residues found
    np.random.shuffle(residues)
    return residues[0]
