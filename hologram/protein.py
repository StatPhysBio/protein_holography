#
# Python module protein.py for holographic machine learning
#
# This module contains all structures relating to protein conventions including...
#  - dictionaries for translating amino acids to indices and vice-versa
#  - dictionaries for translating elements to indices and vice-versa
#  - dictionaries for amino acid information such as the number of atoms typically
#     associated with that amino acid in pdb structures
#


# dictionaries for converting amino acid to index and vice-versa
aa_to_ind = {'CYS': 2, 'ILE': 8, 'GLN': 12, 'VAL': 6, 'LYS': 13,
             'PRO': 4, 'GLY': 0, 'THR': 5, 'PHE': 16, 'GLU': 14,
             'HIS': 15, 'MET': 11, 'ASP': 7, 'LEU': 9, 'ARG': 17,
             'TRP': 19, 'ALA': 1, 'ASN': 10, 'TYR': 18, 'SER': 3}
ind_to_aa = {0: 'GLY', 1: 'ALA', 2: 'CYS', 3: 'SER', 4: 'PRO',
             5: 'THR', 6: 'VAL', 7: 'ASP', 8: 'ILE', 9: 'LEU',
             10: 'ASN', 11: 'MET', 12: 'GLN', 13: 'LYS', 14: 'GLU',
             15: 'HIS', 16: 'PHE', 17: 'ARG', 18: 'TYR', 19: 'TRP'}

# dictionaries to convert element to index
el_to_ind = {'C':0 , 'N':1, 'O':2, 'S':3}

# dictionaries for amino acid statistics
atoms_per_aa = {'CYS': 6, 'ASP': 8, 'SER': 6, 'GLN': 9, 'LYS': 9,
                'ASN': 8, 'PRO': 7, 'GLY': 4, 'THR': 7, 'PHE': 11,
                'ALA': 5, 'MET': 8, 'HIS': 10, 'ILE': 8, 'LEU': 8,
                'ARG': 11, 'TRP': 14, 'VAL': 7, 'GLU': 9, 'TYR': 12}

aas = aa_to_ind.keys()
els = el_to_ind.keys()
