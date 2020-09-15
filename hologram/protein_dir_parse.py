#
# Module for retrieving protein names from pdb files in a given directory
#

import os

# get all protein names from the pdb files in a given directory
def get_proteins_from_dir(protein_dir):

    files = os.listdir(protein_dir)
    proteins = [x[:-4] for x in files if 'pdb' in x[-3:]]

    return proteins


