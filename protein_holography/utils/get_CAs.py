import sys

import h5py
import hdf5plugin
import numpy as np


def printname(x):
    print(x)

def main():

    
    with h5py.File(sys.argv[1],'r') as f:
        total_cas = 0
        for protein in f[sys.argv[2]]:
            total_cas += np.count_nonzero(protein['atom_names'] == b' CA ')
    sys.exit(str(total_cas))
    

if __name__ == "__main__":
    main()
