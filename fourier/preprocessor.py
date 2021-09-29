import pandas as pd
import signal
import numpy as np
import time
import os
import logging
import itertools
import functools
import warnings
from multiprocessing import Pool, TimeoutError
from Bio.PDB import PDBList
from Bio.PDB import PDBParser
from Bio.PDB.mmtf import MMTFParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import h5py

def load_data(nb):

    try:
        name = nb[1].decode('utf-8')
    except:
        print('Problematic namne ',nb[1])
    parser = PDBParser(QUIET=True)

    try:
        struct = parser.get_structure(name,
                                      '/gscratch/stf/mpun/data/casp11/training30/{}.pdb'.format(name))
        exists = True
    except:
        exists= False
    
    return exists


def process_data(nb):
    assert(process_data.callback)

    parser = MMTFParser()
    name = nb[1].decode('utf-8')
    nh = (int(nb[2].decode('utf-8')),
          nb[3].decode('utf-8'),
          (nb[4].decode('utf-8'),
          int(nb[5].decode('utf-8')),
          nb[6].decode('utf-8')))
    try:
        with h5py.File('/gscratch/spe/mpun/protein_holography/data/coordinates/1PGA_SASA.hdf5',
                       'r') as f:
            C_coords = np.array(f["{}/{}/{}/C".format(name,nh,10.)])
            N_coords = np.array(f["{}/{}/{}/N".format(name,nh,10.)])
            O_coords = np.array(f["{}/{}/{}/O".format(name,nh,10.)])
            S_coords = np.array(f["{}/{}/{}/S".format(name,nh,10.)])
            HOH_coords = np.array(f["{}/{}/{}/HOH".format(name,nh,10.)])
            SASA_coords = np.array(f["{}/{}/{}/SASA".format(name,nh,10.)])
            weights = np.array(f["{}/{}/{}/SASA_weights".format(name,nh,10.)])
            
    except Exception as e:
        print(e)
        print(nb)
        print('failed')
        return False
    coords = [C_coords,O_coords,N_coords,S_coords,HOH_coords,SASA_coords]


    return process_data.callback(coords, weights, nb, **process_data.params)



def initializer(init, callback, params, init_params):
    if init is not None:
        init(**init_params)
    process_data.callback = callback
    process_data.params = params
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class HDF5Preprocessor:
    def __init__(self, hdf5_file, nh_list, coord_file):
#    def __init__(self, path):
        #df = pd.read_table(path, header=None, names=["aa", "neighborhood", "extra"])
        with h5py.File(hdf5_file,'r') as f:
            nh_list = np.unique(np.array(f[nh_list]),axis=0)
            print(nh_list)
        self.__data = nh_list
        self.coord_file = coord_file
#        self.__data = pd.Series(nh_list,
#                                index=['aa',
#                                       'pdb',
#                                       'model',
#                                       'chain',
#                                       'hetero',
#                                       'seq id',
#                                       'insertion']
#                            )

        #print(df)
        #self.__data = df['neighborhood'].apply(lambda x: eval(x))

    def count(self):
        return self.__data.shape[0]

    def execute(self, callback, parallelism = 8, limit = None, params = None, init = None, init_params = None):
        if limit is None:
            data = self.__data
        else:
            data = self.__data[:limit]
        with Pool(initializer = initializer, processes=parallelism, initargs = (init, callback, params, init_params)) as pool:

            #all_loaded = functools.reduce(lambda x, y: x and y, pool.imap(load_data, data))
            all_loaded = True
            if all_loaded:
                # logging.info('All PDB files are loaded.')
                pass
            else:
                raise Exception("Some PDB files could not be loaded.")
            
            for coords in pool.imap(process_data, data):
                if coords:
                    yield coords

