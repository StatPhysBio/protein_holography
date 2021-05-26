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
    try:
        with warnings.catch_warnings(): 
            warnings.simplefilter('ignore', PDBConstructionWarning)
            struct = parser.get_structure('/gscratch/stf/mpun/data/casp11/training30/{}.mmtf'.format(name))
    except:
        print('failed')
        return False
    
    res = struct[int(nb[2].decode('utf-8'))][nb[3].decode('utf-8')][int(nb[5].decode('utf-8'))]
    return process_data.callback(struct, res, **process_data.params)

def initializer(init, callback, params, init_params):
    if init is not None:
        init(**init_params)
    process_data.callback = callback
    process_data.params = params
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class PDBPreprocessor:
    def __init__(self, hdf5_file, nh_list):
#    def __init__(self, path):
        #df = pd.read_table(path, header=None, names=["aa", "neighborhood", "extra"])
        with h5py.File(hdf5_file,'r') as f:
            nh_list = np.array(f[nh_list])
        self.__data = nh_list
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

            all_loaded = functools.reduce(lambda x, y: x and y, pool.imap(load_data, data))
            all_loaded = True
            if all_loaded:
                # logging.info('All PDB files are loaded.')
                pass
            else:
                raise Exception("Some PDB files could not be loaded.")

            for res in pool.imap(process_data, data):
                if res:
                    yield res

