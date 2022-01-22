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

def load_data(pdb):

    try:
        name = pdb.decode('utf-8')
    except:
        print('Problematic namne ',pdb[1])
    parser = PDBParser(QUIET=True)

    try:
        struct = parser.get_structure(name,
                                      '/gscratch/stf/mpun/data/casp12/pdbs/training_30/{}.pdb'.format(name))
        exists = True
    except:
        exists= False
    
    return exists


def process_data(pdb):
    assert(process_data.callback)
    
    #parser = MMTFParser()
    parser = PDBParser()
    try:
        name = pdb.decode('utf-8')
    except Exception as e:
        print(e,pdb)
    try:
        with warnings.catch_warnings(): 
            warnings.simplefilter('ignore', PDBConstructionWarning)
            struct = parser.get_structure('{}'.format(name),
                                          '/gscratch/stf/mpun/data/{}.pdb'.format(name))
#                                          '/gscratch/stf/mpun/data/casp12/pdbs/training_30/{}.pdb'.format(name))
#                                          '/gscratch/stf/mpun/data/TCRStructure/pdbs/{}.pdb'.format(name))
            #struct = parser.get_structure('/gscratch/stf/mpun/data/{}.mmtf'.format(name))
    except Exception as e: 
        print(e)
        print(pdb)
#    except:
#        print(pdb)
#        print('failed')
        return False
    
    return process_data.callback(struct, **process_data.params)

def initializer(init, callback, params, init_params):
    if init is not None:
        init(**init_params)
    process_data.callback = callback
    process_data.params = params
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class PDBPreprocessor:
    def __init__(self, hdf5_file, nh_list):

        with h5py.File(hdf5_file,'r') as f:
            nh_list = np.array(f[nh_list])
        self.__data = nh_list

    def count(self):
        return self.__data.shape[0]

    def execute(self, callback, parallelism = 8, limit = None, params = None, init = None, init_params = None):
        if limit is None:
            data = self.__data
        else:
            data = self.__data[:limit]
        with Pool(initializer = initializer, processes=parallelism, initargs = (init, callback, params, init_params)) as pool:

            all_loaded = True
            if all_loaded:
                pass
            else:
                raise Exception("Some PDB files could not be loaded.")

            for res in pool.imap(process_data, data):
                if res:
                    yield res

