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
import sys
sys.path.append('/gscratch/stf/mpun/software/PyRosetta4.Release.python38.linux.release-299')
import pyrosetta
from pyrosetta.rosetta import core, protocols, numeric, basic, utility

init_flags = '-ignore_unrecognized_res 1 -include_current -ex1 -ex2 -mute all -ignore_zero_occupancy false'
pyrosetta.init(init_flags)

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


def process_data(nb_list):
    assert(process_data.callback)

    pdb = nb_list[0][1].decode('utf-8')
    print(nb_list[0][1])
    
    #pdb_file = '/gscratch/stf/mpun/data/' + pdb + '.pdb'
    #pdb_file = '/gscratch/stf/mpun/data/TCRStructure/pdbs/' + pdb + '.pdb'
    pdb_file = '/gscratch/stf/mpun/data/casp12/pdbs/validation/' + pdb + '.pdb'
    try:
        pose = pyrosetta.pose_from_pdb(pdb_file)
    except:
        print('Pose could ot be created for protein {}'.format(pdb))
        return process_data.callback(nb_list,None,**process_data.params)
    return process_data.callback(nb_list, pose, **process_data.params)

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
        nh_lists = []

        unique_pdbs = np.unique(nh_list[:,1])
        for pdb in unique_pdbs:
            nh_lists.append(nh_list[nh_list[:,1] == pdb])
        self.__data = nh_lists
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
        return len(self.__data)

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

            for res in pool.imap(process_data, data):
                if res:
                    yield res

