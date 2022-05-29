import functools
import itertools
import logging
from multiprocessing import Pool, TimeoutError
import os
import signal
import sys
import time
import warnings
sys.path.append('/gscratch/stf/mpun/software/PyRosetta4.Release.python38.linux.release-299')

from Bio.PDB import PDBList, PDBParser
from Bio.PDB.mmtf import MMTFParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import h5py
import numpy as np
import pandas as pd
import pyrosetta
from pyrosetta.rosetta import core, protocols, numeric, basic, utility

init_flags = '-ignore_unrecognized_res 1 -include_current -ex1 -ex2 -mute all -ignore_zero_occupancy false -obey_ENDMDL 1'
pyrosetta.init(init_flags)

def process_data(pdb,pdb_dir):
    assert(process_data.callback)

    pdb = pdb.decode('utf-8')
    #pdb_file = '/gscratch/scrubbed/mpun/data/T4/pdbs/' + pdb + '.pdb'
    #pdb_file = '/gscratch/stf/mpun/data/casp12/pdbs/training_30/' + pdb + '.pdb'
    #pdb_file = '/gscratch/stf/mpun/data/casp12/pdbs/validation/' + pdb + '.pdb'
    #pdb_file = '/gscratch/scrubbed/mpun/data/DunhamBeltrao/pdbs/' + pdb + '.pdb'
    #pdb_file = '/gscratch/scrubbed/mpun/data/CoV2_ACE2/' + pdb + '.pdb'
    #pdb_file = '/gscratch/stf/mpun/data/TCRStructure/pdbs/' + pdb + '.pdb'
    #pdb_file = '/gscratch/stf/mpun/data/proteinG/' + pdb + '.pdb'
    pdb_file = pdb_dir + pdb + '.pdb'
    
    try:
        pose = pyrosetta.pose_from_pdb(pdb_file)
    except:
        print('Pose could ot be created for protein {}'.format(pdb))
        return process_data.callback(None,**process_data.params)
    #print('pdb is ',pdb,pose.pdb_info().name())
    return process_data.callback(pose, **process_data.params)

def initializer(init, callback, params, init_params):
    if init is not None:
        init(**init_params)
    process_data.callback = callback
    process_data.params = params
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class PDBPreprocessor:
    def __init__(self, hdf5_file, pdb_list, pdb_dir):
        with h5py.File(hdf5_file,'r') as f:
            pdb_list = np.array(f[pdb_list])

        self.pdb_dir = pdb_dir
        self.__data = pdb_list
        self.size = len(pdb_list)
    def count(self):
        return len(self.__data)

    def execute(self, callback, parallelism = 8, limit = None, params = None, init = None, init_params = None):
        if limit is None:
            data = self.__data
        else:
            data = self.__data[:limit]
        with Pool(initializer = initializer, processes=parallelism, initargs = (init, callback, params, init_params)) as pool:

    
            all_loaded = True
            if all_loaded:
                # logging.info('All PDB files are loaded.')
                pass
            else:
                raise Exception("Some PDB files could not be loaded.")
            process_data_pdbs = functools.partial(
                process_data,
                pdb_dir=self.pdb_dir
            )
            for res in pool.imap(process_data_pdbs, data):
                if res:
                    yield res

