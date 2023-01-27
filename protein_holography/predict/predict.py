""" predict and write pseudoenergies for a set of proteins """

from argparse import ArgumentParser
from functools import partial
import logging
import os
from typing import List

import h5py
import numpy as np
import pandas as pd
from protein_holography.coordinates.get_neighborhoods import (
    get_padded_neighborhoods)
from protein_holography.coordinates.get_zernikegrams import (
    get_zernikegrams)
from protein_holography.network import network_loader
from protein_holography.utils.protein import (
    ol_to_ind_size, ind_to_ol_size)
from protein_holography.utils.pyrosetta_utils import c_struct
from scipy.special import softmax
from tensorflow.data import Dataset

# global constants for loading network weights
saved_network_dir = (
    '../model_weights/best_network/')

saved_network = (
        'bsize=256_ch=chain_connection=full_d=10.0_dropout_rate=0.000549'
        '_hdim=14_k=0j+20+0j_l=5_learnrate=0.001_n_dense=2_netL=5_nlayers=4'
        '_opt=Adam_proj=zgram_reg_strength=1.2e-16_rmax=10.0_scale=1.0'
)

def tf_ds(
    zgrams: np.ndarray,
    nh_list: np.ndarray,
    L_max: int=5,    
):
    aas = np.array(nh_list[:,0], dtype=str)
    labels = [ol_to_ind_size[x] for x in aas]
    one_hots = np.zeros(shape=(aas.shape[0],20))
    one_hots[np.arange(aas.shape[0]),labels] = 1

    zgrams_dict = {l: zgrams[str(l)] for l in range(L_max + 1)}

    return Dataset.from_tensor_slices((zgrams_dict, one_hots))
    

def tf_ds_from_hdf5(
    zgram_hdf5: str,
    dataset: str,
    L_max: int=5,
):
    """ Get tensorflow dataset from hdf5 file """
    with h5py.File(zgram_hdf5, 'r') as f:
        zgrams = f[dataset][:]
        nh_list = f['nh_list'][:]
    return nh_list, tf_ds(zgrams, nh_list, L_max)

def predict_from_zernikegrams(
    zgram_hdf5: str,
    dataset: str,
    network_dir: str,
    network_name: str,
    batchsize: int=64
):
    """ 
    Predict on zernikegrams 
    
    Useful for large datasets or for running on different options
    """
    load_network_from_weights = network_loader.load_network_from_weights
    network = load_network_from_weights(network_name)
    network.load_weights(
        os.path.join(network_dir,network_name)
    )

    nh_list, ds = tf_ds_from_hdf5(zgram_hdf5, dataset)
    
    pseudoenergies = network.predict(ds.batch(batchsize))

    return pseudoenergies, nh_list
        

def write_csv(
    filename: str,
    pseudoenergies: np.ndarray,
    nh_list: np.ndarray,
):
    """ Write csv of pseudoenrgies in standard format """
    df = pd.DataFrame(
        np.array(nh_list,dtype=str),
        columns=[
            'amino acid','pdb','chain',
            'site','icode','secondary structure'
        ]
    )
    df[[f'prob_{ind_to_ol_size[i]}'
        for i in range(20)]] = softmax(pseudoenergies,axis=-1)
    df[[f'pseudoenergy_{ind_to_ol_size[i]}'
        for i in range(20)]] = pseudoenergies

    df.to_csv(filename)
    
    return df

def make_structural_info(
    pose,
    max_atoms: int=15000,
) -> np.ndarray:
    """ Make structural info from pyrosetta pose """
    dt = np.dtype([
            ('pdb','S4',()),
            ('atom_names', 'S4', (max_atoms)),
            ('elements', 'S1', (max_atoms)),
            ('res_ids', 'S5', (max_atoms,6)),
            ('coords', 'f8', (max_atoms,3)),
            ('SASAs', 'f8', (max_atoms)),
            ('charges', 'f8', (max_atoms)),
        ])
    np_protein = np.zeros(shape=(1),dtype=dt) 

    si = c_struct(pose,padded_length=max_atoms)
    np_protein[0] = (*si,)

    return np_protein


def make_predictions_from_pdb(
    pdbfile: str,
    network_dir: str,
    network_name: str,
    batchsize: int=64,
    L_max: int=5,
):
    load_network_from_weights = network_loader.load_network_from_weights
    network = load_network_from_weights(network_name)
    network.load_weights(
        os.path.join(network_dir,network_name)
    )

    import pyrosetta
    init_flags = '-ignore_unrecognized_res 1 -include_current -ex1 -ex2 -ignore_waters 1'
    pyrosetta.init(init_flags)
    pose = pyrosetta.pose_from_pdb(pdbfile)

    np_protein = make_structural_info(pose)

    rmax = 10.
    nhs = get_padded_neighborhoods(
        np_protein[0],
        rmax,
        padded_length=1000,
        unique_chains=True)

    ks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
          12, 13, 14, 15, 16, 17, 18, 19, 20]
    num_combi_channels = len(ks)*7
    get_zernikegrams_partial = partial(
        get_zernikegrams,
        L_max = 5,
        ks = ks,
        num_combi_channels = num_combi_channels,
        r_max = 10.
    )
    zgrams = list(map(get_zernikegrams_partial,nhs))
    np_zgrams = np.zeros(
        shape=(nhs.shape),
        dtype=np.dtype([(str(l),'complex64',(num_combi_channels,2*l+1)) for l in range(5 + 1)])
    )
    for j,zgram in enumerate(zgrams):
        np_zgrams[j] = zgram[0]

    nh_ids = nhs['res_id']

    ds = tf_ds(np_zgrams, nh_ids, L_max)
    
    pseudoenergies = network.predict(ds.batch(batchsize))
    
    return pseudoenergies, nh_ids


def calculate_network_energies(
    df: pd.DataFrame,
    outfile: str    
) -> pd.DataFrame:

    pdb_chain_combos = df.groupby(['pdb','chain']).size().index
    pnEs = []
    for pdb,chain in pdb_chain_combos:
        curr_locs = np.logical_and(df['pdb'] == pdb,df['chain'] == chain)
        subdf = df[curr_locs]
        seq = ''.join(subdf['amino acid'])
        label_array = np.array([ol_to_ind_size[x] for x in seq])
        pnE = np.sum(
            subdf[[
                x for x in df.columns if 'pseudoenergy' in x
            ]].to_numpy()[np.arange(len(seq)),label_array])/len(seq)
        pnEs.append(pnE)
        
    pnE_df = pd.DataFrame(
        {'pdb_chain':
         ['_'.join(x) for x in df.groupby(['pdb','chain']).size().index]}
    )
    pnE_df['protein network energy'] = pnEs
    pnE_df.to_csv(outfile)


def main():
    parser = ArgumentParser()

    parser.add_argument(
        '--input', dest='input', type=str,
        help='Input data type. Can be of type pdb or zgrams')
    parser.add_argument(
        '--network_name', dest='network_name', type=str,
        default=saved_network,
        help='Network to use for predictions. By default set to the'\
        'network used in Pun et al, 2022')
    parser.add_argument(
        '--network_dir', dest='network_dir', type=str,
        default=saved_network_dir)
    parser.add_argument(
        '--zgram_file', dest='zgram_file', type=str)
    parser.add_argument(
        '--zgram_dataset', dest='zgram_dataset', type=str)
    parser.add_argument(
        '--pdb_file', dest='pdb_file', type=str)
    parser.add_argument(
        '--outfile', dest='outfile', type=str)
    parser.add_argument(
        '--pnE_outfile',dest='pnE_outfile',type=str)
    
    args = parser.parse_args()

    if args.input == 'zgram':
        logging.info('Getting predictions from premade zgram file')
        # TODO: check that zgram_dataset and zgram_file exist
        pseudoenergies, nh_list = predict_from_zernikegrams(
            args.zgram_file, args.zgram_dataset, 
            args.network_dir, args.network_name)

    if args.input == 'pdb':
        logging.info('Getting predictions from pdb file')
        pseudoenergies, nh_list = make_predictions_from_pdb(
            args.pdb_file, args.network_dir, args.network_name)
        
    logging.info('Writing predictions to csv')
    write_csv(args.outfile, pseudoenergies, nh_list)

    logging.info('Writing protein network energies')
    calculate_network_energies(
        pd.read_csv(args.outfile),
        args.pnE_outfile)
    
    logging.info('Program terminating succesfully')            
    
if __name__ == "__main__":
    logging.getLogger().setLevel('INFO')
    main()
