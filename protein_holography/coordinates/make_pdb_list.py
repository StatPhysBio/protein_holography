"""Create a pdb list hdf5 file for base of coordinate processing pipeline"""

from argparse import ArgumentParser
import logging
import os
import urllib.request

import pandas as pd
import h5py
import numpy as np


def csv_to_hdf5(
    csv_filename: str,
    hdf5_filename: str,
    ds_name: str,
    download: bool=False,
    pdb_dir: str='.'
):
    """Create a pdb list hdf5 file from csv of pdbs"""
    df = pd.read_csv(csv_filename)
    
    max_length = df['pdb'].str.len().max()
    pdbs = np.array(df['pdb'],f'|S{max_length}')
    
    with h5py.File(hdf5_filename,'w') as f:
        f.create_dataset(ds_name,data=pdbs)
    
    logging.info(
        f'Created {hdf5_filename} with dataset from {csv_filename}'
    )
    
    if download:
        retrieve_pdbs(df['pdb'].values,pdb_dir)
        
def retrieve_pdbs(
    pdb_list,
    pdb_dir
):
    for pdb in pdb_list:
        urllib.request.urlretrieve(
            f"http://www.rcsb.org/pdb/files/{pdb}.pdb.gz", 
            os.path.join(pdb_dir,f"{pdb}.pdb.gz"))
        os.system("gunzip -f " + os.path.join(pdb_dir,f"{pdb}.pdb.gz"))
    
    logging.info("Done fetching pdbs")

def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--csv_file',dest='csv_file',type=str,required=True)
    parser.add_argument(
        '--hdf5_file',dest='hdf5_file',type=str,required=True)
    parser.add_argument(
        '--dataset',dest='dataset',type=str,required=True)
    parser.add_argument(
        '--download',dest='download',action='store_true')
    parser.add_argument(
        '--pdb_dir',dest='pdb_dir',type=str,required=False,default='.')
    
    args = parser.parse_args()
    
    logging.debug(
        'Arguments to csv creation script are ',
        ' '.join([
            args.csv_file,
            args.hdf5_file,
            args.dataset,
            str(args.download),
            args.pdb_dir
        ])
    )

        
    csv_to_hdf5(
        args.csv_file,
        args.hdf5_file,
        args.dataset,
        download=args.download,
        pdb_dir=args.pdb_dir
    )


if __name__ == "__main__":
    logging.getLogger().setLevel('INFO')
    main()
    
    
