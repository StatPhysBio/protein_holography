"""Module for parallel gathering of zernikegrams"""

from argparse import ArgumentParser
import logging
from progress.bar import Bar
import sys
import traceback

import h5py
from hdf5plugin import LZ4
import numpy as np

from protein_holography.coordinates.pyrosetta_hdf5_zernikegrams import (
    get_hologram)
from protein_holography.coordinates.preprocessor_hdf5_neighborhoods import (
    PDBPreprocessor)
from protein_holography.utils.posterity import get_metadata,record_metadata




def get_zernikegrams(np_nh,L_max,ks,num_combi_channels,r_max):

    #try:
    hgm = get_hologram(np_nh,L_max,ks,num_combi_channels,r_max)

    #except Exception as e:
    #    print(e)
    #    print('Error with',np_nh[0])
    #    #print(traceback.format_exc())
    #    return (None,)
    
    


    return hgm,np_nh['res_id']

def get_zernikegrams_from_dataset(
        hdf5_in,
        neighborhood_list,
        num_nhs,
        r_max,
        Lmax,
        ks,
        hdf5_out,
        parallelism,
        compression=LZ4()    
):
        
    # get metadata
    metadata = get_metadata()

    
    logging.basicConfig(level=logging.DEBUG)
    ds = PDBPreprocessor(hdf5_in,neighborhood_list)
    bad_neighborhoods = []
    n = 0
    channels = ['C','N','O','S','H','SASA','charge']
    num_combi_channels = len(channels) * len(ks)
    
    dt = np.dtype([(str(l),'complex64',(num_combi_channels,2*l+1)) for l in range(Lmax + 1)])

    logging.info(f"Transforming {num_nhs} in zernikegrams")
    logging.info("Writing hdf5 file")
    
    nhs = np.empty(shape=num_nhs,dtype=('S5',(6)))
    with h5py.File(hdf5_out,'w') as f:
        f.create_dataset(neighborhood_list,
                         shape=(num_nhs,),
                         dtype=dt,
                         compression=compression)
        f.create_dataset('nh_list',
                         dtype=('S5',(6)),
                         shape=(num_nhs,),
                         compression=compression
        )

    with Bar('Processing', max = ds.count(), suffix='%(percent).1f%%') as bar:
        with h5py.File(hdf5_out,'r+') as f:
            for i,hgm in enumerate(ds.execute(
                    get_zernikegrams,
                    limit = None,
                    params = {'L_max': Lmax,
                              'ks':ks,
                              'num_combi_channels': num_combi_channels,
                              'r_max': r_max},
                    parallelism = parallelism)):
                if hgm is None or hgm[0] is None:
                    bar.next()
                    print('error')
                    continue
                f['nh_list'][i] = hgm[1]
                f[neighborhood_list][i] = hgm[0]
                #print(hgm[0].shape)
                bar.next()

    #with h5py.File(hdf5_out,'r+') as f:
    #    f.create_dataset('nh_list',
    #                     data=nhs)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--hdf5_out', dest='hdf5_out', type=str,
        help='ouptut hdf5 filename', required=True)
    parser.add_argument(
        '--neighborhood_list', dest='neighborhood_list', type=str,
        help='neighborhood list within hdf5_in file', required=True)
    parser.add_argument(
        '--parallelism', dest='parallelism', type=int,
        help='ouptput file name', default=4)
    parser.add_argument(
        '--hdf5_in', dest='hdf5_in', type=str,
        help='hdf5 filename', default=False)
    parser.add_argument(
        '--num_nhs', dest='num_nhs', type=int,
        help='number of neighborhoods in protein set')
    parser.add_argument(
        '--Lmax', dest='Lmax', type=int,
        help='maximu spherical frequency to use in projections')
    parser.add_argument(
        '-k', dest='ks', type=int, nargs='+')
    parser.add_argument(
        '--r_max', dest='r_max', type=float,
        help='maximum radius to use in projections')
                        
    args = parser.parse_args()

    get_zernikegrams_from_dataset(
        args.hdf5_in,
        args.neighborhood_list,
        args.num_nhs,
        args.r_max,
        args.Lmax,
        args.ks,
        args.hdf5_out,
        args.parallelism
    )
    
if __name__ == "__main__":
    main()
