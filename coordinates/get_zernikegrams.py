from pyrosetta_hdf5_zernikegrams import get_hologram
from preprocessor_hdf5_neighborhoods import PDBPreprocessor
from argparse import ArgumentParser
import numpy as np
import h5py
import sys
sys.path.append('/gscratch/spe/mpun/protein_holography/utils')
from posterity import get_metadata,record_metadata
import logging
from progress.bar import Bar
import traceback

def c(np_nh,L_max,ks,num_combi_channels,r_max):

    #try:
    hgm = get_hologram(np_nh,L_max,ks,num_combi_channels,r_max)

    #except Exception as e:
    #    print(e)
    #    print('Error with',np_nh[0])
    #    #print(traceback.format_exc())
    #    return (None,)
    
    


    
    return hgm,np_nh['res_id']


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--hdf5_out', dest='hdf5_out', type=str, help='ouptut hdf5 filename', required=True)
    parser.add_argument('--neighborhood_list', dest='neighborhood_list', type=str, help='neighborhood list within hdf5_in file', required=True)
    parser.add_argument('--parallelism', dest='parallelism', type=int, help='ouptput file name', default=4)
    parser.add_argument('--hdf5_in', dest='hdf5_in', type=str, help='hdf5 filename', default=False)
    parser.add_argument('--num_nhs', dest='num_nhs', type=int, help='number of neighborhoods in protein set')
    parser.add_argument('--Lmax', dest='Lmax', type=int, help='maximu spherical frequency to use in projections')
    parser.add_argument('-k', dest='ks', type=int, nargs='+')
    parser.add_argument('--r_max', dest='r_max', type=float, help='maximum radius to use in projections')

                        
    args = parser.parse_args()
    
    # get metadata
    metadata = get_metadata()

    
    logging.basicConfig(level=logging.DEBUG)
    ds = PDBPreprocessor(args.hdf5_in,args.neighborhood_list)
    bad_neighborhoods = []
    n = 0
    channels = ['C','N','O','S','H','SASA','charge']
    num_combi_channels = len(channels) * len(args.ks)
    
    dt = np.dtype([(str(l),'complex64',(num_combi_channels,2*l+1)) for l in range(args.Lmax + 1)])




    print(dt)
    print(args.num_nhs)
    print('writing hdf5 file')
    

    with h5py.File(args.hdf5_out,'w') as f:
        f.create_dataset(args.neighborhood_list,
                         shape=(args.num_nhs,),
                         dtype=dt)
    print('calling parallel process')
    nhs = np.empty(shape=args.num_nhs,dtype=('S5',(6)))
    with Bar('Processing', max = ds.count(), suffix='%(percent).1f%%') as bar:
        with h5py.File(args.hdf5_out,'r+') as f:
            for i,hgm in enumerate(ds.execute(
                    c,
                    limit = None,
                    params = {'L_max': args.Lmax,
                              'ks':args.ks,
                              'num_combi_channels': num_combi_channels,
                              'r_max': args.r_max},
                    parallelism = args.parallelism)):
                if hgm is None or hgm[0] is None:
                    bar.next()
                    print('error')
                    continue
                nhs[i] = hgm[1]
                f[args.neighborhood_list][i] = hgm[0]
                #print(hgm[0].shape)
                bar.next()

    print(len(nhs))
    with h5py.File(args.hdf5_out,'r+') as f:
        f.create_dataset('nh_list',
                         data=nhs)
    
    print('Done with parallel computing')
