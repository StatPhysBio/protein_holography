#
# This file computes the atomic spherical coordinates in a given set of
# neighborhoods and outputs a file with these coordinates.
#
# It takes as arguments:
#  - The name of the ouput file
#  - Name of central residue dataset
#  - Number of threads
#  - The neighborhood radius
#  - "easy" flag to include central res
#

from preprocessor import HDF5Preprocessor
import pandas as pd
import numpy as np
import logging
import itertools
import os
from Bio.PDB import Selection, NeighborSearch
from argparse import ArgumentParser
from progress.bar import Bar
import h5py
import sys
sys.path.append('/gscratch/spe/mpun/protein_holography/utils')
from posterity import get_metadata,record_metadata
import protein
import hologram as hgm
import naming

def c(coords,nb,ks,proj,l,rmax):
    EL_CHANNEL_NUM = 4
    # turn coordinates into coefficients


    # to be used once coefficients are gathered for all channels for current res
    curr_coeffs = []
    # list of the dicts for each channel for current res
    channel_dicts = []

    for k in ks:
        for curr_ch in range(EL_CHANNEL_NUM):
            curr_channel_coeffs = {}
            for l in range(l + 1):
                # if the current channel has no signal then append zero holographic signal
                if len(coords[curr_ch][0]) == 0:
                    curr_channel_coeffs[l] = np.array([0.]*(2*l+1))
                    continue
                if proj == 'hgram':
                    curr_channel_coeffs[l] = hgm.hologram_coeff_l(coords[curr_ch][0],
                                                                  coords[curr_ch][1],
                                                                  coords[curr_ch][2],
                                                                  rH, k, l)

                if proj == 'zgram':
                    curr_channel_coeffs[l] = hgm.zernike_coeff_l(coords[curr_ch][0],
                                                                 coords[curr_ch][1],
                                                                 coords[curr_ch][2],
                                                                 k, rmax, l)
            channel_dicts.append(curr_channel_coeffs)

    # coefficients gathered for every channel for this sample residue
    # channels_dicts currently has the structure n_c x l x m
    # we can now swap n_c and l
    for l in range(l + 1):
        curr_coeffs.append(np.stack([x[l] for x in channel_dicts]))
#    print(curr_coeffs)

    return (curr_coeffs,nb)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--output', dest='output', type=str, help='ouptput file name', required=True)
    parser.add_argument('--dataset', dest='dataset', type=str, help='dataset file name', required=True)
    parser.add_argument('--parallelism', dest='parallelism', type=int, help='ouptput file name', default=4)
    parser.add_argument('--invariants', dest='invariants', type=str, help='invariants file name', default='curated_invariants.txt')
    parser.add_argument('-d', dest='d', type=float, help='radius', default=5.0,nargs='+')
    parser.add_argument('--easy', action='store_true', help='easy', default=False)
    parser.add_argument('--hdf5', dest='hdf5', type=str, help='hdf5 filename', default=False)
    parser.add_argument('-k', dest='k', type=complex, nargs='+')
    parser.add_argument('--projection',dest='proj',help='radial projection',default=['zgram'],nargs='+')
    parser.add_argument('--ch',dest='ch',help='channel description',nargs='+',default=None)
    parser.add_argument('--rmax', dest='rmax', type=float, nargs='+', help='rescale radius for zernike', default=None)
    parser.add_argument('-L', dest='l', type=int, help='maximum spherical order', default=[5], nargs='+')
    parser.add_argument('-e', dest='e',help='examples',default=[128],nargs='+')

    args = parser.parse_args()
    
    # get metadata
    metadata = get_metadata()

    
    logging.basicConfig(level=logging.DEBUG)
    ds = HDF5Preprocessor(args.hdf5,args.dataset,'/gscratch/spe/mpun/protein_holography/data/coordinates/casp11_training30.hdf5')
 
    hgm_labels = []
    a = []
    n = 0
    with Bar('Processing', max = ds.count(), suffix='%(percent).1f%%') as bar:
        for proj,nb in ds.execute(
                c, limit = None, params = {'ks': args.k, 'proj': args.proj[0], 'l': args.l[0],
                                           'rmax': args.rmax[0]},
                parallelism = args.parallelism):
            name = nb[1].decode('utf-8')
            nh = (int(nb[2].decode('utf-8')),
                  nb[3].decode('utf-8'),
                  (nb[4].decode('utf-8'),
                   int(nb[5].decode('utf-8')),
                   nb[6].decode('utf-8')
               )
            )
            nh_name = "{}/{}/{}/".format(name,nh,10.)

            with h5py.File('/gscratch/spe/mpun/protein_holography/data/fourier/casp11_training30.hdf5','r+') as f:
                a.append(proj)
                zer = np.zeros(20)
                zer[protein.aa_to_ind[nb[0].decode('utf-8')]] = 1.
                hgm_labels.append(zer)
                for l in range(args.l[0] + 1):

                    #    print(proj[l])
                    
                    try:

                        dset = f.create_dataset(nh_name+'{}'.format(l),
                                                data = proj[l]
                                            )
                        record_metadata(metadata,dset)
                    except:
                        print("Unexpected error:", sys.exc_info()[0])
                    
                    
                    #o.write(str(n) + ',' + res + ',' + str(fid)+ ',' + str(i) +',{},{},{}'.format(invs[0][i][j],invs[1][i][j],invs[2][i][j]) + '\n')
            n+=1
            bar.next()


    data_id = naming.get_data_id(args)
    print(data_id)

    hgm_coeffs = {}
    hgm_coeffs_real = {}
    hgm_coeffs_imag = {}

    for l in range(args.l[0]+1):
        hgm_coeffs[l] = np.stack([x[l] for x in a]).astype('complex64')

    for k in hgm_coeffs.keys():
        print('key ',k)
        print(hgm_coeffs[k].shape)
    np.save('../data/zgram/' + data_id,hgm_coeffs)
    np.save('../data/zgram/labels_'+data_id,hgm_labels)
