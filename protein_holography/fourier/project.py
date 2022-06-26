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
from argparse import ArgumentParser
import itertools
import logging
import os
import sys

from Bio.PDB import Selection, NeighborSearch
import h5py
import numpy as np
import pandas as pd
from progress.bar import Bar

import protein_holography.fourier.hologram as hgm
import protein_holography.fourier.naming as naming
from protein_holography.fourier.preprocessor import HDF5Preprocessor
from protein_holography.utils.posterity import get_metadata, record_metadata
import protein_holography.utils.protein as protein

def c(coords,weights,nb,ks,proj,l,rmax):
    EL_CHANNEL_NUM = len(coords)
    # turn coordinates into coefficients
    #print(EL_CHANNEL_NUM)
    # to be used once coefficients are gathered for all channels for current res
    curr_coeffs = []
    # list of the dicts for each channel for current res
    channel_dicts = []
    coords[6] = np.einsum('ij->ji',coords[6])
    try:
        # for coord in coords:
        #     print(coord.shape)
        # for weight in weights:
        #     if weight is None:
        #         continue
        #     else:
        #         print(weight.shape)

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
                                                                      rH, k, l
                        )

                    if proj == 'zgram':
                        curr_channel_coeffs[l] = hgm.zernike_coeff_l(coords[curr_ch][0],
                                                                     coords[curr_ch][1],
                                                                     coords[curr_ch][2],
                                                                     k, rmax, l,
                                                                     weights=weights[curr_ch]
                        )
                #print(curr_channel_coeffs[0].shape)
                channel_dicts.append(curr_channel_coeffs)

        # coefficients gathered for every channel for this sample residue
        # channels_dicts currently has the structure n_c x l x m
        # we can now swap n_c and l
        for l in range(l + 1):
            curr_coeffs.append(np.stack([x[l] for x in channel_dicts]))
            #    print(curr_coeffs)
    except Exception as e:
        print(e)
        return (None,nb)
    print('success')
    return (curr_coeffs,nb)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--output', dest='output', type=str, help='output file name', required=True)
    parser.add_argument('--input', dest='input', type=str, help='input file name', required=True)
    parser.add_argument('--dataset', dest='dataset', type=str, help='dataset file name', required=True)
    parser.add_argument('--parallelism', dest='parallelism', type=int, help='ouptput file name', default=4)
    parser.add_argument('-d', dest='d', type=float, help='radius', default=5.0,nargs='+')
    parser.add_argument('--ch', dest='ch', type=str, help='channel name',nargs='+')
    parser.add_argument('--easy', action='store_true', help='easy', default=False)
    parser.add_argument('--hdf5', dest='hdf5', type=str, help='hdf5 filename', default=False)
    parser.add_argument('-k', dest='k', type=complex, nargs='+')
    parser.add_argument('--projection',dest='proj',help='radial projection',default=['zgram'],nargs='+')
    parser.add_argument('--rmax', dest='rmax', type=float, nargs='+', help='rescale radius for zernike', default=None)
    parser.add_argument('-L', dest='l', type=int, help='maximum spherical order', default=[5], nargs='+')
    parser.add_argument('-e', dest='e',help='examples',default=[128],nargs='+')

    args = parser.parse_args()
    
    # get metadata
    metadata = get_metadata()

    
    logging.basicConfig(level=logging.DEBUG)
    ds = HDF5Preprocessor(args.hdf5,args.dataset,'/gscratch/spe/mpun/protein_holography/data/coordinates/{}'.format(args.input))
 
    hgm_labels = []
    a = []
    n = 0
    data_ids = []
    i = 0
    t = 0
    with Bar('Processing', max = ds.count(), suffix='%(percent).1f%%') as bar:
        for proj,nb in ds.execute(
                c, limit = None, params = {'ks': args.k, 'proj': args.proj[0], 'l': args.l[0],
                                           'rmax': args.rmax[0]},
                parallelism = args.parallelism):
            t += 1
            if proj is None:
                print(nb,' returned error')
                i += 1
                continue
            if t%100 == 0:
                print('\n\n Status ', i/t, '\n\n')
            name = nb[1].decode('utf-8')
            nh = (int(nb[2].decode('utf-8')),
                  nb[3].decode('utf-8'),
                  (nb[4].decode('utf-8'),
                   int(nb[5].decode('utf-8')),
                   nb[6].decode('utf-8')
               )
            )
            nh_name = "{}/{}/{}/".format(name,nh,10.)
            data_ids.append(nb)
            with h5py.File('/gscratch/spe/mpun/protein_holography/data/fourier/{}'.format(args.output),'r+') as f:
#            with h5py.File('/gscratch/stf/mpun/data/{}'.format(args.output),'r+') as f:
                a.append(proj)
                zer = np.zeros(20)
                zer[protein.aa_to_ind[nb[0].decode('utf-8')]] = 1.
                hgm_labels.append(zer)
                # for l in range(args.l[0] + 1):

                #     #    print(proj[l])
                    
                #     try:

                #         dset = f.create_dataset(nh_name+'{}'.format(l),
                #                                 data = proj[l]
                #                             )
                #         record_metadata(metadata,dset)
                #     except:
                #         print("Unexpected error:", sys.exc_info()[0])
                    
                    
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
    np.save('/gscratch/spe/mpun/protein_holography/data/zgram/' + data_id,hgm_coeffs)
    np.save('/gscratch/spe/mpun/protein_holography/data/zgram/labels_'+data_id,hgm_labels)
    np.save('/gscratch/spe/mpun/protein_holography/data/zgram/data_ids_'+data_id,np.array(data_ids))
