#
# Main file of program intended to run analysis on protein structural data using
# holographic machine learning techniques
#

# new import statements
import sys, os
from protein_dir_parse import get_proteins_from_dir
import pdb_interface as pdb_int
import hologram
import numpy as np
from argparse import ArgumentParser
import imp


print('Finished importing modules')

# constants
AA_NUM = 20

#default_proteindir = os.path.join(os.path.dirname(__file__), "../data/proteins")
#default_outputdir = os.path.join(os.path.dirname(__file__), "../output/holograms")

#default_proteindir = '../../data/proteins'
#default_outputdir = '../../output/holograms'


# for testing purposes
default_proteindir = '/gscratch/stf/mpun/data/casp11/training30/'
default_outputdir = '/gscratch/spe/mpun/protein_holography/data/holograms/'


parser = ArgumentParser()
parser.add_argument('-L',
                    dest='L',
                    type=int,
                    default=1,
                    help='L value')
parser.add_argument('-k',
                    dest='k',
                    type=float,
                    default=1.,
                    help='k value')
parser.add_argument('-d',
                    dest='d',
                    type=float,
                    default=5.0,
                    help='d value')
parser.add_argument('--rH',
                    dest='rH',
                    type=float,
                    default=5.0,
                    help='hologram radius value')
parser.add_argument('--proteindir',
                    dest='proteindir',
                    type=str,
                    default=default_proteindir,
                    help='data directory')
parser.add_argument('--outputdir',
                    dest='outputdir',
                    type=str,
                    default=default_outputdir,
                    help='data directory')
parser.add_argument('-e',
                    dest='e',
                    type=int,
                    default=2,
                    help='examples per aminoacid')
parser.add_argument('--ch',
                    dest='ch',
                    type=str,
                    default='aa',
                    help='channel type')

args =  parser.parse_args()


param_tag = "ch={}_e={}_k={}_rH={}_d={}_l={}".format(args.ch, args.e, args.k,
                                                     args.rH, args.d, args.L)



#
# get proteins
#
print('Getting proteins from ' + args.proteindir)
trainProteins = get_proteins_from_dir(args.proteindir)
np.random.shuffle(trainProteins)
print(str(len(trainProteins)) + ' training proteins gathered')




#
# get amino acid structures from all training proteins
#

print('Getting ' + str(args.e) + ' training holograms per amino ' +
      'acid from training proteins')
train_hgrams_real,train_hgrams_imag,train_labels = pdb_int.get_amino_acid_aa_shapes_from_protein_list(trainProteins,args.proteindir,args.e,args.d,args.rH,args.k,args.L)

print('Saving ' + param_tag + ' to ' + args.outputdir)
hologram.save(train_hgrams_real, 'hgram_real_' + param_tag, args.outputdir)
hologram.save(train_hgrams_imag, 'hgram_imag_' + param_tag, args.outputdir)
hologram.save(train_labels,'labels_' + param_tag, args.outputdir)


print('Terminating successfully')
    

