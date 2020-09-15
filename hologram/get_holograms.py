
#
# Main file of program intended to run analysis on protein structural data using
# holographic machine learning techniques
#

# new import statements
import sys, os

import pdb_interface as pdb_int
import hologram
import numpy as np
from argparse import ArgumentParser
import imp
import protein_dir_parse as pdp
imp.reload(pdp)
imp.reload(pdb_int)


print('Finished importing modules')


#default_proteindir = os.path.join(os.path.dirname(__file__), "../data/proteins")
#default_outputdir = os.path.join(os.path.dirname(__file__), "../output")
#default_hgramdir = os.path.join(os.path.dirname(__file__), "../data/holograms")



# for testing purposes
default_proteindir = '/gscratch/stf/mpun/data/casp11/training30'
default_outputdir = '/gscratch/spe/mpun/protein_holography/data/holograms'
default_hgramdir = "/gscratch/spe/mpun/protein_holography/data/holograms"

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
                    help='protein directory')
parser.add_argument('--outputdir',
                    dest='outputdir',
                    type=str,
                    default=default_outputdir,
                    help='log/error directory')
parser.add_argument('--hgramdir',
                    dest='hgramdir',
                    type=str,
                    default=default_hgramdir,
                    help='hgram directory')
parser.add_argument('-e',
                    dest='e',
                    type=int,
                    default=2

,
                    help='examples per aminoacid')
parser.add_argument('--ch',
                    dest='ch',
                    type=str,
                    default='elnc',
                    help='channel type')

args =  parser.parse_args()


param_tag = "ch={}_e={}_l={}_k={}_d={}_rH={}".format(args.ch, args.e,
                                                      args.L, args.k,
                                                      args.d, args.rH)



#
# get proteins
#
print('Getting proteins from ' + args.proteindir)
trainProteins = pdp.get_proteins_from_dir(args.proteindir)
np.random.shuffle(trainProteins)
print(str(len(trainProteins)) + ' training proteins gathered')




#
# get amino acid structures from all training proteins
#

print('Getting ' + str(args.e) + ' training holograms per amino ' +
      'acid from training proteins')
if args.ch == 'aa':
    train_hgrams_real,train_hgrams_imag,train_labels = pdb_int.get_amino_acid_aa_shapes_from_protein_list(trainProteins,args.proteindir,args.e,args.d,args.rH,args.k,args.L)
if args.ch == 'aaCOA':
    train_hgrams_real,train_hgrams_imag,train_labels = pdb_int.get_amino_acid_aa_shapes_from_protein_list_COA(trainProteins,args.proteindir,args.e,args.d,args.rH,args.k,args.L)
if args.ch == 'el':
    train_hgrams_real,train_hgrams_imag,train_labels = pdb_int.get_amino_acid_el_shapes_from_protein_list(trainProteins,args.proteindir,args.e,args.d,args.rH,args.k,args.L,True)
if args.ch == 'elnc':
    train_hgrams_real,train_hgrams_imag,train_labels = pdb_int.get_amino_acid_el_shapes_from_protein_list(trainProteins,args.proteindir,args.e,args.d,args.rH,args.k,args.L,False)

print('Saving ' + param_tag + ' to ' + args.outputdir)

os.system('pwd')
os.system('ls')
hologram.save(train_hgrams_real, 'hgram_real_' + param_tag, args.hgramdir)
hologram.save(train_hgrams_imag, 'hgram_imag_' + param_tag, args.hgramdir)
hologram.save(train_labels,'labels_' + param_tag, args.hgramdir)


print('Terminating successfully')
    
