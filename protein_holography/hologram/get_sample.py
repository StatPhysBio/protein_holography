
#
# Main file of program intended to run analysis on protein structural data using
# holographic machine learning techniques
#

# new import statements
from argparse import ArgumentParser
import imp
import os
import sys

import numpy as np

import protein_holography.hologram.pdb_interface as pdb_int
import protein_holography.hologram.protein_dir_parse as pdp
import protein_holography.hologram.hologram as hologram

print('Finished importing modules')


#default_proteindir = os.path.join(os.path.dirname(__file__), "../data/proteins")
#default_outputdir = os.path.join(os.path.dirname(__file__), "../output")
#default_hgramdir = os.path.join(os.path.dirname(__file__), "../data/holograms")



# for testing purposes
default_proteindir = '/gscratch/stf/mpun/data/casp11/training30'
default_outputdir = '/gscratch/spe/mpun/protein_holography/data/samples'
default_hgramdir = "/gscratch/spe/mpun/protein_holography/data/holograms"

parser = ArgumentParser()
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
                    default=2,
                    help='examples per aminoacid')

args =  parser.parse_args()

#
# get proteins
#
print('Getting proteins from ' + args.proteindir)
trainProteins = pdp.get_proteins_from_dir(args.proteindir,suf='pdb')
np.random.shuffle(trainProteins)
print(str(len(trainProteins)) + ' training proteins gathered')




#
# get amino acid structures from all training proteins
#
os.chdir(args.proteindir)
print('Getting ' + str(args.e) + ' training samples per amino ' +
      'acid from training proteins')
sample = pdb_int.get_amino_acid_sample_from_protein_list(trainProteins,args.proteindir,args.e)

os.chdir(args.outputdir)
print('Saving sample to ' + args.outputdir)


f = open('aa_dataset_sim=30_res=2.5_e={}.dat'.format(args.e),'w')
for s in sample:
    f.write('{}\t{}\n'.format(s[0],s[1]))
f.close()

print('Terminating successfully')
    
