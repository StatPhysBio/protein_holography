#
# Main file of program intended to run analysis on protein structural data using
# holographic machine learning techniques
#

# new import statements
import sys
from protein_dir_parse import get_proteins_from_dir

# 
# old Import statements
#
print('Importing modules')
import os
print('os imported')
import pdb_interface as pdb_int
print('pdb_int imported')
import tensorflow as tf
print('tf imported')
import protein
print('protein imported')
#import hnn
print('hnn imported')
import hologram
print('holgram imported')
import sys
print('sys imported')
print('Finished importing modules')
import numpy as np
# constants
AA_NUM = 20

#
# parameters for the current analysis
#
print('Getting parameters')
# l value associated with maximum frequency used in fourier transforms
cutoffL = int(sys.argv[1])
# frequency to be used to make the holograms
k = float(sys.argv[2])
# hologram radius
rH = 5.
# noise distance
d = float(sys.argv[3])
# directories of proteins and workoing space
casp7Dir = '/gscratch/stf/mpun/data/casp11'
workDir = casp7Dir + '/workspace'
trainDir = casp7Dir + '/training30'
testDir = casp7Dir + '/val'




#
# get train and test proteins
#
print('Getting training proteins from ' + trainDir)
trainProteins = get_proteins_from_dir(trainDir)
np.random.shuffle(trainProteins)
print(str(len(trainProteins)) + ' training proteins gathered')
#print('Gathering testing proteins from ' + testDir)
#testProteins = pdb_int.get_proteins_from_dir(testDir)
#print(str(len(testProteins)) + ' testing proteins gathered')




#
# get amino acid structures from all training proteins
#
trainExamplesPerAa = 4
print('Getting ' + str(trainExamplesPerAa) + ' training holograms per amino ' +
      'acid from training proteins')
train_hgrams_real,train_hgrams_imag,train_labels = pdb_int.get_amino_acid_aa_shapes_from_protein_list(trainProteins,trainDir,trainExamplesPerAa,d,rH,k,cutoffL)

hologram_dir = '/gscratch/spe/mpun/holograms/'
hologram.save(train_hgrams_real,'aa_hgram_noCenter_real_example_examplesPerAA=' + str(trainExamplesPerAa) + '_k='+str(k)+'_d='+str(d)+'_l='+ str(cutoffL),hologram_dir)
hologram.save(train_hgrams_imag,'aa_hgram_noCenter_imag_example_examplesPerAA=' + str(trainExamplesPerAa) + '_k='+str(k)+'_d='+str(d)+'_l='+ str(cutoffL),hologram_dir)
hologram.save(train_labels,'aa_labels_noCenter_examplesPerAA=' + str(trainExamplesPerAa) + '_k='+str(k)+'_d='+str(d)+'_l='+ str(cutoffL),hologram_dir)


print('Terminating successfully')
    

