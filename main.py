#
# Main file of program intended to run analysis on protein structural data using
# holographic machine learning techniques
#

# 
# Import statements
#
print('Importing modules')
import os
print('os imported')
import pdb_interface as pdb_int
print('pdb_int imported')
import tensorflow.compat.v1 as tf
print('tf imported')
import protein
print('protein imported')
import hnn
print('hnn imported')
import hologram
print('holgram imported')
import sys
print('sys imported')
from tensorfieldnetworks.utils import FLOAT_TYPE
print('Finished importing modules')
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
testDir = casp7Dir + '/validation'




#
# get train and test proteins
#
print('Getting training proteins from ' + trainDir)
trainProteins = pdb_int.get_proteins_from_dir(trainDir)
print(str(len(trainProteins)) + ' training proteins gathered')
print('Gathering testing proteins from ' + testDir)
testProteins = pdb_int.get_proteins_from_dir(testDir)
print(str(len(testProteins)) + ' testing proteins gathered')




#
# get amino acid structures from all training proteins
#
trainExamplesPerAa = 20
print('Getting ' + str(trainExamplesPerAa) + ' training holograms per amino ' +
      'acid from training proteins')
train_hgrams_real,train_hgrams_imag,train_labels = pdb_int.get_amino_acid_shapes_from_protein_list(trainProteins,trainDir,trainExamplesPerAa,d,rH,k,cutoffL)


#
# load premade dataset
#
#cutoff_l = 4
#rh = 5.0
#k = 0.1
#(train_hgrams_real,train_hgrams_imag,train_labels,
# test_hgrams_real,test_hgrams_imag,test_labels) = hologram.load_holograms(k,rh,cutoff_l)


# PUT THIS IN LATER ONCE NETWORK WORKS
#tf.reset_default_graph()


network,label,inputs_real,inputs_imag,loss,boltzmann_weights = hnn.hnn([4,10,10],AA_NUM,cutoffL)

optim = tf.train.AdamOptimizer(learning_rate=0.001)
train_op = optim.minimize(loss)

print('GRAPH:\n')
print(tf.Graph())

print('Initializing the network')
sess = tf.Session()
sess.run(tf.global_variables_initializer())

print('Examine the data')

print('Training network')
# train the network
LAYER_0 = 0
REAL = 0
IMAG = 1
epochs = 1000
print_epoch = 100
hnn.train_on_data(train_hgrams_real,train_hgrams_imag,train_labels,
                  inputs_real,inputs_imag,label,
                  sess,loss,train_op,boltzmann_weights,
                  epochs,print_epoch,cutoffL)

print('Terminating successfully')
    

