#
# Main file of program intended to run analysis on protein structural data using
# holographic machine learning techniques
#

# 
# Import statements
#
import os
import pdb_interface as pdb_int
import tensorflow as tf
import protein
import hnn

from tensorfieldnetworks.utils import FLOAT_TYPE

# constants
AA_NUM = 20

#
# parameters for the current analysis
#

# l value associated with maximum frequency used in fourier transforms
cutoffL = 1
# frequency to be used to make the holograms
k = 0.0001
# hologram radius
rH = 5.
# noise distance
d = 2.0
# directories of proteins and workoing space
casp7Dir = '/home/mpun/scratch/protein_workspace/casp7'
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
trainExamplesPerAa = 1
print('Getting ' + str(trainExamplesPerAa) + ' training holograms per amino ' +
      'acid from training proteins')
train_hgrams_real,train_hgrams_imag,train_labels = pdb_int.get_amino_acid_shapes_from_protein_list(trainProteins,trainDir,
                                                          trainExamplesPerAa,
                                                          d,rH,k,cutoffL)

#
# get amino acid holograms from all test proteins
#

# PUT THIS IN LATER ONCE NETWORK WORKS
#tf.reset_default_graph()


network,label,inputs_real,inputs_imag,loss = hnn.hnn([4,10,10],AA_NUM,cutoffL)

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
epochs = 100
print_epoch = 10
hnn.train_on_data(train_hgrams_real,train_hgrams_imag,train_labels,
                  inputs_real,inputs_imag,label,
                  sess,loss,train_op,epochs,print_epoch)

print('Terminating successfully')
    

