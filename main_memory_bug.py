#
# Main file of program intended to run analysis on protein structural data using
# holographic machine learning techniques
#

from config import Config
config = Config()

import resource
soft, hard = 10**9, 10**9
# soft, hard = 10**8, 10**8   # uncommenting this allows program to finish
resource.setrlimit(resource.RLIMIT_DATA,(soft, hard))


# 
# Import statements
#
print('Importing modules')
import os
print('os imported')
#import pdb_interface as pdb_int
#print('pdb_int imported')
import tensorflow.compat.v1 as tf
print('tf imported')
#import protein
#print('protein imported')
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
# example shapes per aa
examples_per_aa = 20

#
# load premade dataset
#
#cutoff_l = 4
#rh = 5.0
#k = 0.1
holograms_dir = os.path.join(config.get('datadir'), 'holograms/')
(train_hgrams_real,train_hgrams_imag,train_labels)  = hologram.load_holograms(
    k,d,cutoffL,examples_per_aa, file_workspace=holograms_dir)


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

