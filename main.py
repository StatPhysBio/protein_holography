#
# Main file of program intended to run analysis on protein structural data using
# holographic machine learning techniques
#

# 
# Import statements
#
import os
import tensorflow as tf
import sys
import hnn
import hologram

from tensorfieldnetworks.utils import FLOAT_TYPE

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
examples_per_aa_train = 200
examples_per_aa_test = 20


#
# load premade dataset

(train_hgrams_real,train_hgrams_imag,train_labels) = hologram.load_holograms(k,d,cutoffL,examples_per_aa_train)
(test_hgrams_real,test_hgrams_imag,test_labels) = hologram.load_holograms_test(k,d,cutoffL,examples_per_aa_test)




network,label,inputs_real,inputs_imag,loss,boltzmann_weights = hnn.hnn([4,10,10],AA_NUM,cutoffL)

optim = tf.train.AdamOptimizer(learning_rate=0.001)
train_op = optim.minimize(loss)

print('GRAPH:\n')
print(tf.Graph())

print('Initializing the network')
sess = tf.Session()
sess.run(tf.global_variables_initializer())


print('Training network')
# train the network
LAYER_0 = 0
REAL = 0
IMAG = 1
epochs = 1000
print_epoch = 100
hnn.train_on_data(train_hgrams_real,train_hgrams_imag,train_labels,
                  test_hgrams_real,test_hgrams_imag,test_labels,
                  inputs_real,inputs_imag,label,
                  sess,loss,train_op,boltzmann_weights,
                  epochs,print_epoch,cutoffL)

print('Terminating successfully')
    

