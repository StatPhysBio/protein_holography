#
# File to test the hnn.py class on fabricated data and premade data. 
#
# This file establishes the clebsch gordan coefficients and sets up an hnn with given parameters
# Then data is made up with proper dimensions and sizing. The hnn is tested on this data via a function call.
# Then this script loads holograms from .npy files, and then tests the network via a function call.
#

import tensorflow as tf
import numpy as np
import linearity
import nonlinearity
import hnn
import os


L_MAX = 3

# load clebsch gordan coefficients
cg_file = 'CG_matrix_l=5.npy'
cg_matrices = np.load(cg_file, allow_pickle=True).item()

tf_cg_matrices = {}
for l in range(L_MAX + 1):
    for l1 in range(L_MAX + 1):
        for l2 in range(0,l1+1):
            tf_cg_matrices[(l,l1,l2)] = tf.convert_to_tensor(cg_matrices[(l,l1,l2)],dtype=tf.complex64)
                    

# network parameters: 
num_layers = 3
num_aa = 20
hidden_l_dims = []
for i in range(num_layers):
    curr_l_dims = []
    for l in range(L_MAX + 1):
        curr_l_dims.append(np.random.randint(low=1,high=5))
    hidden_l_dims.append(curr_l_dims)
print('Making network with L_MAX=' + str(L_MAX) + ' with '  + str(num_layers) + ' layers with hidden dimensions ' + 
      str(hidden_l_dims))

# declare network
network = hnn.hnn(L_MAX, hidden_l_dims, num_layers, num_aa, tf_cg_matrices)

# fabricate data to test model
print('Making test input to model')
batch_size = 400
in_dims = [4,4,4,4]
network_input = {}
for l in range(L_MAX + 1):
    network_input[l] = tf.convert_to_tensor([[tf.complex(np.random.random(2*l+1),
                                                       np.random.random(2*l+1)) 
                                              for i in range(in_dims[l])]
                                             for j in range(batch_size)]
                                            ,dtype=tf.dtypes.complex64)



# test model
print('Testing hnn model')

print('network input:')
print(network_input)
    
prediction = network.predict(network_input)
print(prediction)

print('Tested model on fabricated data successfully')


# load premade holograms
print('Loading premade data to test model')
train_hgrams_real = np.load('train_hgram_real_example_examplesPerAA=20_k=0.0001_d=0.0_l=3.npy'
                            ,allow_pickle=True,encoding='latin1')[()]
train_hgrams_imag = np.load('train_hgram_imag_example_examplesPerAA=20_k=0.0001_d=0.0_l=3.npy'
                            ,allow_pickle=True,encoding='latin1')[()]
train_hgrams = {}
for l in range(L_MAX + 1):
    train_hgrams[l] = tf.convert_to_tensor(train_hgrams_real[l] + 1j * train_hgrams_imag[l],
                                           dtype=tf.dtypes.complex64)

 
new_prediction = network(train_hgrams)
print(new_prediction)


print('Terminating successfully')