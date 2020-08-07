#
# File to test the hnn.py class. 
#
# This file establishes the clebsch gordan coefficients, sets up an hnn with given parameters,
# loads holograms from .npy files, and then tests the network via a function call.
#


import tensorflow as tf
import numpy as np
import linearity
import nonlinearity
import hnn
import os


L_MAX = 3

# load clebsch gordan coefficients
cg_file = 'CG_matrix_l=10.npy'
cg_matrices = np.load(cg_file, allow_pickle=True).item()

tf_cg_matrices = {}
for l in range(L_MAX + 1):
    for l1 in range(L_MAX + 1):
        for l2 in range(0,l1+1):
            tf_cg_matrices[(l,l1,l2)] = tf.convert_to_tensor(cg_matrices[(l,l1,l2)],dtype=tf.complex64)
                    

# network parameters
num_layers = 2
num_aa = 20
hidden_l_dims = []
for i in range(num_layers):
    curr_l_dims = []
    for l in range(L_MAX + 1):
        curr_l_dims.append(10)
    hidden_l_dims.append(curr_l_dims)
print('Making network with L_MAX=' + str(L_MAX) + ' with '  + str(num_layers) + ' layers with hidden dimensions ' + 
      str(hidden_l_dims))

network = hnn.hnn(L_MAX, hidden_l_dims, num_layers, num_aa, tf_cg_matrices)

@tf.function
def loss_fn(truth, pred):
    return tf.nn.softmax_cross_entropy_with_logits(
        labels = truth,
        logits = pred)

optimizer = tf.keras.optimizers.Adam(learning_rate=1.e-3)

network.compile(optimizer=optimizer, loss=loss_fn)

# load premade holograms
k = 0.0001
d = 0.0
examples_per_aa = 20

print('Loading test input to model')
train_hgrams_real = np.load('train_hgram_real_example_examplesPerAA=' + str(examples_per_aa) + '_k=' + str(k) + '_d=' + str(d) + '_l=' + str(L_MAX) + '.npy'
                            ,allow_pickle=True,encoding='latin1')[()]
train_hgrams_imag = np.load('train_hgram_imag_example_examplesPerAA=' + str(examples_per_aa) + '_k=' + str(k) + '_d=' + str(d) + '_l=' + str(L_MAX) + '.npy'
                            ,allow_pickle=True,encoding='latin1')[()]
train_hgrams = {}
for l in range(L_MAX + 1):
    train_hgrams[l] = (train_hgrams_real[l] + 1j * train_hgrams_imag[l]).astype("complex64")


labels = np.load('train_labels_examplesPerAA=' + str(examples_per_aa) + '_k=' + str(k) + '_d=' + str(d) + '_l=' + str(L_MAX) + '.npy',
                 allow_pickle=True,encoding='latin1')


#print('Running network via predict')
#print(network.predict(train_hgrams,batch_size=50))

print('Training network')
network.fit(x=train_hgrams,y=labels,batch_size=400,epochs=1000,shuffle=True)


print('Terminating successfully')
