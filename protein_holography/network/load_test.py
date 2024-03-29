#
# File to train networks built from the hnn.py class. 
#
# This file establishes the clebsch gordan coefficients, sets up an hnn with given parameters,
# loads holograms from .npy files, and then tests the network via a function call.
#

import os

import numpy as np
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

import protein_holography.network.hnn as hnn
import protein_holography.network.clebsch as clebsch

L_MAX = 6

# load clebsch gordan coefficients
cg_file = '/gscratch/spe/mpun/protein_holography/clebsch/CG_matrix_l=10.npy'
tf_cg_matrices = clebsch.load_clebsch(cg_file,L_MAX)

# network parameters
num_layers = 4
num_aa = 20
hidden_l_dims = []
for i in range(num_layers):
    curr_l_dims = []
    for l in range(L_MAX + 1):
        curr_l_dims.append(10)
    hidden_l_dims.append(curr_l_dims)
print('Making network with L_MAX=' + str(L_MAX) + ' with '  + str(num_layers) + ' layers with hidden dimensions ' + 
      str(hidden_l_dims))
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ^^^ put this into config files ^^^
#

network = hnn.hnn(L_MAX, hidden_l_dims, num_layers, num_aa, tf_cg_matrices)

@tf.function
def loss_fn(truth, pred):
    return tf.nn.softmax_cross_entropy_with_logits(
        labels = truth,
        logits = pred)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

network.compile(optimizer=optimizer, loss=loss_fn, metrics =['categorical_accuracy'])

# load premade holograms
k = 0.001
d = 10.0
examples_per_aa = 1000
examples_per_aa_val = 1000
d_val = 10.0


hologram_dir = "/gscratch/spe/mpun/holograms"
print('Loading test input to model')
train_hgrams_real = np.load(hologram_dir + '/train_hgram_real_example_examplesPerAA=' + str(examples_per_aa) + '_k=' + str(k) + '_d=' + str(d) + '_l=' + str(L_MAX) + '.npy'
                            ,allow_pickle=True,encoding='latin1')[()]
train_hgrams_imag = np.load(hologram_dir + '/train_hgram_imag_example_examplesPerAA=' + str(examples_per_aa) + '_k=' + str(k) + '_d=' + str(d) + '_l=' + str(L_MAX) + '.npy'
                            ,allow_pickle=True,encoding='latin1')[()]
train_hgrams = {}
for l in range(L_MAX + 1):
    train_hgrams[l] = (train_hgrams_real[l] + 1j * train_hgrams_imag[l]).astype("complex64")

print('Loading vaalidation input to model')
val_hgrams_real = np.load(hologram_dir + '/train_hgram_real_example_examplesPerAA=' + str(examples_per_aa_val) + '_k=' + str(k) + '_d=' + str(d_val) + '_l=' + str(L_MAX) + '.npy'
                            ,allow_pickle=True,encoding='latin1')[()]
val_hgrams_imag = np.load(hologram_dir + '/train_hgram_imag_example_examplesPerAA=' + str(examples_per_aa_val) + '_k=' + str(k) + '_d=' + str(d_val) + '_l=' + str(L_MAX) + '.npy'
                            ,allow_pickle=True,encoding='latin1')[()]
val_hgrams = {}
for l in range(L_MAX + 1):
    val_hgrams[l] = (val_hgrams_real[l] + 1j * val_hgrams_imag[l]).astype("complex64")


labels = np.load(hologram_dir + '/train_labels_examplesPerAA=' + str(examples_per_aa) + '_k=' + str(k) + '_d=' + str(d) + '_l=' + str(L_MAX) + '.npy',
                 allow_pickle=True,encoding='latin1')



val_labels = np.load(hologram_dir + '/train_labels_examplesPerAA=' + str(examples_per_aa_val) + '_k=' + str(k) + '_d=' + str(d_val) + '_l=' + str(L_MAX) + '.npy',                 allow_pickle=True,encoding='latin1')


print('Running network via predict')
#network.predict(train_hgrams,batch_size=1)

#print(network.summary())
network.load_weights('saved_weights/weights')
print('Testing network')
test_results = network.test_on_batch(x=train_hgrams,y=labels)
print(test_results)
print('Terminating successfully')
