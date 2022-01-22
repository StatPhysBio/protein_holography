import os

import numpy as np
import tensorflow as tf

import protein_holography.network.hnn as hnn
import protein_holography.network.linearity as linearity
import protein_holography.network.nonlinearity as nonlinearity

# load clebsch gordan coefficients
cg_file = 'CG_matrix_l=5.npy'
cg_matrices = np.load(cg_file, allow_pickle=True).item()
L_MAX = 3
tf_cg_matrices = {}
for l in range(L_MAX + 1):
    for l1 in range(L_MAX + 1):
        for l2 in range(0,l1+1):
            tf_cg_matrices[(l,l1,l2)] = tf.convert_to_tensor(cg_matrices[(l,l1,l2)],dtype=tf.complex64)
                    
# load premade holograms
print('Loading test input to model')
train_hgrams_real = np.load('train_hgram_real_example_examplesPerAA=20_k=0.0001_d=0.0_l=3.npy'
                            ,allow_pickle=True,encoding='latin1')[()]
train_hgrams_imag = np.load('train_hgram_imag_example_examplesPerAA=20_k=0.0001_d=0.0_l=3.npy'
                            ,allow_pickle=True,encoding='latin1')[()]
train_hgrams = {}
for l in range(L_MAX + 1):
    train_hgrams[l] = tf.convert_to_tensor(train_hgrams_real[l] + 1j * train_hgrams_imag[l],
                                           dtype=tf.dtypes.complex64)

labels = tf.convert_to_tensor(np.load('train_labels_examplesPerAA=20_k=0.0001_d=0.0_l=3.npy',
                 allow_pickle=True,encoding='latin1'))

linear_l_dims = []
for l in range(L_MAX + 1):
    linear_l_dims.append(np.random.randint(3,5))
print(linear_l_dims)
# declare linear layer
linear_layer = linearity.Linearity(linear_l_dims, 0, L_MAX)
linear_output = linear_layer(train_hgrams)
print(linear_output)

nonlinear_layer = nonlinearity.Nonlinearity(L_MAX, tf_cg_matrices)
nonlinear_output = nonlinear_layer(linear_output)

print('Terminating successfully')
