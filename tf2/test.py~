import tensorflow as tf
import numpy as np
import linearity
import nonlinearity
import os

L_MAX = 1
l_dims = [2,4]
in_dims = [2,4]
print('Declaring linearity layer')

linear_layer = linearity.Linearity(l_dims, 0, L_MAX)

linear_input = {}
for l in range(L_MAX + 1):
    linear_input[l] = tf.convert_to_tensor([tf.complex(np.random.random(2*l+1),np.random.random(2*l+1)) 
                                           for i in range(in_dims[l])],dtype=tf.dtypes.complex64)

print('Linear input:')
print(linear_input)
print('\n')

linear_output = linear_layer(linear_input)

print('Linear output:')
print(linear_output)
print('\n')



cg_file = 'CG_matrix_l=5.npy'
cg_matrices = np.load(cg_file, allow_pickle=True).item()

tf_cg_matrices = {}
for l in range(L_MAX + 1):
    for l1 in range(L_MAX + 1):
        for l2 in range(0,l1+1):
            tf_cg_matrices[(l,l1,l2)] = tf.convert_to_tensor(cg_matrices[(l,l1,l2)],dtype=tf.complex64)
                    

nonlinear_layer = nonlinearity.Nonlinearity(L_MAX, tf_cg_matrices)

nonlinear_input = linear_output
nonlinear_output = nonlinear_layer(nonlinear_input)

print(nonlinear_output)

print('Terminating successfully')
