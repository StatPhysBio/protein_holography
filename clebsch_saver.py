#
# Program to write Clebsch Gordan coefficients to text file 
#

print('Program starting')

import numpy as np
#from tensorfieldnetworks.utils import FLOAT_TYPE
import clebsch

cutoff_l = 13

print('Gathering cg coefficients')

# here we implement the Clebsch Gordan coefficients as
# 2l+1 x 2(l1)+1 x 2(l2)+1 matrices for use in taking direct products
# Fourier coefficients
cg_matrices = {}
tf_cg_matrices = {}
tf_add_cg_matrices = {}
add_cg_matrices = {}
for l in range(cutoff_l + 1):
    print(l)
    for l1 in range(cutoff_l + 1):
        for l2 in range(0,l1+1):
            cg_matrices[(l,l1,l2)] = np.zeros([2*l + 1, 2*l1 + 1, 2*l2 +1])
            for m in range(2*l+1):
                for m1 in range(2*l1 + 1):
                    for m2 in range(2*l2 + 1):
                        cg_matrices[(l,l1,l2)][m,m1,m2] = clebsch.clebsch(l1,m1-l1,l2,m2-l2,l,m-l)
                        #tf_cg_matrices[(l,l1,l2)] = tf.convert_to_tensor(cg_matrices[(l,l1,l2)],dtype=tf.complex64)
                        #tf_add_cg_matrices[(l,l1,l2)] = tf.convert_to_tensor(np.where(cg_matrices[(l,l1,l2)]>0,1,0))
                        #add_cg_matrices[(l,l1,l2)] = np.where(cg_matrices[(l,l1,l2)]!=0,1,0)


import os
os.chdir('/gscratch/spe/mpun/')
np.save('CG_matrix_l='+str(cutoff_l)+'.npy',cg_matrices)

print('Terminating')
