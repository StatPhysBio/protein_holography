#
# Nonlinearity class for use in holographic neural networks
#
# L:ayer takes inputs g^i of shape l x c(l,i) x m(l) and produces all squares
# in the irreducible basis via the Clebsch_Gordan coefficients
#
# In Einstein notation, this layer is summarized by the equation
#
#     f^{i}_{l,c(l,i),m(l)} = C^{l,m(l)}_{l_1,m_1(l_1),l_2,m_2(l_2)}
#
import tensorflow as tf
import numpy as np

class Nonlinearity(tf.keras.layers.Layer):
    def __init__(self, L_MAX, cg_matrices, out_L_max=False, **kwargs):
        super().__init__(**kwargs)
        self.L_MAX = L_MAX
        self.cg_matrices = cg_matrices
        if out_L_max == False:
            self.out_L_max = L_MAX
        else:
            self.out_L_max = out_L_max
        print('out_l_max', out_L_max)
        print('Nonlinear layer details:')
        print('L_MAX ',self.L_MAX)
        print('out_L_max ',self.out_L_max)
        print('\n')
    @tf.function
    def call(self, input,training=None):
        
        output = {}
        for L in range(self.out_L_max + 1):
            output[L] = []
        print('output keys: ',output.keys())
        # take products between all possible Ls and channels
#         for l1 in range(self.L_MAX + 1):
#             for l2 in range(l1,self.L_MAX + 1):
#                 for L in range(l2-l1,np.minimum(self.L_MAX,l1+l2) + 1):
#                     product = tf.einsum('Mnm,bim,bjn->bijM',
#                                         self.cg_matrices[(L,l2,l1)],input[l1],input[l2]) # no conjugation
# #                                        self.cg_matrices[(L,l2,l1)],input[l1],tf.math.conj(input[l2])) # conjugation
#                     batch_size = -1
#                     dim1 = product.shape[1]
#                     dim2 = product.shape[2]
#                     output[L].append(tf.reshape(product,[batch_size,dim1*dim2,2*L+1]))




        # # take products between only self squares
        #print('-----input keys: ',input.keys())
        for l1 in range(self.L_MAX + 1):
            l2 = l1
            for L in range(l2-l1,np.minimum(self.out_L_max,l1+l2) + 1):
                product = tf.einsum('Mnm,bim,bin->biM',
                                    self.cg_matrices[(L,l2,l1)],input[l1],input[l2])
                batch_size = -1
                output[L].append(product)

        for L in range(self.out_L_max + 1):
            output[L] = tf.concat(output[L],axis=1)

        return output


        
