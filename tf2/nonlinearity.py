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
    def __init__(self, L_MAX, cg_matrices, **kwargs):
        super().__init__(**kwargs)
        self.L_MAX = L_MAX
        self.cg_matrices = cg_matrices

    @tf.function
    def call(self, input):
        output = {}
        for L in range(self.L_MAX + 1):
            output[L] = []

        for l1 in range(self.L_MAX + 1):
            for l2 in range(l1,self.L_MAX + 1):
                for L in range(l2-l1,np.minimum(self.L_MAX,l1+l2) + 1):
                    print((l1,l2,L))
                    product = tf.einsum('Mnm,im,jn->ijM',self.cg_matrices[(L,l2,l1)],input[l1],input[l2])
                    dim1 = product.shape[0]
                    dim2 = product.shape[1]
                    print('dim1 = ' + str(dim1) + ' dim2 = ' +str(dim2))
                    output[L].append(tf.reshape(product,[dim1*dim2,2*L+1]))
        for L in range(self.L_MAX + 1):
            output[L] = tf.concat(output[L],axis=0)
        return output


        
