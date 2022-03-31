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

    def __init__(self, L_MAX, cg_matrices, connection, out_L_max=None, **kwargs):
        super().__init__(**kwargs)
        self.L_MAX = L_MAX
        self.cg_matrices = cg_matrices
        self.out_L_max = out_L_max
        self.relu = tf.keras.layers.LeakyReLU(
            alpha=0.3, **kwargs
        )
        valid_connections = ['self','self_square','simple','simply_connected','all','full','fuly_connected']
        if connection not in valid_connections:
            raise ValueError('Connection type not recognized\n',
                              'Valid connections are',valid_connections,'\n',
                              'You asked for',connection)

        self.connection = connection
        self.self_square_connections = ['self','self_square','simple','simply_connected']
        self.full_connections = ['all','full','fuly_connected']
        
    @tf.function
    def call(self, input,training=None):
        output = {}
        for L in range(self.L_MAX + 1):
            output[L] = []
        
        
        if self.connection in self.self_square_connections:
            # take products between all possible Ls and channels
            for l1 in range(self.L_MAX + 1):
                for l2 in range(l1,self.L_MAX + 1):
                    for L in range(l2-l1,np.minimum(self.L_MAX,l1+l2) + 1):
                        product = tf.einsum('Mnm,bim,bjn->bijM',
                                            self.cg_matrices[(L,l2,l1)],input[l1],input[l2]) # no conjugation
#                                        self.cg_matrices[(L,l2,l1)],input[l1],tf.math.conj(input[l2])) # conjugation
                        batch_size = -1
                        dim1 = product.shape[1]
                        dim2 = product.shape[2]
                        output[L].append(tf.reshape(product,[batch_size,dim1*dim2,2*L+1]))



        if self.connection in self.full_connections:
            # # take products between only self squares
            for l1 in range(self.L_MAX + 1):
                if l1 == 0:
                    L = 0
                    product = input[l1] * tf.cast(self.relu(tf.abs(input[l1])),tf.complex64)
                    batch_size = -1
                    output[L].append(product)
                    continue
                l2 = l1
                for L in range(l2-l1,np.minimum(self.L_MAX+1,l1+l2+1)):
                    product = tf.einsum('Mnm,bim,bin->biM',
                                        self.cg_matrices[(L,l2,l1)],input[l1],input[l2])
                    batch_size = -1
                    output[L].append(product)

        for L in range(self.L_MAX + 1):
            output[L] = tf.concat(output[L],axis=1)
            
        # if self.out_L_max != None:
        #     for L in range(self.out_L_max + 1):
        #         output[L] = []
        #     # # take products between only self squares
        #     for l1 in range(self.L_MAX + 1):
        #         l2 = l1
        #         for L in range(l2-l1,np.minimum(self.out_L_max+1,l1+l2+1)):
        #             product = tf.einsum('Mnm,bim,bin->biM',
        #                                 self.cg_matrices[(L,l2,l1)],input[l1],input[l2])
        #             batch_size = -1
        #             output[L].append(product)
                    
        #     for L in range(self.out_L_max + 1):
        #         output[L] = tf.concat(output[L],axis=1)
        return output


        
