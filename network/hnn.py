#
# Pythonla module to implement equivariant neural network
# based on Clebsch-Gordan networks. This module uses linearity.py
# and nonlinearity.py to implement the linear and nonlinear operations
# of CG networks. Then it implements a dense layer at the end to 
# classify the scalar output of every layer in the CG network.
#

import tensorflow.keras.backend as K
import tensorflow as tf
import linearity
import nonlinearity
import spherical_batch_norm as sbn
import numpy as np

class hnn(tf.keras.Model):
    
    def __init__(self, L_MAX, hidden_l_dims, num_layers,
                 num_classes, cg_matrices, num_dense_layers,
                 reg_strength, dropout_rate, scale, connection, **kwargs):
        print('hnn n classes = {}'.format(num_classes))
        # call to the super function tf.keras.Model
        super().__init__(**kwargs)


        # max order L used in the network
        self.L_MAX = L_MAX
        # array of dimensions num_layers x l_max whose entries specify the dimension
        # to project each order l onto
        self.hidden_l_dims = hidden_l_dims
        # number of layers in the network
        # one linearity and one nonlinearity in each layer
        self.num_layers = num_layers
        # clesbch-gordan coefficients
        self.cg_matrices = cg_matrices
        # number of classes possible in classification task
        self.num_classes = num_classes
        # number of dense layers
        self.num_dense_layers = num_dense_layers

        # create the layers
        temp_layers = []
        for i in range(num_layers):
            curr_L_MAX = L_MAX
#            curr_L_MAX = np.min([L_MAX+1,10])
            if i == 0:
#                temp_layers.append(sbn.SphericalBatchNorm(i, self.L_MAX, scale=False))
                temp_layers.append(linearity.Linearity(hidden_l_dims[i], i, curr_L_MAX,
                                                       reg_strength,scale = scale))
                temp_layers.append(sbn.SphericalBatchNorm(i, curr_L_MAX, scale=False))
            else:
                temp_layers.append(linearity.Linearity(hidden_l_dims[i], i, curr_L_MAX,
                                                       reg_strength))
                temp_layers.append(sbn.SphericalBatchNorm(i, curr_L_MAX, scale=False))
            temp_layers.append(nonlinearity.Nonlinearity(curr_L_MAX, self.cg_matrices, connection))
#            temp_layers.append(sbn.SphericalBatchNorm(i, self.L_MAX, scale=False))
        if num_dense_layers > 1:
            dense_layer_hdims = np.linspace(500,5000,num_dense_layers-1,dtype=int)
            for i in range(num_dense_layers-1):
                temp_layers.append(
                    tf.keras.layers.Dropout(dropout_rate)
                )
                temp_layers.append(
                    tf.keras.layers.Dense(
                        dense_layer_hdims[-i],
                        kernel_initializer=tf.keras.initializers.Orthogonal(),
                        bias_initializer=tf.keras.initializers.GlorotUniform(),
                        kernel_regularizer=tf.keras.regularizers.l1(reg_strength),                 
                    )
                )
        temp_layers.append(
            tf.keras.layers.Dropout(dropout_rate)
        )
        temp_layers.append(
            tf.keras.layers.Dense(
                num_classes,
                kernel_initializer=tf.keras.initializers.Orthogonal(),
                bias_initializer=tf.keras.initializers.GlorotUniform(),
                kernel_regularizer=tf.keras.regularizers.l1(reg_strength),                 
            )
        )
        
        print(temp_layers)
        # assignment of layers to a class feature
        self.layers_ = temp_layers

    def _get_training_value(self, training=None):
        if training is None:
            training = K.learning_phase()
        return training


#    @tf.function
    def call(self, input):
#        tf.print('hnn call learning phase = {}'.format(K.learning_phase()))
        # list to keep track of scalar output at each layer
        scalar_output = []
        # variable to keep track of the latest nodes in the network computation
        curr_nodes = input
        self.input_ = input
        # begin recording scalar output with the input 
        # USE FOR INVARIANT NETWORK
        scalar_output.append(curr_nodes[0])
        
        # compute the layers in the network while recording the scalar output 
        # after the nonlinearity steps
        for layer in self.layers_[:-self.num_dense_layers*2]:
            curr_nodes = layer(curr_nodes)
#            if isinstance(layer,(sbn.SphericalBatchNorm)):
            # USE FOR INVARIANT NETWORK
            if isinstance(layer,(nonlinearity.Nonlinearity)):
                scalar_output.append(curr_nodes[0])
            # USE FOR NON-INVARIANT NETWORK
            # if layer == self.layers[-3]:
            #     a = [tf.reshape(curr_nodes[i],
            #                      [-1,
            #                       curr_nodes[i].shape[1]*curr_nodes[i].shape[2]])
            #                      for i in range(self.L_MAX + 1)]
            #    scalar_output = tf.concat(a,axis=-1)

        # transform scalar output from list of complex with dimensions 
        # num_layers x L_max x 1 to a list of floats with one dimension
        scalar_output = tf.concat(scalar_output,axis=1)
        scalar_out_real = tf.math.real(scalar_output)
        scalar_out_imag = tf.math.imag(scalar_output)
        scalar_output = tf.squeeze(scalar_out_real,axis=-1)
        #scalar_output = tf.squeeze(
        #                    tf.concat([scalar_out_real,scalar_out_imag], axis=1), axis=-1
        #                )
        #scalar_output = tf.concat([scalar_out_real,scalar_out_imag], axis=1)
        
        # feed scalar output into dense layer
        for i in range(self.num_dense_layers*2):
            scalar_output = self.layers_[-2*self.num_dense_layers+i](scalar_output)

        return scalar_output
            
    @tf.function
    def model(self):
        return tf.keras.Model(inputs=[self.input_],outputs=self.call(self.input_))
