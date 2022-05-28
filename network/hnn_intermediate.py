#
# Python module to implement equivariant neural network
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

class hnn_intermediate(tf.keras.Model):
    
    def __init__(self, L_MAX, hidden_l_dims, num_layers,
                 num_classes, cg_matrices, num_dense_layers,
                 reg_strength, dropout_rate,  scale, connection,#layer,
                 **kwargs):

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
        # layer for intermediate output
        #self.layer = layer
        self.reg_strength = reg_strength
        self.dropout_rate = dropout_rate
        # create the layers
        temp_layers = []
        for i in range(num_layers):
            if i == 0:
#                temp_layers.append(sbn.SphericalBatchNorm(i, self.L_MAX, scale=False))
                temp_layers.append(linearity.Linearity(hidden_l_dims[i], i, self.L_MAX, self.reg_strength, scale = scale))                
                temp_layers.append(sbn.SphericalBatchNorm(i, self.L_MAX, scale=False))
            else:
                temp_layers.append(linearity.Linearity(hidden_l_dims[i], i, self.L_MAX, self.reg_strength))
                temp_layers.append(sbn.SphericalBatchNorm(i, self.L_MAX, scale=False))
            temp_layers.append(nonlinearity.Nonlinearity(self.L_MAX, self.cg_matrices, connection))
#            temp_layers.append(sbn.SphericalBatchNorm(i, self.L_MAX, scale=False))
        for i in range(num_dense_layers):
            temp_layers.append(
                tf.keras.layers.Dropout(dropout_rate)
            )
            temp_layers.append(
                tf.keras.layers.Dense(
                    num_classes,kernel_initializer=tf.keras.initializers.Orthogonal(),
                    bias_initializer=tf.keras.initializers.GlorotUniform(),
                    kernel_regularizer=tf.keras.regularizers.l1(reg_strength),                    
                )
            )

        # assignment of layers to a class feature
        self.layers_ = temp_layers

#    @tf.function
    def call(self, input):
        print('Called')
        # list to keep track of scalar output at each layer
        scalar_output = []
        
        # list to keep track of the outputs from all layers
        #intermediate_output = {}

        # variable to keep track of the latest nodes in the network computation
        curr_nodes = input
        self.input_ = input
        # begin recording scalar output with the input 
        scalar_output.append(curr_nodes[0])
        
        # record the inputs as the first intermediate outputs
        #intermediate_output['input'] = curr_nodes

        # compute the layers in the network while recording the scalar output 
        # after the nonlinearity steps
        for i,layer in enumerate(self.layers_[:-1]):
            print('her')
            tf.print(layer)
            curr_nodes = layer(curr_nodes)
            #intermediate_output[layer.name] = curr_nodes
#            if isinstance(layer,(sbn.SphericalBatchNorm)):
            if isinstance(layer,(nonlinearity.Nonlinearity)):
                scalar_output.append(curr_nodes[0])

        # transform scalar output from list of complex with dimensions 
        # num_layers x L_max x 1 to a list of floats with one dimension
        scalar_output = tf.concat(scalar_output,axis=1)
        scalar_out_real = tf.math.real(scalar_output)
        scalar_out_imag = tf.math.imag(scalar_output)
        scalar_output = tf.squeeze(
                            tf.concat([scalar_out_real,scalar_out_imag], axis=1), axis=-1
                        )

        intermediate_output['scalar input'] = scalar_output
        # feed scalar output into dense layer
        for i in range(self.num_dense_layers):
            scalar_output = self.layers_[-self.num_dense_layers+i](scalar_output)
        #intermediate_output['output energies'] = scalar_output

        return scalar_output #intermediate_output
            
    @tf.function
    def model(self):
        return tf.keras.Model(inputs=[self.input_],outputs=self.call(self.input_))
