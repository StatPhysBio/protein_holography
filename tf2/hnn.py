#
# Model to implement hnn
#

import tensorflow as tf
import linearity
import nonlinearity

class hnn(tf.keras.Model):
    
    def __init__(self, L_MAX, hidden_l_dims, num_layers, num_classes, cg_matrices, **kwargs):
        super().__init__(**kwargs)
        self.L_MAX = L_MAX
        self.hidden_l_dims = hidden_l_dims
        self.num_layers = num_layers
        self.cg_matrices = cg_matrices
        self.num_classes = num_classes

        temp_layers = []
        for i in range(num_layers):
            temp_layers.append(linearity.Linearity(hidden_l_dims[i], i, self.L_MAX))
            temp_layers.append(nonlinearity.Nonlinearity(self.L_MAX, self.cg_matrices))
        temp_layers.append(tf.keras.layers.Dense(num_classes))
        self.layers_ = temp_layers

    @tf.function
    def call(self, input):
        curr_nodes = input
        for layer in self.layers_[:-1]:
            curr_nodes = layer(curr_nodes)
        scalar_output = curr_nodes[0]
        scalar_out_real = tf.math.real(scalar_output)
        scalar_out_imag = tf.math.imag(scalar_output)
        print('Scalar output: ')
        print(scalar_output)
        scalar_output = tf.squeeze(tf.concat([scalar_out_real,scalar_out_imag], axis=0))
        output = self.layers_[-1](scalar_output)
        return output
            
        