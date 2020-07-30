#
# Model to implement hnn
#

import tensorflow as tf
import linearity
import nonlinearity

class hnn(tf.keras.Model):
    
    def __init__(self, L_MAX, hidden_l_dims, num_layers, cg_matrices, **kwargs):
        super().__init__(**kwargs)
        self.L_MAX = L_MAX
        self.hidden_l_dims = hidden_l_dims
        self.num_layers = num_layers
        self.cg_matrices = cg_matrices
        
        temp_layers = []
        for i in range(num_layers):
            temp_layers.append(linearity.Linearity(hidden_l_dims[i], i, self.L_MAX))
            temp_layers.append(nonlinearity.Nonlinearity(self.L_MAX, self.cg_matrices))
        self.layers_ = temp_layers
            
    def call(self, input):
        curr_nodes = input
        for layer in self.layers_:
            curr_nodes = layer(curr_nodes)
        return curr_nodes
            
        
