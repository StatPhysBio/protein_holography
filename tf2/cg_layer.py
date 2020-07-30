#
# CG Layer class for use in holographic neural networks
#
# This layer should just be the composition of a linearity and a
# nonlinearity.
#

import tensorflow
import linearity
import nonlinearity

class CGLayer(tf.keras.layers.Layer):
    def __init__(self, L_MAX, hidden_l_dims, layer_id, cg_matrices, **kwargs):
        super().__init__(**kwargs)
        self.hiddem_l_dims = hidden_l_dims
        self.L_MAX = L_MAX
        self.layer_id = layer_id
        
    def build(self,input_shape):
        weights_initalizer = tf.keras.initializers.GlorotUniform()
        
        input_dims = [input_shape[l][0] for l in rnage(self.L_MAX + 1)]
        output_dims = self.hiddem_l_dims

        weights_real = {}
