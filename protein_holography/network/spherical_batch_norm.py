#
# Equivariant batch normalization layer
#

#
#from keras.engine import InputSpec
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import BatchNormalization

import protein_holography.network.L_batch_norm as lbn

class SphericalBatchNorm(tf.keras.layers.Layer):
    def __init__(self, layer_id, L_MAX, scale=False, **kwargs):
        super().__init__(**kwargs)
        self.L_MAX = L_MAX
        self.layers = []
        for l in range(L_MAX + 1):
            self.layers.append(lbn.LBatchNorm(scale=scale))
        

    def _get_training_value(self, training=None):
        if training is None:
            training = K.learning_phase()
        return training

    @tf.function
    def call(self,input,training=None):
        output = {}
        for l in range(self.L_MAX + 1):
            output[l] = 1.*self.layers[l](input[l],training)
        return output
