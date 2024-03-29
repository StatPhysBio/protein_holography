#
# Equivariant batch normalization layer
#
# Uses L_batch_norm.py to take batch norm across all Ls of inputs
#
from keras import backend as K
from keras.engine import InputSpec
from keras.layers import BatchNormalization
import tensorflow as tf

import protein_holography.network.L_batch_norm_gpu as lbn

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
        training = self._get_training_value(training)
        output = {}
        for l in range(self.L_MAX + 1):
            output[l] = 1.*self.layers[l](input[l],training)
        return output
