#
# Equivariant batch normalization module for taking
# batch norm of a given L
#

from keras import backend as K
from keras.engine import InputSpec
from keras.layers import BatchNormalization
import tensorflow as tf


class LBatchNorm(BatchNormalization):

    def __init__(self,axis=-2,momentum=0.99, epsilon=1e-3, center=False, scale=False,
                 beta_initializer='zeros',gamma_initializer='ones',
                 moving_mean_initializer='zeros', moving_variance_initializer='ones',
                 beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
                 gamma_constraint=None, **kwargs):
        super(LBatchNorm, self).__init__(axis=axis, momentum=momentum, epsilon=epsilon,
                                                 center=center, scale=scale, 
                                                 beta_initializer=beta_initializer,
                                                 gamma_initializer=gamma_initializer,
                                                 moving_mean_initializer=moving_mean_initializer,
                                                 moving_variance_initializer=moving_variance_initializer,
                                                 beta_regularizer=beta_regularizer,
                                                 gamma_regularizer=gamma_regularizer,
                                                 beta_constraint=beta_constraint,
                                                 gamma_constraint=gamma_constraint, **kwargs)

    def build(self,input_shape):
        # get the dimension of the index not to be averaged over
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')
        # shape of the weights 
        # 
        # (this should be the channel dimension
        #  and then it will be broadcast to N x c x m
        #  in the call method)
        #
        shape = (dim,)

        # this ensures the input tensor has the proper shape??
        self.input_spec = InputSpec(ndim=len(input_shape),
                                    axes={self.axis: dim})

        # declare the moving mean and beta and gamma as constants
        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None

        # declare the moving variance as a constant
        self.moving_mean = tf.zeros(
            shape=shape,
            name='moving_mean')
        
        # declare the moving variance as a weight
        self.moving_variance = self.add_weight(
            shape=shape,
            name='moving_variance',
            initializer=self.moving_variance_initializer,
            trainable=False)

        self.built = True
        
    def _get_training_value(self, training=None):
        if training is None:
            training = K.learning_phase()
        return training

        
    
    @tf.function
    def call(self, inputs, training=None):
#        print('inputs = ' + str(inputs))
        def broadcast_to_input_shape(tensor,input_shape):
#            print(type(tensor))
#            print(input_shape)
            extra_dim_tensor = tensor[tf.newaxis,:,tf.newaxis]
            bc_tensor = tf.broadcast_to(extra_dim_tensor,input_shape)
            return bc_tensor

        training = self._get_training_value(training)
        
        input_shape = tf.shape(inputs)
        ndim = len(input_shape)
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
#        print('Reduction axes: ' + str(reduction_axes))
        
        if training in [0,False]:

            # broadcast all shapes to fit the inputs
#            print('Initial shapes = {},{}'.format(self.moving_mean.shape,
#                                                  self.moving_variance.shape)
#                  )

            broadcast_mean = broadcast_to_input_shape(self.moving_mean,
                                                      input_shape)
            broadcast_variance = broadcast_to_input_shape(self.moving_variance,
                                                      input_shape)
            if self.scale:
                broadcast_gamma = broadcast_to_input_shape(self.moving_gamma,
                                                      input_shape)
            else:
                broadcast_gamma = None
            if self.center:
                broadcast_beta = broadcast_to_input_shape(self.moving_beta,
                                                      input_shape)
            else:
                broadcast_beta = None
                
            
            # broadcast all shapes to fit the inputs
#            print('Final shapes = {},{}'.format(broadcast_mean.shape,
#                                                broadcast_variance.shape)
#                  )


            # normalize the inputs
            normalized_inputs = K.batch_normalization(
                inputs,
                broadcast_mean,
                broadcast_variance,
                broadcast_beta,
                broadcast_gamma,
                epsilon=self.epsilon)
#            print('Not training')
            return normalized_inputs


        else: # training
            # compute the current norm
            norms = tf.einsum('nc,nc->ncm',
                               inputs,
                               tf.math.conj(inputs))
            curr_mean_norm_per_channel = tf.reduce_mean(norms,axis=(0,-1))
            zero_mean = tf.zeros(shape=input_shape)
            
            # update moving norm
            self.add_update(
                [K.moving_average_update(
                        self.moving_variance,
                        curr_mean_norm_per_channel,
                        self.momentum)
                 ])

            # broadcast all shapes to fit the inputs
#            print('Initial training shapes = {},{}'.format(self.moving_mean.shape,
#                                                  curr_mean_norm_per_channel.shape)
#                  )
#            print(self.moving_mean)
            broadcast_mean = broadcast_to_input_shape(self.moving_mean,
                                                      input_shape)
#            broadcast_variance = broadcast_to_input_shape(curr_mean_norm_per_channel,
            broadcast_variance = broadcast_to_input_shape(self.moving_variance,
                                                      input_shape)
            if self.scale:
                broadcast_gamma = broadcast_to_input_shape(self.moving_gamma,
                                                      input_shape)
            else:
                broadcast_gamma = None
            if self.center:
                broadcast_beta = broadcast_to_input_shape(self.moving_beta,
                                                      input_shape)
            else:
                broadcast_beta = None
                
            
            # broadcast all shapes to fit the inputs
#            print('Final shapes = {},{}'.format(broadcast_mean.shape,
#                                                broadcast_variance.shape)
#                  )


            # normalize the inputs
            normalized_inputs = K.batch_normalization(
                inputs,
                broadcast_mean,
                broadcast_variance,
                broadcast_beta,
                broadcast_gamma,
                epsilon=self.epsilon)

            return normalized_inputs
