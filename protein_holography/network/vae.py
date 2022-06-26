#
#  Python module to implement equivariant VAE.
#

import tensorflow.keras.backend as K
import tensorflow as tf
import linearity
import nonlinearity
import spherical_batch_norm as sbn

# reload imports for testing
from importlib import reload
reload(nonlinearity)


class vae(tf.keras.Model):

    def __init__(self,
                 num_decoder_layers,
                 num_encoder_layers,
                 encoder_hdims,
                 decoder_hdims,
                 encoder_Lmaxs,
                 decoder_Lmaxs,
                 cg_matrices,
                 reg_strength,
                 dropout_rate,
                 input_dims,
                 input_L_max,
                 **kwargs
    ):
        super().__init__(**kwargs)

        self.num_decoder_layers = num_decoder_layers
        self.num_encoder_layers = num_encoder_layers
        self.encoder_hdims = encoder_hdims
        self.decoder_hdims = decoder_hdims
        self.encoder_Lmaxs = encoder_Lmaxs
        self.decoder_Lmaxs = decoder_Lmaxs
        self.input_dims = input_dims
        self.input_L_max = input_L_max
        self.cg_matrices = cg_matrices
        
        # ENCODER
        temp_layers = []
        for i in range(num_encoder_layers):
            print('Encoder layer ',i)
            print('L_max this layer: ',encoder_Lmaxs[i])
            # add linear layer
            temp_layers.append(
                linearity.Linearity(
                    [encoder_hdims[i]] * (encoder_Lmaxs[i] + 1),
                    i,
                    encoder_Lmaxs[i],
                    reg_strength,
                    scale=1.0,
                    name='encoder_linear_' + str(i)
                )
            )
            # add spherical bath norm
            temp_layers.append(
                sbn.SphericalBatchNorm(
                    i,
                    encoder_Lmaxs[i],
                    scale=False,
                    name='encoder_sbn_' + str(i)
                )
            )
            # add nonlinear layer
            temp_layers.append(
                nonlinearity.Nonlinearity(
                    encoder_Lmaxs[i],
                    cg_matrices,
                    out_L_max = encoder_Lmaxs[i+1],
                    name='encoder_nonlinear_' + str(i)
                )
            )
 
        
            
        # DECODER
        for i in range(num_decoder_layers):
            print('Decoder layer: ',num_decoder_layers + i)
            print(decoder_hdims)
            # add linear layer
            temp_layers.append(
                linearity.Linearity(
                    [decoder_hdims[i]] * (decoder_Lmaxs[i + 1] + 1),
                    num_encoder_layers + i,
                    decoder_Lmaxs[i],
                    reg_strength,
                    scale=1.0,
                    name='decoder_linear_' + str(i)
                )
            )
            # add spherical bath norm
            temp_layers.append(
                sbn.SphericalBatchNorm(
                    num_encoder_layers + i,
                    decoder_Lmaxs[i],
                    scale=False,
                    name='decoder_sbn_' + str(i)
                )
            )
            # add nonlinear layer
            temp_layers.append(
                nonlinearity.Nonlinearity(
                    decoder_Lmaxs[i],
                    cg_matrices,
                    out_L_max = decoder_Lmaxs[i+1],
                    name='decoder_nonlinear_' + str(i)
                )

            )
        # add linear layer
        temp_layers.append(
            linearity.Linearity(
                input_dims,
                'final',
                input_L_max,
                reg_strength,
                scale=1.0,
                name='final',
            )
        )
        
        self.layers_ = temp_layers
        self.lyers = temp_layers
        
    # def build(self, input_shape):
    #     print('This is a test statement')
        
    #     self.input_shape_ = []
    #     for l in range(self.encoder_Lmaxs[0] + 1):
    #         self.input_shape_.append(tf.shape(input_shape[l])[1])
            
    #     print(input_shape)
    #     print(self.input_shape_)
    def call(self, input):
        self.input_ = input

        curr_nodes = input
        print('Input dimensions:')
        print('l values:',curr_nodes.keys())
        print('l0 dimensions:',tf.shape(curr_nodes[0]))
        print('\n')
        for layer in self.layers_:
            print(layer,layer.name)
            curr_nodes = layer(curr_nodes)
            print('l values:',curr_nodes.keys())
            print('l0 dimensions:',tf.shape(curr_nodes[0]))
            print('\n')
        return curr_nodes

                
    @tf.function
    def model(self):
        return tf.keras.Model(
            inputs=[self.input_],
            outputs=self.call(self.input_)
        )

    # @tf.function
    # def loss_fn(self,truth,pred):
    
    #     # get error
    #     print('he')
    #     print(truth[2])
    #     print(pred[2])
    #     L_max = self.encoder_Lmaxs[0]
    #     print('LMAX = ',L_max)
    #     error = {}
    #     for l in range(L_max + 1):
    #         error[l] = truth[l] - pred[l]
    #     print(truth,pred)
    #     print(error)
    #     print('still')
    #     # square error
    #     square_error = []
    #     for l in range(L_max + 1):
    #         print(l)
    #         square_error.append(
    #             tf.einsum(
    #                 'Mmn,im,in->iM',
    #                 self.cg_matrices[(0,l,l)],
    #                 error[l],
    #                 error[l]
    #             )
    #         )
    #     print('ello')
    #     square_error = tf.concat(square_error,axis=0)
    #     print('huh')
    #     # get absolute error
    #     abs_error = tf.abs(square_error)
        
    #     return tf.mean(abs_error)

        
