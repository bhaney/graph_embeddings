from __future__ import print_function

#Using Keras 2
from keras.engine import Layer
from keras.layers import Dense, Activation, Dropout
from keras import activations, initializers, regularizers
from keras import backend as K

class DenseTied(Dense):
    def __init__(self, master_layer, **kwargs):
        #output_dim needs to be equal to the input dimensions of the master_layer
        self.output_dim = master_layer.input_shape[-1]
        super(DenseTied, self).__init__(self.output_dim, **kwargs)
        self.master_layer = master_layer
    
    def build(self,input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.kernel = K.transpose(self.master_layer.kernel)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.output_dim,),
                            initializer=self.bias_initializer,
                            name='bias',
                            regularizer=self.bias_regularizer,
                            constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True
        
    def call(self, inputs):
        output = K.dot(inputs, K.transpose(self.master_layer.kernel))
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output

class DistMult(Activation):
    # Input shape
    # 3D tensor with shape: `(batch_size, sequence_length, input_dim)`.
    # Output shape
    # 2D tensor with shape: `(batch_size, 1)`.
    def __init__(self, activation, **kwargs):
        self.activation = activations.get(activation)
        super(DistMult, self).__init__(self.activation, **kwargs)

    def compute_output_shape(self, input_shape):
        assert(len(input_shape) >= 3)
        return (input_shape[0], 1) #batch_size

    def call(self, inputs):
        output = K.prod(inputs, axis=1, keepdims=True)
        output = K.sum(output, axis=2, keepdims=False)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def get_config(self):
        config = {'activation': self.activation.__name__}
        base_config = super(DistMult, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

