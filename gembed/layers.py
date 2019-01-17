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
    # DistMultLayer([sequence_of_inputs, sequence_of_relations])
    # Both inputs have the same Input shape
    # sequence_of_inputs is 3D tensor with shape: `(batch_size, sequence_length, input_dim)`.
    # sequence_of_relations is 2D tensor with shape: `(batch_size, sequence_length)`.
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

class GraphConvolution(Layer):
    def __init__(self, output_dim, support=1, featureless=False,
                 kernel_initializer='glorot_uniform', activation='linear',
                 weights=None, kernel_regularizer=None,
                 bias_regularizer=None, use_bias=False, dropout=0.,input_dim=None,
                 **kwargs):
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.activation = activations.get(activation)
        self.output_dim = output_dim  # number of features per node
        self.support = support  # filter support / number of weights
        self.featureless = featureless  # use/ignore input features
        self.dropout = dropout

        assert support >= 1

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.use_bias = use_bias
        self.initial_weights = weights

        # these will be defined during build()
        self.input_dim = None
        self.kernel = None
        self.kernel_comp = None
        self.bias = None
        self.num_nodes = None

        super(GraphConvolution, self).__init__(**kwargs)

    def compute_output_shape(self, input_shapes):
        features_shape = input_shapes[0]
        support_shape = input_shapes[1]
        output_shape = (features_shape[0], self.output_dim)
        return output_shape  # (batch_size, output_dim)

    def build(self, input_shapes):
        features_shape = input_shapes[0]
        supports_shape = input_shapes[1]
        if self.featureless:
            self.num_nodes = features_shape[1]  # NOTE: Assumes featureless input (i.e. square identity mx)
        print('features shape',features_shape)
        assert len(features_shape) == 2
        print('supports shape', supports_shape)
        assert len(supports_shape) == 2
        self.input_dim = features_shape[1]

        self.kernel = self.add_weight(shape=(self.input_dim*self.support, self.output_dim),
                                                initializer=self.kernel_initializer,
                                                name='{}_kernel'.format(self.name),
                                                regularizer=self.kernel_regularizer)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.output_dim,),
                                     initializer='zero',
                                     name='{}_bias'.format(self.name),
                                     regularizer=self.bias_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, mask=None):
        features = inputs[0]
        A = inputs[1:]

        # convolve
        supports = list()
        for i in range(self.support):
            if not self.featureless:
                supports.append(K.dot(A[i], features))
            else:
                supports.append(A[i])
        supports = K.concatenate(supports, axis=-1)
        output = K.dot(supports, self.kernel)

        # if featureless add dropout to output, by elementwise multiplying with column vector of ones,
        # with dropout applied to the vector of ones.
        if self.featureless:
            tmp = K.ones(self.num_nodes)
            tmp_do = Dropout(self.dropout)(tmp)
            output = K.transpose(K.transpose(output) * tmp_do)
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'kernel_initializer': self.kernel_initializer.__name__,
                  'activation': self.activation.__name__,
                  'kernel_regularizer': self.kernel_regularizer.get_config() if self.kernel_regularizer else None,
                  'bias_regularizer': self.bias_regularizer.get_config() if self.bias_regularizer else None,
                  'use_bias': self.use_bias,
                  'input_dim': self.input_dim}
        base_config = super(GraphConvolution, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
