from __future__ import print_function

#Using Keras 2
from keras.engine import Layer, InputSpec
from keras.layers import Dense, Activation, Dropout
from keras import activations, initializers, regularizers, constraints
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

class SpectralConv(Layer):
    # Input Shape
    # 3D tensor with shape: (batch_size, n_nodes, n_features)
    # Output Shape
    # 3D tensor with shape: (batch_size, n_nodes, n_filters)
    def __init__(self, filters, eigenvectors,
	         activation=None, n_eigen=None, # how many eigenvectors to use
		 data_format='channels_last',
		 kernel_initializer='glorot_uniform',
		 kernel_regularizer=None,
                 kernel_constraint=None, activity_regularizer=None):
        self.filters = filters
        self.activation = activation
        self.eigenvectors = eigenvectors
        self.eigenvectors_transpose = eigenvectors.transpose()
        if (n_eigen is None) or (n_eigen <= 0):
            self.n_eigen = self.eigenvectors.shape[1] #use all eigenvectors
        else:
            assert(n_eigen <= self.eigenvectors.shape[1])
            self.n_eigen = n_eigen
        self.data_format = data_format
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        super(SpectralConv, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        output_shape = (input_shapes[0], input_shapes[1], self.filters)
        return output_shape  # (batch_size, n_nodes, n_filters)

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
	    channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        self.eigenvectors = np.tile(self.eigenvectors, (input_shape[0],1,1))
        self.eigenvectors_transpose = np.tile(self.eigenvectors_transpose, (input_shape[0],1,1))
	kernel_shape = (self.n_eigen, input_dim, self.filters)
        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
				      constraint=self.kernel_constraint)
        self.input_spec = InputSpec(ndim=3,
                                    axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):
        # input 3D tensor with shape: (batch_size, n_nodes, n_features)
        # eigenvectors_tranpose 3D tensor with shape: (n_batch, n_eigen, n_nodes)
        # kernel 3D tensor with shape: (n_eigen, n_features, n_filters)
        # convovled with 3D tensor of shape: (n_batch, n_eigen, n_features)
        output = K.batch_dot(self.eigenvectors_transpose, inputs)
        output = K.conv1d(output, self.kernel, stride=0, padding='SAME', data_format=self.data_format)
        output = K.batch_dot(self.eigenvectors, output)
        if self.activation is not None:
            output = self.activation(output)
        return output
