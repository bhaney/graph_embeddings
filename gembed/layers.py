from __future__ import print_function

#Using Keras 2
from keras.engine import Layer
from keras.layers import Dense, Activation
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

class DistMult(Layer):
    # DistMultLayer([sequence_of_inputs, sequence_of_relations])
    # Both inputs have the same Input shape
    # sequence_of_inputs is 3D tensor with shape: `(batch_size, sequence_length, input_dim)`.
    # sequence_of_relations is 2D tensor with shape: `(batch_size, sequence_length)`.
    # Output shape
    # 2D tensor with shape: `(batch_size, 1)`.
    def __init__(self, activation, n_relations=1,
            kernel_initializer='uniform', kernel_regularizer=None,**kwargs):
        self.activation = activations.get(activation)
        self.n_relations = n_relations
        self.kernel_initializer=initializers.get(kernel_initializer)
        self.kernel_regularizer=regularizers.get(kernel_regularizer)
        super(DistMult, self).__init__(**kwargs)
    
    def build(self,input_shape):
        assert(type(input_shape) is list) # must have mutiple input shape tuples
        input_sequence_shape = input_shape[0]
        relation_sequence_shape = input_shape[1]
        assert(len(relation_sequence_shape) >= 2)
        input_dim = input_sequence_shape[-1] #encoding dimension
        self.embeddings = self.add_weight(shape=(self.n_relations, input_dim),
                                        initializer=self.kernel_initializer, 
                                        name='relation_embeddings',
                                        regularizer=self.kernel_regularizer)
        self.built = True

    def compute_output_shape(self, input_shape):
        assert(type(input_shape) is list) # must have mutiple input shape tuples
        input_sequence_shape = input_shape[0]
        relation_sequence_shape = input_shape[1]
        if len(input_sequence_shape) < 3:
            raise ValueError("'input_shape' is {}, must be at least a 3D tensor.".format(input_shape))
        batch_size = input_sequence_shape[0]
        return (batch_size, 1)
    
    def call(self, inputs):
        input_vectors = inputs[0]
        relation_vector_index = inputs[1]
        if K.dtype(relation_vector_index) != 'int32':
            relation_vector_index = K.cast(relation_vector_index, 'int32')
        relations = K.gather(self.embeddings, relation_vector_index)
        #relations = K.transpose(relations)
        output = K.concatenate([relations, input_vectors], axis = 1)
        output = K.prod(output, axis=1, keepdims=True)
        output = K.sum(output, axis=2, keepdims=False)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def get_config(self):
        config = {'output_dim': 1,
	         'n relations': self.n_relations, 
                 'kernel_initializer': self.kernel_initializer.__name__,
                 'activation': self.activation.__name__,
                 'kernel_regularizer': self.kernel_regularizer.get_config() if self.kernel_regularizer else None}
        base_config = super(DistMult, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
