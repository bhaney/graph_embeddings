# Based on paper SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORKS
# arXiv:1609.02907v4
from __future__ import print_function

import pickle, os, random, json
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import unicodecsv as csv

from collections import defaultdict
from future.utils import iteritems

from gembed.multigraph import Multigraph
from gembed.utils import normed_laplacian, get_train_test_labels, get_matrix_of_features
from gembed.utils import categorical_metrics
from rgcn.utils import sample_mask, get_splits

from keras.layers import Input
from keras.engine import Layer
from keras.models import Model
from keras.optimizers import Adam
from keras import activations, initializers, regularizers, constraints
from keras import backend as K

class FirstChebConv(Layer):
    # Input Shape
    # 3D tensor with shape: (batch_size, n_nodes, n_features)
    # Output Shape
    # 3D tensor with shape: (batch_size, n_nodes, n_filters)
    def __init__(self, filters, n_nodes,
                 activation=None, data_format='channels_last',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None, kernel_constraint=None,
                 activity_regularizer=None, **kwargs):
        self.filters = filters
        self.activation = activations.get(activation)
        self.n_nodes = n_nodes
        self.data_format = data_format
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        super(FirstChebConv, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        feature_shape = input_shape[-1]
        output_shape = (feature_shape[0], feature_shape[1], self.filters)
        return output_shape  # (batch_size, n_nodes, n_filters)

    def build(self, input_shape):
        feature_shape = input_shape[-1]
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if feature_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = feature_shape[channel_axis]
        self.input_dim = input_dim
        kernel_shape = (self.input_dim, self.filters)
        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.built = True

    def call(self, inputs):
        # input 3D tensor with shape: (batch_size, n_nodes, n_features)
        # laplacian 3D tensor with shape: (batch_size, n_nodes, n_nodes)
        # kernel 2D tensor with shape: (n_features, n_filters)
        laplacian = inputs[0]
        features = inputs[1]
        output = K.map_fn(lambda x: K.dot(x, self.kernel), features)
        output = K.batch_dot(laplacian, output)
        if self.activation is not None:
            output = self.activation(output)
        return output

def gcn_model(n_nodes, n_features, encoding_dim, output_dim, learn_rate=0.01, l2_regularization=0.0):
    #hyper parameters   
    L2 = l2_regularization
    LR = learn_rate
    #inputs
    L_in = Input(shape=(n_nodes,n_nodes), sparse=False, name="laplacian")
    X_in = Input(shape=(n_nodes,n_features), sparse=False, name="features")
    # Define model architecture
    ConvLayer1 = FirstChebConv(encoding_dim, n_nodes, activation='tanh', kernel_regularizer=regularizers.l2(L2))
    ConvLayer2 = FirstChebConv(encoding_dim, n_nodes, activation='tanh', kernel_regularizer=regularizers.l2(L2))
    ConvLayer3 = FirstChebConv(output_dim, n_nodes, activation='softmax')
    #outputs
    R = ConvLayer1([L_in, X_in])
    the_code = ConvLayer2([L_in, R])
    Y = ConvLayer3([L_in, the_code])
    # Compile model
    training_model = Model(inputs=[L_in, X_in], outputs=Y)
    embedding_model = Model(inputs=[L_in, X_in], outputs=the_code)
    training_model.compile(loss='categorical_crossentropy', sample_weight_mode='temporal', optimizer=Adam(lr=LR), weighted_metrics=['accuracy'])
    training_model.summary()
    return (embedding_model, training_model)

def gcn_embeddings(graph, embedding_dim, target_csv, features_json=None, epochs=10):
    num_nodes = graph.n_nodes
    num_relations = graph.n_rels
    ###
    # get inputs for model
    ###
    L = normed_laplacian(graph.get_adjacency_matrix()).toarray()
    if features_json is None:
        X = np.identity(num_nodes)
    else:
        X = get_matrix_of_features(features_json, num_nodes).toarray()
    # have to reshape to be 3D, all nodes are processed at same time
    L = L.reshape((1, L.shape[0], L.shape[1]))
    X = X.reshape((1, X.shape[0], X.shape[1]))
    ###
    # get targets for model
    ###
    y, train_idx, test_idx = get_train_test_labels(graph, target_csv, train_frac=0.80)
    y_train, y_val, y_test, idx_train, idx_val, idx_test = get_splits(y, train_idx, test_idx, validation=False)
    train_mask = sample_mask(idx_train, num_nodes).astype(int) #sets all the nodes not used for training to False
    test_mask = sample_mask(idx_test, num_nodes).astype(int) #sets all the nodes not used for testing to False
    # have to reshape to be 3D, all nodes are processed at same time
    y =y.toarray()
    y = y.reshape((1, y.shape[0], y.shape[1]))
    y_val = y_val.reshape((1, y_val.shape[0], y_val.shape[1]))
    y_train = y_train.reshape((1, y_train.shape[0], y_train.shape[1]))
    y_test = y_test.reshape((1, y_test.shape[0], y_test.shape[1]))
    train_mask = train_mask.reshape((1,train_mask.shape[0]))
    test_mask = test_mask.reshape((1,test_mask.shape[0]))
    ###
    # make model for embedding
    ###
    output_dim = y_train.shape[-1]
    embedding_model, training_model = gcn_model(num_nodes, X.shape[-1], embedding_dim, output_dim)
    ###
    #train model
    ###
    training_model.fit([L,X], y=y, epochs=epochs, sample_weight=train_mask, shuffle=False, verbose=1)
    ###
    # print the test results
    ###
    result = training_model.predict([L,X])
    result = np.squeeze(result)
    targets = np.squeeze(y)
    categorical_metrics(result, targets, train_mask[0], test_mask[0])
    # reformat targets
    ###
    #get embeddings
    ###
    embeddings = np.squeeze(embedding_model.predict([L,X]))
    return embeddings
