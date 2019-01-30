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
from gembed.layers import FirstChebConv
from gembed.utils import normed_laplacian, get_target_labels, get_matrix_of_features, categorical_metrics
from rgcn.utils import sample_mask, get_splits

from keras.layers import Input
from keras.engine import Layer
from keras.models import Model
from keras.optimizers import Adam
from keras import activations, initializers, regularizers, constraints
from keras import backend as K


def gcn_model(n_nodes, n_features, encoding_dim, output_dim, learn_rate=0.01, l2_regularization=0.0):
    #hyper parameters   
    L2 = l2_regularization
    LR = learn_rate
    #inputs
    L_in = Input(shape=(n_nodes,n_nodes), sparse=False, name="laplacian")
    X_in = Input(shape=(n_nodes,n_features), sparse=False, name="features")
    # Define model architecture
    ConvLayer0 = FirstChebConv(2*encoding_dim, n_nodes, activation='tanh', kernel_regularizer=regularizers.l2(L2))
    ConvLayer1 = FirstChebConv(2*encoding_dim, n_nodes, activation='tanh', kernel_regularizer=regularizers.l2(L2))
    ConvLayer2 = FirstChebConv(encoding_dim, n_nodes, activation='tanh', kernel_regularizer=regularizers.l2(L2))
    ConvLayer3 = FirstChebConv(output_dim, n_nodes, activation='softmax')
    #outputs
    R = ConvLayer0([L_in, X_in])
    R = ConvLayer1([L_in, R])
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
    y, target_mask = get_target_labels(graph, target_csv)
    training_fraction = 0.8
    if sum(target_mask) != num_nodes:
        print("Not all nodes have a label in {}".format(target_csv))
    train_mask = np.random.binomial(1, training_fraction, num_nodes)
    test_mask = np.invert(train_mask.astype(bool)).astype(int)
    #for karate club, only give one example from each class
    #train_mask = np.zeros(num_nodes)
    #train_mask[np.array([13,0,15,23])] = 1
    #test_mask = np.ones(num_nodes)
    # have to reshape to be 3D, all nodes are processed at same time
    y =y.toarray()
    y = y.reshape((1, y.shape[0], y.shape[1]))
    train_mask = train_mask.reshape((1,train_mask.shape[0]))
    test_mask = test_mask.reshape((1,test_mask.shape[0]))
    ###
    # make model for embedding
    ###
    output_dim = y.shape[-1]
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
