# Based on "A Comprehensive Survey on Graph Neural Networks" by Zonghan Wu et al.
# arXiv:1901.00596v1
from __future__ import print_function

import pickle, os, random, json
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import unicodecsv as csv

from collections import defaultdict
from future.utils import iteritems

from gembed.multigraph import Multigraph
from gembed.layers import SpectralConv
from gembed.utils import laplacian, get_train_test_labels
from rgcn.utils import sample_mask, get_splits, evaluate_preds

from keras.layers import Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

def spectral_model(n_nodes, n_features, n_eigen, encoding_dim, output_dim, learn_rate=0.01, dropout=0.0, l2_regularization=0.0):
    #hyper parameters   
    L2 = l2_regularization
    LR = learn_rate
    DO = dropout
    #inputs
    E_in = Input(shape=(n_nodes,n_eigen), sparse=False)
    E_T_in = Input(shape=(n_eigen,n_nodes), sparse=False)
    X_in = Input(shape=(n_nodes,n_features), sparse=False)
    # Define model architecture
    ConvLayer1 = SpectralConv(encoding_dim, eigenvectors=n_eigen, activation='relu', kernel_regularizer=l2(L2))
    ConvLayer2 = SpectralConv(output_dim, eigenvectors=n_eigen, activation='softmax')
    #outputs
    the_code = ConvLayer1([E_in, E_T_in, X_in])
    Y = ConvLayer2([E_in, E_T_in, the_code])
    # Compile model
    training_model = Model(inputs=[E_in, E_T_in, X_in], outputs=Y)
    embedding_model = Model(inputs=[E_in, E_T_in, X_in], outputs=the_code)
    training_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=LR), metrics=['accuracy'])
    training_model.summary()
    return (embedding_model, training_model)

def get_matrix_of_features(features_json, n_nodes):
    with open(features_json,'r') as f:
        feature_dict = json.load(f) # feature_dict[feature] = [node1, node2, ...]
    feature_matrix = np.zeros( (1, n_nodes, len(feature_dict)) )
    for (i,v) in enumerate(feature_dict.values()):
        for j in v:
            feature_matrix[0,j,i] = 1
    return feature_matrix

def spectral_embeddings(graph, features_json, target_csv, embedding_dim, epochs=10):
    num_nodes = graph.n_nodes
    num_relations = graph.n_rels
    # get inputs for model
    L = laplacian(graph.get_adjacency_matrix())
    EV = eigsh(L, k=1000)[1]
    EV_T = EV.transpose()
    EV = EV.reshape((1,EV.shape[0],EV.shape[1]))
    EV_T = EV_T.reshape((1,EV_T.shape[0],EV_T.shape[1]))
    print(EV.shape)
    X = get_matrix_of_features(features_json, num_nodes)
    # get targets for model
    y, train_idx, test_idx = get_train_test_labels(graph, target_csv, train_frac=1.00)
    y_train, y_val, y_test, idx_train, idx_val, idx_test = get_splits(y, train_idx, test_idx, validation=False)
    y_train = y_train.reshape((1, y_train.shape[0], y_train.shape[1]))
    y_val = y_val.reshape((1, y_val.shape[0], y_val.shape[1]))
    y_test = y_test.reshape((1, y_test.shape[0], y_test.shape[1]))
    train_mask = sample_mask(idx_train, num_nodes)
    # make model for embedding
    output_dim = y_train.shape[-1]
    embedding_model, training_model = spectral_model(num_nodes, X.shape[-1], EV.shape[-1], embedding_dim, output_dim)
    #train model
    training_model.fit([EV,EV_T,X], y=y_train, epochs=epochs, shuffle=True, verbose=1)
    ## print the test results
    #score = training_model.evaluate([EV,EV_T,X], y=y_test, verbose=1)
    #print('Test loss: {}'.format(score[0]))
    #print('Test accuracy: {}'.format(score[1]))
    #result = np.squeeze(training_model.predict([EV,EV_T,X]))
    #print('result shape ',result.shape)
    #preds = [1 if i > 0.5 else 0 for i in result]
    #y_test = 
    #print(classification_report(y_test, preds))
    #get embeddings
    embeddings = np.squeeze(embedding_model.predict([EV,EV_T,X]))
    print(embeddings.shape)
    return embeddings
