from __future__ import print_function

import pickle, os, random
import numpy as np
import scipy as sp
import unicodecsv as csv

from collections import defaultdict
from future.utils import iteritems

from gembed.multigraph import Multigraph
from gembed.layers import DistMult, GraphConvolution
from gembed.utils import evaluate_preds_sigmoid
from gembed.embedding_models.rgcn_node_classification import rgcn_inputs
from gembed.embedding_models.distmult import negative_samples, get_train_test_labels

import keras
import keras.backend as K
from keras.layers import Input, Dropout, Embedding
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

from sklearn.metrics import classification_report

def rgcn_distmult_embeddings(graph, embedding_dim, epochs=10):
    num_nodes = graph.n_nodes
    num_relations = graph.n_rels
    print("node number: {}".format(num_nodes))
    print("relation number: {}".format(num_relations))
    # training set is list of connections in graph
    triples = graph.get_connections_list()
    print("number of connections: {}".format(len(triples)))
    
    # need to also make negative examples, 10 for each positive one
    negative_triples = negative_samples(triples, num_nodes, num_relations, 20)
    
    # split the dataset in train and test sets
    # positive examples get target 1, negative examples get target 0
    x_train, y_train, x_test, y_test = get_train_test_labels(triples, negative_triples, train_frac=0.8)
    print("size of training samples is ",x_train.shape)
    print("size of testing samples is ",x_test.shape)
    #split up the relations from the nodes
    x_train_split = [x_train[:,0], x_train[:,1], x_train[:,2]]
    x_test_split = [x_test[:,0], x_test[:,1], x_test[:,2]]
    
    # get the other inputs

    A = rgcn_inputs(graph)
    X_train = np.zeros((x_train.shape[0],num_nodes))
    X_test = np.zeros((x_test.shape[0],num_nodes))
    
    # make model for binary classification of triplets
    
    embedding_model, training_model = rgcn_distmult_model(num_nodes, num_relations, X_train, A, embedding_dim)
    
    # train model
    
    training_model.fit([X_train]+x_train_split, y=y_train, epochs=epochs, shuffle=True, verbose=1)

    # print the test results
    score = training_model.evaluate([X_test]+x_test_split, y = y_test, verbose=1)
    print('Test loss: {}'.format(score[0]))
    print('Test accuracy: {}'.format(score[1]))
    result = training_model.predict([X_test]+x_test_split)
    preds = [1 if i > 0.5 else 0 for i in result]
    print(classification_report(y_test, preds))
    
    #return the embeddings
    
    node_indices = np.array(range(num_nodes))
    embeddings = embedding_model.predict([X, node_indices])
    embeddings = np.squeeze(embeddings)
    return embeddings


def rgcn_distmult_model(num_nodes, num_relations, features, A, encoding_dim, learn_rate=0.01, l2_regularization=0.0):
    #hyper parameters   
    L2 = l2_regularization
    LR = learn_rate
    support = len(A)
    print("support is {}".format(support))
    #inputs
    X_in = Input(shape=(features.shape[1],), sparse=False)
    subject_in = Input(shape=(1,)) 
    relation_in = Input(shape=(1,))
    object_in = Input(shape=(1,))
    # Define model architecture
    AdjacencyLayers = [ Embedding(num_nodes, num_nodes, weights=[A[i].toarray()], 
                                    input_length=1, trainable=False) for i in range(support) ]
    RelationAdjacencyLayer = Embedding(num_relations, encoding_dim, input_length=1)

    ConvLayer = GraphConvolution(encoding_dim, support=support, featureless=True,
                                    activation='relu', kernel_regularizer=l2(L2))
    DistMultLayer = DistMult(activation='sigmoid')
    #outputs
    adjacency_1 = [ K.squeeze(adj(subject_in), axis=1) for adj in AdjacencyLayers]
    adjacency_2 = [ K.squeeze(adj(object_in), axis=1) for adj in AdjacencyLayers]
    relation = RelationAdjacencyLayer(relation_in) 
    relation = K.squeeze(relation, axis=1)
    embedded_1 = ConvLayer([X_in] + adjacency_1)
    embedded_2 = ConvLayer([X_in] + adjacency_2)
    merged_embedding = keras.layers.concatenate([embedded_1, relation, embedded_2], axis=1)
    Y = DistMultLayer(merged_embedding)
    # Compile model
    training_model = Model(inputs=[X_in, subject_in, relation_in, object_in], outputs=Y)
    embedding_model = Model(inputs=[X_in, subject_in], outputs=embedded_1)
    training_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=LR), metrics=['accuracy'])
    training_model.summary()
    return (embedding_model, training_model)

