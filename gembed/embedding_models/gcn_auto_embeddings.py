# Based primarily on https://github.com/tkipf/relational-gcn/blob/master/rgcn/train.py
# From Thomas Kipf's relational-gcn repo
from __future__ import print_function

import pickle, os, random
import numpy as np
import scipy as sp
import unicodecsv as csv

from collections import defaultdict
from future.utils import iteritems

from gembed.multigraph import Multigraph
from rgcn.utils import sample_mask, get_splits, evaluate_preds

from keras.layers import Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

def get_train_test_labels(graph, train_frac=0.8):
    train_ids = list()
    test_ids = list()
    mat = graph.get_adjacency_matrix()
    for idx in range(graph.n_nodes):
        #randomly sort sample into the training or test set
        if random.random() < train_frac:
            train_ids.append(idx)
        else:
            test_ids.append(idx)
    return (mat, train_ids, test_ids) 

def rgcn_inputs(graph):
    A = list()
    for k in graph.rel_names:
	A.append(graph.get_adjacency_matrix_k(k))
	#A.append(graph.get_transpose_adjacency_matrix_k(k))
    A.append(sp.sparse.identity(graph.n_nodes).tocsr())
    # Normalize adjacency matrices
    for i in range(len(A)):
	d = np.array(A[i].sum(1)).flatten()
	d_inv = 1. / d
	d_inv[np.isinf(d_inv)] = 0.
	D_inv = sp.sparse.diags(d_inv)
	A[i] = D_inv.dot(A[i]).tocsr()
    return A

def rgcn_model(num_nodes, encoding_dim, output_dim, support, learn_rate=0.01, dropout=0.0, l2_regularization=0.0):
    from rgcn.layers.graph import GraphConvolution 
    from rgcn.layers.input_adj import InputAdj 
    #hyper parameters   
    L2 = l2_regularization
    LR = learn_rate
    DO = dropout
    #inputs
    A_in = [InputAdj(sparse=True) for _ in range(support)]
    X_in = Input(shape=(num_nodes,), sparse=True)
    # Define model architecture
    ConvLayer1 = GraphConvolution(encoding_dim, support=support, featureless=True, 
                                    activation='relu', kernel_regularizer=l2(L2))
    DropOutLayer = Dropout(DO)
    ConvLayer2 = GraphConvolution(output_dim, support=support, featureless=False, 
                                    activation='sigmoid')
    #outputs
    the_code = ConvLayer1([X_in] + A_in)
    H = DropOutLayer(the_code)
    Y = ConvLayer2([H] + A_in)
    # Compile model
    training_model = Model(inputs=[X_in] + A_in, outputs=Y)
    embedding_model = Model(inputs=[X_in] + A_in, outputs=the_code)
    training_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=LR))
    return (embedding_model, training_model)

def train_rgcn(model, data,  y_train, y_val, idx_train, idx_val, epochs=10, train_mask=None):
    # Train the R-GCN. Need to feed all nodes at the same time, so set batch_size=num_nodes.
    # Necessary that Inputs have sparse=True. batch_size would have size of many GB otherwise
    import time
    preds = None
    num_nodes = data[1].shape[1]
    for ep in range(1, epochs + 1):
        t = time.time()
        # Single training iteration
        model.fit(data, y=y_train, batch_size=num_nodes, sample_weight=train_mask, epochs=1, shuffle=False, verbose=0)
        if ep % 1 == 0:
            # Predict on full dataset
            preds = model.predict(data, batch_size=num_nodes)
            # Train / validation scores
            train_val_loss, train_val_acc = evaluate_preds(preds, [y_train, y_val], [idx_train, idx_val])
            print("Epoch: {:04d}".format(ep),
                  "train_loss= {:.4f}".format(train_val_loss[0]),
                  "train_acc= {:.4f}".format(train_val_acc[0]),
                  "val_loss= {:.4f}".format(train_val_loss[1]),
                  "val_acc= {:.4f}".format(train_val_acc[1]),
                  "time= {:.4f}".format(time.time() - t))

        else:
            print("Epoch: {:04d}".format(ep),
                  "time= {:.4f}".format(time.time() - t))
    return preds

def gcn_auto_embeddings(graph, embedding_dim, epochs=10):
    num_nodes = graph.n_nodes
    num_relations = graph.n_rels
    # get inputs for model
    A = rgcn_inputs(graph)
    X = sp.sparse.csr_matrix(A[0].shape)
    support = len(A)
    # get targets for model
    y, train_idx, test_idx = get_train_test_labels(graph, train_frac=0.8)
    y_train, y_val, y_test, idx_train, idx_val, idx_test = get_splits(y, train_idx, test_idx, True)
    train_mask = sample_mask(idx_train, y.shape[0])
    # make model for embedding
    output_dim = y_train.shape[1]
    embedding_model, training_model = rgcn_model(num_nodes, embedding_dim, output_dim, support)
    #train model
    preds = train_rgcn(training_model, [X] + A, y_train, y_val, idx_train, idx_val, epochs=epochs, train_mask=train_mask)
    # print the test results
    test_loss, test_acc = evaluate_preds(preds, [y_test], [idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(test_loss[0]),
          "accuracy= {:.4f}".format(test_acc[0]))
    #get embeddings
    embeddings = embedding_model.predict([X] + A, batch_size=num_nodes)
    return embeddings
