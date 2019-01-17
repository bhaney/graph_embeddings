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

import keras
from keras.layers import Input, Dropout, Reshape
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

from sklearn.metrics import classification_report

def rgcn_embeddings(graph, embedding_dim, target_csv, epochs=10):
    num_nodes = graph.n_nodes
    num_relations = graph.n_rels
    # get inputs for model
    A = rgcn_inputs(graph)
    X = sp.sparse.csr_matrix(A[0].shape)
    support = len(A)

    # training set is labels for certain nodes in the graph
    # x is the node number, and y is a one-hot vector of its category
    
    x_train, y_train, x_test, y_test = get_train_test_labels(graph, target_csv, train_frac=0.8)
    print("size of training samples is ",x_train.shape)
    print("size of testing samples is ",x_test.shape)
    
    # make model for embedding

    output_dim = y_train.shape[1]
    embedding_model, training_model = rgcn_model(num_nodes, embedding_dim, output_dim, support)

    #train model
    #training_model.fit([X]+A, y=y_train, batch_size=num_nodes, sample_weight=train_mask, epochs=epochs, shuffle=False, verbose=1)
    # print the test results
    #score = training_model.evaluate([X]+A, y = y_test, verbose=1)
    #print('Test loss: {}'.format(score[0]))
    #print('Test accuracy: {}'.format(score[1]))
    #result = training_model.predict([x_test_nodes,x_test_relations])
    #preds = [1 if i > 0.5 else 0 for i in result]
    #print(classification_report(y_test, preds))
    #test_loss, test_acc = evaluate_preds(preds, [y_test], [idx_test])
    #print("Test set results:",
    #      "loss= {:.4f}".format(test_loss[0]),
    #      "accuracy= {:.4f}".format(test_acc[0]))
    #get embeddings
    embeddings = embedding_model.predict([X] + A, batch_size=num_nodes)
    return embeddings

def get_train_test_labels(graph, filename, train_frac=0.8):
    training_dict = defaultdict(list)
    label_set = set()
    train_ids = list()
    test_ids = list()
    with open(filename, 'r') as csvfile:
        graphreader = csv.reader(csvfile)
        n = -1 #first row is the header, skip it
        for row in graphreader:
            if n < 0:
                n += 1
                continue
            #replace name of node with node index in the graph, 
            node_id = graph.nodes[row[0]]
            #append it to its category
            label_set.add(row[1])
            training_dict[idx] = row[1]
            #randomly sort sample into the training or test set
            if random.random() < train_frac:
                train_ids.append(idx)
            else:
                test_ids.append(idx)
            n += 1
    label_dict = {name:i for (i,name) in enumerate(list(label_set)) }
    #create the list of one-hot vectors for targets
    train_targets = np.zeros(len(train_ids),len(label_dict))
    for index in range(len(train_ids)):
        train_targets[index,label_dict[train_ids[index]]] = 1
    test_targets = np.zeros(len(test_ids),len(label_dict))
    for index in range(len(test_ids)):
        train_targets[index,label_dict[test_ids[index]]] = 1
    return (labels, train_ids, test_ids) 
    return (np.array(trains_ids), train_targets, np.array(test_ids), test_targets)

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

def rgcn_model(num_nodes, encoding_dim, output_dim, A, learn_rate=0.01, dropout=0.0, l2_regularization=0.0):
    #from rgcn.layers.graph import GraphConvolution 
    from gembed.layers import GraphConvolution 
    #hyper parameters   
    L2 = l2_regularization
    LR = learn_rate
    DO = dropout
    support = len(A)
    print("support is {}".format(support))
    #inputs
    X_in = Input(shape=(num_nodes,), sparse=False)
    node_in = Input(shape=(1,)) 
    # Define model architecture
    AdjacencyLayers = [ Embedding(num_nodes, num_nodes, weights=[A[i].todense()], 
                                    input_length=1, trainable=False) for i in range(support) ]
    ConvLayer1 = GraphConvolution(encoding_dim, support=support, featureless=True, 
                                    activation='relu', kernel_regularizer=l2(L2))
    #DropOutLayer = Dropout(DO)
    ConvLayer2 = GraphConvolution(output_dim, support=support, featureless=False, 
                                    activation='softmax')
    #outputs
    adjacencies = [ adj(node_in)  for adj in AdjacencyLayers]
    merged_adj = keras.layers.concatenate(adjacencies, axis=1)
    reshaped_adj = Reshape((support*num_nodes,))(merged_adj)
    the_code = ConvLayer1([X_in, reshaped_adj])
    #H = DropOutLayer(the_code)
    Y = ConvLayer2([the_code, reshaped_adj])
    # Compile model
    training_model = Model(inputs=[X_in] + A_in, outputs=Y)
    embedding_model = Model(inputs=[X_in] + A_in, outputs=the_code)
    training_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=LR), metrics=['accuracy'] )
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

