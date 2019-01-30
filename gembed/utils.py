from __future__ import print_function
from future.utils import iteritems

import scipy.sparse as sp
import numpy as np
import unicodecsv as csv
import random, json
import gembed.multigraph
from collections import defaultdict
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score
from sklearn.metrics import log_loss

def laplacian(A):
    assert(A.get_shape()[0] == A.get_shape()[1])
    d = np.array(A.sum(1)).flatten()
    d_inv = 1. / d
    d_inv[np.isinf(d_inv)] = 0.
    D_inv = sp.diags(d_inv)
    I = sp.identity(A.get_shape()[0]).tocsr()
    L = (I - D_inv.dot(A)).tocsr()
    return L 

def normed_laplacian(A):
    assert(A.get_shape()[0] == A.get_shape()[1])
    At = A + sp.identity(A.get_shape()[0]).tocsr()
    dt = np.array(At.sum(1)).flatten()
    dt_inv = 1. / dt
    dt_inv[np.isinf(dt_inv)] = 0.
    Dt_inv = sp.diags(dt_inv)
    L = (Dt_inv.dot(At)).tocsr()
    return L 

def get_target_labels(graph, filename):
    target_dict = defaultdict(list)
    with open(filename, 'r') as csvfile:
        graphreader = csv.reader(csvfile)
        i = 0 #first row is the header, skip it
        for row in graphreader:
            if i == 0:
                i += 1
                continue
            #replace name of node with node index in the graph
            idx = graph.nodes[row[0]] 
            target_dict[row[1]].append(idx)
    num_labels = len(list(target_dict))
    #create a sparse matrix with all the nodes
    labels = sp.lil_matrix((graph.n_nodes, num_labels))
    #if some nodes do not have labels, they will be False in has_label
    has_label = np.full(graph.n_nodes, False)
    label_i = 0 #this converts a label k to a number label_i 
    for (k,v) in iteritems(target_dict):
        for s in v:
            #print('node(name) {}({}) in category {}'.format(s,graph.get_node_name(s),k))
            labels[s, label_i] = 1
            has_label[s] = True
        label_i += 1
    return (labels, has_label)

def get_matrix_of_features(features_json, n_nodes):
    with open(features_json,'r') as f:
        feature_dict = json.load(f) # feature_dict[feature] = [node1, node2, ...]
    #feature_matrix = np.zeros( (n_nodes, len(feature_dict)) )
    feature_rows, feature_cols, feature_vals = [], [] ,[]
    for (i,v) in enumerate(feature_dict.values()):
        for j in v:
            feature_rows.append(j)
            feature_cols.append(i)
            feature_vals.append(1)
            #feature_matrix[j,i] = 1
    #rows are the nodes, columns are the features
    feature_matrix = sp.csr_matrix((feature_vals, (feature_rows, feature_cols)), shape=(n_nodes, len(feature_dict)))
    return feature_matrix

def categorical_metrics(predictions, targets, train_mask, test_mask):
    preds = np.argmax(predictions,axis=1)
    targs = np.argmax(targets,axis=1)
    #print reports
    print('####')
    print('Train accuracy: {}'.format(accuracy_score(targs, preds, True, sample_weight=train_mask)))
    print('Train balanced accuracy: {}'.format(balanced_accuracy_score(targs, preds,sample_weight=train_mask)))
    print('Train loss: {}'.format(log_loss(targets, predictions,sample_weight=train_mask)))
    print('####')
    if sum(test_mask) != 0:
        print('Test accuracy: {}'.format(accuracy_score(targs, preds, True, sample_weight=test_mask)))
        print('Test balanced accuracy: {}'.format(balanced_accuracy_score(targs, preds,sample_weight=test_mask)))
        print('Test loss: {}'.format(log_loss(targets, predictions,sample_weight=test_mask)))
        print('####')
        print(classification_report(targs, preds, sample_weight=test_mask))
    else:
        print(classification_report(targs, preds, sample_weight=train_mask))
    print('####')

