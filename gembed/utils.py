from __future__ import print_function
from future.utils import iteritems

import scipy.sparse as sp
import numpy as np
import unicodecsv as csv
import random
import gembed.multigraph
from collections import defaultdict

def laplacian(A):
    assert(A.get_shape()[0] == A.get_shape()[1])
    d = np.array(A.sum(1)).flatten()
    d_inv = 1. / d
    d_inv[np.isinf(d_inv)] = 0.
    D_inv = sp.diags(d_inv)
    I = sp.identity(A.get_shape()[0]).tocsr()
    L = (I - D_inv.dot(A)).tocsr()
    return L 

def get_train_test_labels(graph, filename, train_frac=0.8):
    training_dict = defaultdict(list)
    train_ids = list()
    test_ids = list()
    with open(filename, 'r') as csvfile:
        graphreader = csv.reader(csvfile)
        i = 0 #first row is the header, skip it
        for row in graphreader:
            if i == 0:
                i += 1
                continue
            #replace name of node with node index in the graph
            idx = graph.nodes[row[0]]
            training_dict[row[1]].append(idx)
            #randomly sort sample into the training or test set
            if random.random() < train_frac:
                train_ids.append(idx)
            else:
                test_ids.append(idx)
    num_labels = len(list(training_dict))
    #create a sparse matrix with all the nodes that have labels
    labels = sp.lil_matrix((graph.n_nodes, num_labels))
    i = 0
    for (k,v) in iteritems(training_dict):
        for s in v:
            labels[s, i] = 1
        i += 1
    return (labels, train_ids, test_ids)

