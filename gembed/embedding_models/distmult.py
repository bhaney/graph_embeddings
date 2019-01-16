from __future__ import print_function

import pickle, os, random
import numpy as np
import scipy as sp
import unicodecsv as csv

from collections import defaultdict
from future.utils import iteritems

from gembed.multigraph import Multigraph
from gembed.layers import DistMult
from gembed.utils import evaluate_preds_sigmoid

from keras.layers import Input, Dropout, Embedding
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2


def distmult_embeddings(graph, embedding_dim, epochs=10):
    num_nodes = graph.n_nodes
    num_relations = graph.n_rels
    # training set is list of connections in graph
    triples = graph.get_connections_list()
    # need to also make negative examples, 10 for each positive one
    negative_triples = negative_samples(triples, num_nodes, num_relations, 1)
    # split the dataset in train and test sets
    # positive examples get target 1, negative examples get target 0
    x_train, y_train, x_test, y_test = get_train_test_labels(triples, negative_triples, train_frac=0.8)
    # make model for binary classification of triplets
    embedding_model, training_model = distmult_model(num_nodes, num_relations, embedding_dim)
    # train model
    print("size is",x_train.shape)
    train_distmult(training_model, x_train, y_train, epochs=epochs)
    # print the test results
    preds = training_model.predict(x_test)
    test_loss, test_acc = evaluate_preds_sigmoid(preds, y_test)
    print("Test set results:",
          "loss= {:.4f}".format(test_loss),
          "accuracy= {:.4f}".format(test_acc))
    # return the embeddings just for the nodes (not the relations)
    node_indices = np.array(range(num_nodes))
    embeddings = embedding_model.predict(node_indices)
    embeddings = np.squeeze(embeddings)
    return embeddings

def negative_samples(positive_triples, n_nodes, n_rels, n_negatives=10):
    set_of_triples = set(positive_triples)
    neg_triplets = []
    number_to_generate = len(positive_triples)*n_negatives
    generate_new_subj_or_obj = np.random.binomial(1, 0.5, number_to_generate)
    all_nodes = range(n_nodes)
    all_relations = range(n_rels)
    for i in range(number_to_generate):
        new_triple = False
        while not new_triple:
            sample1 = random.choice(all_nodes)
            sample2 = random.choice(all_relations)
            sample3 = random.choice(all_nodes)
            triple = (sample1, n_nodes+sample2, sample3)
            if triple not in set_of_triples:
                new_triple = True
                neg_triplets.append(triple)
                set_of_triples.add(triple)
    print("Generated {} negative samples.".format(number_to_generate))
    return neg_triplets
        
def get_train_test_labels(pos_samples, neg_samples, train_frac=0.8):
    train_samples = list()
    train_labels = list()
    test_samples = list()
    test_labels = list()
    train_or_test = np.random.binomial(1, train_frac, len(pos_samples))
    for i, sample in enumerate(pos_samples):
            if train_or_test[i]:
                train_samples.append(list(sample))
                train_labels.append(1)
            else:
                test_samples.append(list(sample))
                test_labels.append(1)
    train_or_test = np.random.binomial(1, train_frac, len(neg_samples))
    for i, sample in enumerate(neg_samples):
            if train_or_test[i]:
                train_samples.append(list(sample))
                train_labels.append(0)
            else:
                test_samples.append(list(sample))
                test_labels.append(0)
    return (np.array(train_samples), train_labels, np.array(test_samples), test_labels) 

def distmult_model(num_nodes, num_relations, encoding_dim, learn_rate=0.01, l2_regularization=0.0):
    #hyper parameters   
    L2 = l2_regularization
    LR = learn_rate
    #inputs
    X_in = Input(shape=(None,)) #inputs are the triplets which represent the edges
    # Define model architecture
    EmbeddingLayer = Embedding(num_nodes+num_relations, encoding_dim)
    DistMultLayer = DistMult(activation='sigmoid')
    #outputs
    the_code = EmbeddingLayer(X_in)
    Y = DistMultLayer(the_code)
    # Compile model
    training_model = Model(inputs=X_in, outputs=Y)
    embedding_model = Model(inputs=X_in, outputs=the_code)
    training_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=LR))
    return (embedding_model, training_model)

def train_distmult(model, data,  y_train, epochs=10):
    # Train the link preduction of the DistMult model
    # Can use whatever batch training you want
    import time
    preds = None
    for ep in range(1, epochs + 1):
        t = time.time()
        # Single training iteration
        model.fit(data, y=y_train, epochs=1, shuffle=True, verbose=0)
        if ep % 1 == 0:
            # Predict on full dataset
            preds = model.predict(data)
            # Train / validation scores
            train_loss, train_acc = evaluate_preds_sigmoid(preds, y_train)
            print("Epoch: {:04d}".format(ep),
                  "train_loss= {:.4f}".format(train_loss),
                  "train_acc= {:.4f}".format(train_acc),
                  "time= {:.4f}".format(time.time() - t))

        else:
            print("Epoch: {:04d}".format(ep),
                  "time= {:.4f}".format(time.time() - t))
