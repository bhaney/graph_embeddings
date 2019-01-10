from __future__ import print_function
from __future__ import absolute_import
import numpy as np
import unicodecsv as csv
from collections import defaultdict, Counter
from scipy.sparse import csr_matrix

class Multigraph:
    def __init__(self):
        self.n_nodes = 0
        self.n_rels = 0
        self.nodes = {}
        self.rels = {}
        self.node_names = []
        self.rel_names = []
        self.sparse_graph = {} #{relation: [row list, col list, data list]}
        self.connection_counter = Counter()
    
    def read_csv(self, csv_file, delimiter=",", exclude_first_row=False):
        i = 0
        with open(csv_file, 'r') as f:
            graphreader = csv.reader(f, delimiter=delimiter)
            for row in graphreader:
                if exclude_first_row and (i == 0):
                    i += 1
                    continue
                if len(row) != 3:
                    raise ValueError("Row {} is not in triplet form 'subject, relation, predicate'".format(i))
                self.add_connection(row)
                i += 1
        print('Added {} connections'.format(i))

    def add_connection(self, connection):
        # connection is (source, relation, target)
        src,rel,targ = connection
        #add new nodes and relations to dictionaries
        if src not in self.nodes.keys():
            self.nodes[src] = self.n_nodes
            self.node_names.append(src)
            self.n_nodes += 1
        if targ not in self.nodes.keys():
            self.nodes[targ] = self.n_nodes
            self.node_names.append(targ)
            self.n_nodes += 1
        if rel not in self.rels.keys():
            self.rels[rel] = self.n_rels
            self.rel_names.append(rel)
            self.n_rels += 1
            self.sparse_graph[self.rels[rel]] = [[],[],[]] #{relation: [row, col, data]}
        #count number of connections
        self.connection_counter.update({rel: 1})
        # add new connection to graph
        self.sparse_graph[self.rels[rel]][0].append(self.nodes[src])
        self.sparse_graph[self.rels[rel]][1].append(self.nodes[targ])
        self.sparse_graph[self.rels[rel]][2].append(1)
    
    def get_adjacency_matrix(self):
        #interleave all the columns from the individual adjacency matrices
        #it is an out-going adjacency graph
        full_matrix = [[],[],[]]
        for k in range(self.n_rels):
            full_matrix[0].extend(self.sparse_graph[k][0])
            #shift the column index to fit the added relation 
            col_shift = [i*self.n_rels+k for i in self.sparse_graph[k][1]]
            full_matrix[1].extend(col_shift)
            full_matrix[2].extend(self.sparse_graph[k][2])
        shape = (self.n_nodes,self.n_nodes*self.n_rels)
        return csr_matrix((full_matrix[2], (full_matrix[0],full_matrix[1])), shape=shape, dtype=np.int8)
    
    def get_adjacency_matrix_k(self,k):
        #put in relation name k
        #it is an out-going adjacency graph of relation k
        graph_k = self.sparse_graph[self.rels[k]]
        shape = (self.n_nodes,self.n_nodes)
        return csr_matrix((graph_k[2], (graph_k[0],graph_k[1])), shape=shape, dtype=np.int8)

    def get_transpose_adjacency_matrix_k(self,k):
        #put in relation name k
        #transpose of the out-going adjacency graph of relation k
        graph_k = self.sparse_graph[self.rels[k]]
        shape = (self.n_nodes,self.n_nodes)
        #switch the rows and columns
        return csr_matrix((graph_k[2], (graph_k[1],graph_k[0])), shape=shape, dtype=np.int8)

    def get_node_name(self,index):
        return self.node_names[index]
    
    def get_relation_name(self,index):
        return self.rel_names[index]
    
    def get_connection_counter(self):
        return self.connection_counter
