from __future__ import print_function
from __future__ import absolute_import
import numpy as np
import os
import unicodecsv as csv
import operator
from future.utils import iteritems
from collections import defaultdict, Counter
from scipy.sparse import csr_matrix, hstack

def get_graph(csv_files, undirected=False):
    if undirected:
        print('Graph is undirected.')
    else:
        print('Graph is directed.')
    # create multigraph from list of connections
    if isinstance(csv_files, str):
        csv_files = [csv_files]
    graph = Multigraph(undirected)
    for f in csv_files:
        graph.read_csv(f)
    return graph

def csv_list_from_dir(dir_path):
    return [os.path.join(os.getcwd(),dir_path,f) for f in os.listdir(dir_path) if f[-4:] == '.csv']

class Multigraph:
    def __init__(self, undirected=False):
        # The numbering system should be internal to the object
        self.undirected = undirected
        self.n_nodes = 0
        self.n_rels = 0
        self.nodes = {}
        self.rels = {}
        self.node_names = []
        self.rel_names = []
        self.sparse_graph = {} #{relation: [row list, col list, data list]}
        self.edges = {} #{relation: [(src, rel, tar), (src, rel, tar), ... ]}
    
    def read_csv(self, csv_file, delimiter=",", exclude_first_row=False):
        i = 0
        with open(csv_file, 'r') as f:
            graphreader = csv.reader(f, delimiter=delimiter)
            for row in graphreader:
                if exclude_first_row and (i == 0):
                    i += 1
                    continue
                if len(row) != 3:
                    raise ValueError("Row {} of {} is not in triplet form 'subject, relation, predicate'".format(i,csv_file))
                self.add_connection(row)
                i += 1
        print('Processed {} lines.'.format(i))

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
            self.edges[self.rels[rel]] = set() #{relation: {(src, rel, tar), (src, rel, tar), ... } }
        # add new connection to graph
        self.sparse_graph[self.rels[rel]][0].append(self.nodes[src])
        self.sparse_graph[self.rels[rel]][1].append(self.nodes[targ])
        self.sparse_graph[self.rels[rel]][2].append(1)
        self.edges[self.rels[rel]].add( (self.nodes[src], self.rels[rel], self.nodes[targ]) )
    
    def delete_relation(self, relation):
        if (relation in self.rels) and (self.rels[relation] in self.sparse_graph):
            self.sparse_graph[self.rels[relation]] = [[],[],[]] #set it to empty
            self.edges[self.rels[rel]] = set() #set it to empty
            print("Deleted relation "+relation+" from graph.")
        else:
            print("Relation "+relation+" not in graph.")
    
    def get_connections_list_k(self, k):
        return list(self.edges[self.rels[k]])

    def get_connections_list(self):
        edges = []
        for i in self.rel_names:
            edges.extend(self.get_connections_list_k(i))
        return edges

    def get_adjacency_matrix(self):
        #interleave all the columns from the individual adjacency matrices
        #it is an out-going adjacency graph
        all_matrices = []
        for k in self.rel_names:
            all_matrices.append( self.get_adjacency_matrix_k(k) )
        full_matrix = hstack(all_matrices)
        return full_matrix.tocsr()
    
    def get_adjacency_matrix_k(self,k):
        #put in relation name k
        #it is an out-going adjacency graph of relation k
        graph_k = self.sparse_graph[self.rels[k]]
        shape = (self.n_nodes,self.n_nodes)
        sparse_adj_matrix =  csr_matrix((graph_k[2], (graph_k[0],graph_k[1])), shape=shape, dtype=np.int8)
        if self.undirected:
            transpose_adj_matrix = csr_matrix((graph_k[2], (graph_k[1],graph_k[0])), shape=shape, dtype=np.int8)
            return (sparse_adj_matrix + transpose_adj_matrix)
        else:
            return sparse_adj_matrix

    def get_transpose_adjacency_matrix_k(self,k):
        #put in relation name k
        #transpose of the out-going adjacency graph of relation k
        graph_k = self.sparse_graph[self.rels[k]]
        shape = (self.n_nodes,self.n_nodes)
        #switch the rows and columns
        sparse_adj_matrix =  csr_matrix((graph_k[2], (graph_k[1],graph_k[0])), shape=shape, dtype=np.int8)
        if self.undirected:
            transpose_adj_matrix = csr_matrix((graph_k[2], (graph_k[0],graph_k[1])), shape=shape, dtype=np.int8)
            return (sparse_adj_matrix + transpose_adj_matrix)
        else:
            return sparse_adj_matrix

    def get_list_of_relation(self, k):
        graph_k = self.sparse_graph[self.rels[k]]
        source_names = [self.node_names[i] for i in graph_k[0]]
        target_names = [self.node_names[i] for i in graph_k[1]]
        return zip(source_names, target_names)

    def get_node_name(self,index):
        return self.node_names[index]
    
    def get_relation_name(self,index):
        return self.rel_names[index]
    
    def count_relations(self):
        relation_counter = []
        for k in self.rel_names:
            relation_counter.append( (k,len(self.edges[self.rels[k]])) )
        #sort from greatest to least
        relation_counter.sort(key=operator.itemgetter(1), reverse=True)
        return relation_counter

    def count_node_edges(self):
        node_edge_counts = []
        mat = self.get_adjacency_matrix()
        for i in range(self.n_nodes):
            node_edge_counts.append((self.get_node_name(i), mat[i].indices.size))
        #sort from greatest to least
        node_edge_counts.sort(key=operator.itemgetter(1), reverse=True)
        return node_edge_counts

    def equivalent_nodes(self):
        repeats = defaultdict(list)
        mat = self.get_adjacency_matrix()
        for i in range(self.n_nodes):
            if( (not self.undirected) and mat[i].indices.size > 0) or (self.undirected and mat[i].indices.size > 1): 
                #don't count nodes that are end-points
                repeats[str(mat[i].indices)].append(self.get_node_name(i))
        return [v for (k,v) in iteritems(repeats) if len(v) > 1]

    def summary(self):
        summary_d = []
        connection_count = self.count_relations()
        node_count = self.count_node_edges()
        zero_index = -1
        final = 1 if self.undirected else 0
        for pos,t in enumerate(node_count):
            if t[1] == final:
                zero_index = pos
                break
        eq_nodes = self.equivalent_nodes()
        n_nodes_print = min(5, self.n_nodes)
        n_btm_nodes_print = min(n_nodes_print, zero_index)
        n_rels_print = min(5, self.n_rels)
        summary_d += [('Undirected graph?', self.undirected)]
        summary_d += [('n nodes', self.n_nodes)]
        summary_d += [('n relation types', self.n_rels)]
        summary_d += [('n connections',  sum(n for _, n in connection_count))]
        summary_d += [('5 most connected nodes', node_count[:n_nodes_print])]
        summary_d += [('5 least connected nodes (non-terminal)', node_count[zero_index-n_btm_nodes_print:zero_index])]
        summary_d += [('n terminal nodes', len(node_count[zero_index:]))]
        #summary_d += [('terminal nodes', [i[0] for i in node_count[zero_index:]])]
        summary_d += [('5 most frequent relations',connection_count[:n_rels_print])]
        summary_d += [('5 least frequent relations', connection_count[-n_rels_print:])]
        summary_d += [('equivalent nodes', eq_nodes)]
        for e in summary_d:
            print('{}: {}'.format(e[0],e[1]))
