# input: path to CSV files of connections (as well as features and targets)
# output: list of tuples of form (node_name, node_embedding)
from __future__ import print_function

import argparse, pickle, os
from gembed.multigraph import Multigraph, get_graph, csv_list_from_dir

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def get_graph_embeddings(algo, graph, embedding_dim, epochs=1, **kwargs):
    if algo == "rgcn":
        from gembed.embedding_models.rgcn_node_classification import rgcn_embeddings
        if 'target_csv' not in kwargs or kwargs['target_csv'] is None:
            raise ValueError("R-GCN requires a target CSV file.")
        embeddings = rgcn_embeddings(graph, embedding_dim, kwargs['target_csv'], epochs)
    elif algo == "ae":
        from gembed.embedding_models.simple_autoencode import autoencoder
        embeddings = autoencoder(graph, embedding_dim, epochs)
    elif algo == "distmult":
        from gembed.embedding_models.distmult import distmult_embeddings
        embeddings = distmult_embeddings(graph, embedding_dim, epochs)
    elif algo == "spectral":
        if 'target_csv' not in kwargs or kwargs['target_csv'] is None:
            raise ValueError("Spectral requires a target CSV file.")
        if 'features' not in kwargs or kwargs['features'] is None:
            raise ValueError("Spectral requires a features JSON file.")
        if graph.n_rels > 1:
            raise ValueError("Graph cannot have more than 1 realation to use spectral.")
        from gembed.embedding_models.spectral import spectral_embeddings
        embeddings = spectral_embeddings(graph, embedding_dim, kwargs['features'], kwargs['target_csv'],  epochs, kwargs['n_eigen'])
    elif algo == "gcn":
        if 'target_csv' not in kwargs or kwargs['target_csv'] is None:
            raise ValueError("Spectral requires a target CSV file.")
        if graph.n_rels > 1:
            raise ValueError("Graph cannot have more than 1 realation to use GCN.")
        from gembed.embedding_models.gcn import gcn_embeddings
        embeddings = gcn_embeddings(graph, embedding_dim, kwargs['target_csv'], kwargs['features'], epochs)
    return zip(graph.node_names, embeddings)

if __name__ == "__main__":
    algos = ['ae','gcn','rgcn','distmult','spectral']
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-p", "--path", help="path of directory of csv files.")
    group.add_argument("-i", "--input",action='append', help="csv files with connections")
    parser.add_argument("-n", "--name", help="name for output files.", required=True)
    parser.add_argument("-a", "--algo", help="which algorithm to use: {}".format(algos), required=True)
    parser.add_argument("-d", "--dim", type=int, help="embedding dimension.", required=True)
    parser.add_argument("-e", "--epochs", type=int, default=1, help="number of epochs.")
    parser.add_argument("-t", "--target",  help="csv file with targets for training.")
    parser.add_argument("-f", "--features", help="json file with features for training.")
    parser.add_argument("--eigen", type=int, help="number of eigenvectors to use in spectral analysis.")
    parser.add_argument("-u", "--undirected", action='store_true', help="Flag for undirected graphs.")
    args = parser.parse_args()
    # must use one of the available algorithms
    if args.algo not in algos:
        raise ValueError("algo argument must be one of these: {}".format(algos))
    list_of_files = list()
    if args.input:
        list_of_files = [os.path.join(os.getcwd(),f) for f in args.input]
    elif args.path:
        list_of_files = csv_list_from_dir(args.path)
    print('Using the following CSV files:')
    for i in list_of_files:
        print('  '+i)
    other_args = {'target_csv': args.target, 'features' : args.features, 'n_eigen': args.eigen}
    #Get the embedings!
    graph = get_graph(list_of_files, args.undirected)
    embeddings = get_graph_embeddings(args.algo, graph, args.dim, args.epochs, **other_args)
    #Save it to disk
    if not os.path.isdir(os.path.join(os.getcwd(),'results')):
        os.mkdir(os.path.join(os.getcwd(),'results'))
    save_object(graph, os.path.join(os.getcwd(),'results',args.name+'_graph.pkl'))
    print("Saved graph "+args.name+"_graph.pkl to results")
    save_object(embeddings, os.path.join(os.getcwd(),'results',args.name+'_embeddings.pkl'))
    print("Saved embeddings "+args.name+"_embeddings.pkl to results")
