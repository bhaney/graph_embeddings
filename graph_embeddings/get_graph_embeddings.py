# input: path to CSV files of connections
# output: list of tuples of form (node_name, node_embedding)
import argparse, pickle, os
from multigraph import Multigraph
from embedding_models.rgcn_embeddings import rgcn_embeddings
from embedding_models.autoencoder_embeddings import autoencoder

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def get_graph(list_of_files):
    # create multigraph from list of connections
    graph = Multigraph()
    for f in list_of_files:
        graph.read_csv(f)
    return graph

def get_graph_embeddings(graph, embedding_dim, target_csv=None, epochs=1):
    #embeddings = rgcn_embeddings(graph, embedding_dim, target_csv, epochs)
    embeddings = autoencoder(graph, embedding_dim, epochs)
    return zip(graph.node_names, embeddings)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-p", "--path", help="path of directory of csv files.")
    group.add_argument("-f", "--files",action='append', help="csv files with connections, separated by commas.")
    parser.add_argument("-t", "--target", help="csv file with targets for training.")
    parser.add_argument("-n", "--name", help="name for output files.", required=True)
    parser.add_argument("-e", "--epochs", type=int, help="number of epochs.", required=False)
    parser.add_argument("-d", "--dim", type=int, help="embedding dimension.", required=True)
    args = parser.parse_args()
    list_of_files = list()
    if args.files:
        list_of_files = [os.path.join(os.getcwd(),f) for f in args.files]
    elif args.path:
        list_of_files = [os.path.join(os.getcwd(),args.path,f) for f in os.listdir(args.path) if f[-4:] == '.csv']
    print('Using the following CSV files:')
    for i in list_of_files:
        print('  '+i)
    epochs = 1
    if args.epochs:
        epochs = args.epochs
    target_csv = None
    if args.target:
        target_csv = args.target
    name = args.name
    #Get the embedings!
    graph = get_graph(list_of_files)
    embeddings = get_graph_embeddings(graph, args.dim, target_csv=target_csv, epochs=epochs)
    #Save it to disk
    if not os.path.isdir(os.path.join(os.getcwd(),'results')):
        os.mkdir(os.path.join(os.getcwd(),'results'))
    save_object(graph, os.path.join(os.getcwd(),'results',name+'_graph.pkl'))
    print("Saved graph to results")
    save_object(embeddings, os.path.join(os.getcwd(),'results',name+'_embeddings.pkl'))
    print("Saved embeddings to results")
