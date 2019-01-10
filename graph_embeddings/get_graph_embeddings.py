# input: path to CSV files of connections
# output: list of tuples of form (node_name, node_embedding)
import argparse, pickle, os
from multigraph import Multigraph
from rgcn_embeddings import rgcn_embeddings

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def get_graph_embeddings(list_of_files):
    # create multigraph from list of connections
    graph = Multigraph()
    for f in list_of_files:
        graph.read_csv(f)
    embeddings = rgcn_embeddings(graph)
    return zip(graph.node_names, embeddings)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-d", "--dir", help="path of directory of csv files.")
    group.add_argument("-f", "--files",action='append', help="csv files with connections, separated by commas.")
    args = parser.parse_args()
    list_of_files = list()
    if args.files:
        list_of_files = [os.path.join(os.getcwd(),f) for f in args.files]
    elif args.dir:
        list_of_files = [os.path.join(os.getcwd(),args.dir,f) for f in os.listdir(args.dir) if f[-4:] == '.csv']
    print('Using the following CSV files:')
    for i in list_of_files:
        print('  '+i)
    #BEGIN
    embeddings = get_graph_embeddings(list_of_files)
    if not os.path.isdir(os.path.join(os.getcwd(),'results')):
        os.mkdir(os.path.join(os.getcwd(),'results'))
    save_object(embeddings, os.path.join(os.getcwd(),'results','embeddings.pkl'))
    print("Saved embeddings to results")
