# Graph Embeddings

A package for creating embedding vectors for nodes in a relational graph. Input a CSV file (or a directory of CSVs) of connections in a graph (as well as optional features of the nodes of the graph), and receive a list of `(node_name, node_embedding)` tuples for each node in the graph. 


## Installation
This package was created with Python 2.7 and works with Keras 2.2.4 using Tensorflow 1.12.0 as the backend. It's recommended to use Anaconda to set up a clean environment in order to ensure no conflicts between various packages. You can download the anaconda package manager here: https://www.anaconda.com/download/

1. Create the anaconda enviorment
```
conda create --name gembed python=2.7
conda activate gembed
```
2. Download the repo
```
git clone https://github.com/bhaney/graph_embeddings
cd graph_embeddings
```
3. Install all of the dependencies
```
pip install git+https://github.com/bhaney/relational-gcn.git
python setup.py install
```
4. Run the program using the AIFB test data
```
cd gembed
python get_graph_embeddings.py -a rgcn -p data/aifb/relations -t data/aifb/person_affiliations.csv -e 50 -d 16 -n rgcn
python get_graph_embeddings.py -a ae -p data/aifb/relations -e 1 -d 16 -n autoencode
python get_graph_embeddings.py -a distmult -p data/aifb/relations -e 10 -d 16 -n distmult
```
The autoencoder needs about 50 epochs to produce a good embedding, but that can take more than an hour.

5. To plot the AIFB embeddings, there is an iPython notebook you can use.
```
conda install jupyter matplotlib
conda install -c conda-forge umap-learn
jupyter notebook
```
Then select `plot_aifb_graph_embeddings.ipynb`

## Usage

The program `get_graph_embeddings.py` takes the following arguments:
```
  -p PATH,      --path PATH,          Provide path to the directory of .CSV files.
  -i INPUT,     --input INPUT,        If you only want to run over one file, give path of the single CSV file..

  -n NAME,      --name NAME,          Name of the output file of the embeddings in the `results/` directory.
  -a ALGO,      --algo ALGO,          Which algorithm to use.
  -d DIM,       --dim DIM,            Desired embedding dimension.
  OPTIONAL:
  -e EPOCHS,    --epochs EPOCHS,      Number of epochs to train. Default is 1.
  -t TARGET,    --target TARGET,      CSV file with targets for training.
  -f FEATURES,  --features FEATURES,  JSON file with features for training.
    ,           --eigen EIGEN,        Number of eigenvectors to use in spectral analysis.
  -u,           --undirected,         Flag for undirected graphs. Default is directed.
```

The CSV file has to represent the graph by triplet connection of `subject_node, relation_edge, predicate_node`. Each row should contain one connection. 

Example:
```
dog,IS,mammal
dog,MAKES_SOUND,woof
cat,IS,mammal
cat,MAKES_SOUND,meow
mouse,IS,mammal
mouse,MAKES_SOUND,squeak
ball,IS,toy
toy,MAKES_SOUND,squeak
mammal,HAS,fur
rufus,INSTANCE_OF,dog
spot,INSTANCE_OF,dog
algernon,INSTANCE_OF,mouse
rufus,LOCATED_IN,chicago
spot,LOCATED_IN,chicago
chicago,IS,city
```

The output of `get_graph_embeddings` will be a list of `(node_name, node_embedding)` for each node in the graph. It you run it from the command line, the embeddings will be saved as a `.pkl` file in the `results/` directory.

## Multigraph Summary

To see a summary of facts about the graph, you can use the `summary()` function from the Multigraph class. As an example:

```
python
>>> from gembed.multigraph import get_graph
>>> g = get_graph('gembed/data/example_graph.csv')
Processed 15 lines.
>>> g.summary()
n nodes: 15
n relation types: 5
n connections: 15
5 most connected nodes: [(u'dog', 2), (u'cat', 2), (u'mouse', 2), (u'rufus', 2), (u'spot', 2)]
5 least connected nodes (non-terminal): [(u'mammal', 1), (u'ball', 1), (u'toy', 1), (u'algernon', 1), (u'chicago', 1)]
n terminal nodes: 5
5 most frequent relations: [(u'IS', 5), (u'MAKES_SOUND', 4), (u'INSTANCE_OF', 3), (u'LOCATED_IN', 2), (u'HAS', 1)]
5 least frequent relations: [(u'IS', 5), (u'MAKES_SOUND', 4), (u'INSTANCE_OF', 3), (u'LOCATED_IN', 2), (u'HAS', 1)]
equivalent nodes: [[u'rufus', u'spot']]
```
## Acknowledgements

The relation-GCN algorithm is taken from an implementation of _Modeling Relational Data with Graph Convolutional Networks_ (2017) by Michael Schlichtkrull, Thomas N. Kipf, Peter Bloem, Rianne van den Berg, Ivan Titov, and Max Welling.
