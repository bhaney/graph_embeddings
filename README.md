# Graph Embeddings

A package for finding embedding vectors for relational graphs. 


## Installation
This package was created with Python 2.7 and works with Keras 2.2.4 using Tensorflow 1.12.0 as the backend. It's recommended to use an anaconda instance to set up the environment in order to ensure no dependencies conflict. You can download the anaconda package manager here: https://www.anaconda.com/download/

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
python get_graph_embeddings.py -a rgcn -p data/aifb -t data/person_affiliations.csv -e 50 -d 16 -n rgcn
python get_graph_embeddings.py -a auto -p data/aifb -e 5 -d 16 -n autoencode
```
5. To plot the AIFB embeddings, there is an iPython notebook you can use.
```
conda install jupyter
conda install scikit-learn
conda install matplotlib
conda install -c conda-forge umap-learn
jupyter notebook
```
Then select `plot_aifb_graph_embeddings.ipynb`

## Usage

The program `get_graph_embeddings.py` takes the following arguments:

  -p PATH, --path PATH,  Provide path to the directory of .CSV files.
  -f FILES, --files FILES, If you only want to run over one file, give path of the single CSV file..

  -n NAME, --name NAME,  name of the output file of the embeddings in the `results/` directory.
  -a ALGO, --algo ALGO,  which algorithm to use. `auto` or `rgcn` available.
  -d DIM, --dim DIM,     desired embedding dimension.
  -e EPOCHS, --epochs EPOCHS, number of epochs to train.
  -t TARGET, --target TARGET, csv file with targets for training.


The CSV file has to represent the graph by triplet connection of (subject_node, relation_edge, predicate_node). Each row should contain one connection. 

Example:
```
dog,IS,mammal
cat,IS,mammal
mouse,IS,mammal
mammal,HAS,fur
rufus,INSTANCE_OF,dog
algernon,INSTANCE_OF,mouse
rufus,LOCATED_IN,chicago
chicago,IS,city
```

The output of the program will be a list of `(node_name, node_embeddings)`. It will be saved as a `.pkl` file in the `results/` directory.
