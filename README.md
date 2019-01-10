# Graph Embeddings

A package for finding embedding vectors for relational graphs. 


## Installation
This package was created with Python 2.7 and works with Keras 2.2.4 using Tensorflow 1.12.0 as the backend. It's recommended to use an anaconda instance to set up the environment in order to ensure no dependencies conflict. You can download the anaconda package manager here: https://www.anaconda.com/download/

conda create --name gembed python=2.7
conda activate gembed

git clone https://github.com/bhaney/graph_embeddings
cd graph_embeddings
pip install git+https://github.com/bhaney/relational-gcn.git
python setup.py install
cd gembed
python get_graph_embeddings.py -a rgcn -p data/aifb -t data/completeDataset.csv -e 50 -d 16 -n rgcn
python get_graph_embeddings.py -a auto -p data/aifb -e 5 -d 16 -n autoencode

To plot the AIFB embeddings, there is an iPython notebook you can use.

conda install jupyter
conda install scikit-learn
conda install matplotlib
conda install -c conda-forge umap-learn
jupyter notebook

Then select plot_aifb_graph_embeddings.ipynb

## Usage

Input a CSV file where the graph is represented by triplet connection

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
