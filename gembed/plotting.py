import numpy as np
from future.utils import iteritems
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

def scatterplot2D(x_coords, y_coords, vector_labels):
    # vector labels are the labels for each individual point
    plt.figure(figsize=(8,6),dpi=400)
    plt.scatter(x_coords, y_coords, alpha=0.85)
    # label each point
    for i, txt in enumerate(vector_labels):
        plt.annotate(txt, (x_coords[i], y_coords[i]))
    #draw the scatter plot with some padding
    x_pad = (x_coords.max()-x_coords.min())*.02
    y_pad = (y_coords.max()-y_coords.min())*.02
    plt.xlim(x_coords.min()-x_pad, x_coords.max()+x_pad)
    plt.ylim(y_coords.min()-y_pad, y_coords.max()+y_pad)
    plt.show()

def scatterplot2D_legend(x_coords, y_coords, vector_labels):
    # vector labels gives the category that each point belongs to
    # Associate a color to each unique label  
    label_dict = {j:i for (i,j) in enumerate(set(vector_labels))}
    cmap = plt.cm.rainbow
    norm = mcolors.Normalize(vmin=0, vmax=len(label_dict))
    plt.figure(figsize=(8,6),dpi=200)
    plt.scatter(x_coords, y_coords, alpha=0.85, c=[cmap(norm(label_dict[i])) for i in vector_labels])
    #draw the legend
    patches = []
    for k,v in iteritems(label_dict):
        patches.append(mpatches.Patch(color=cmap(norm(v)), label=k))
    plt.legend(handles=patches)
    #draw the scatter plot with some padding
    x_pad = (x_coords.max()-x_coords.min())*.02
    y_pad = (y_coords.max()-y_coords.min())*.02
    plt.xlim(x_coords.min()-x_pad, x_coords.max()+x_pad)
    plt.ylim(y_coords.min()-y_pad, y_coords.max()+y_pad)
    plt.show()

from sklearn.manifold import TSNE
def tsne2D(vectors):
    # find tsne coords for 2 dimensions
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(vectors)
    return Y

import umap
def umap2D(vectors):
    # find umap coords for 2 dimensions
    umapper = umap.UMAP()
    np.set_printoptions(suppress=True)
    Y = umapper.fit_transform(vectors)
    return Y
 
from sklearn.decomposition import PCA
def pca2D(vectors):
    # find PCA coords for 2 dimensions
    pca = PCA(n_components=2)
    np.set_printoptions(suppress=True)
    Y = pca.fit_transform(vectors)
    return Y
