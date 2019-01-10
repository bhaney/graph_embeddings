from __future__ import print_function

from gembed.layers import DenseTied
from gembed.multigraph import Multigraph

from keras.layers import Input, Dense
from keras.models import Model

def autoencoder_model(input_dim, encoding_dim):
    #Define inputs
    inputs = Input(shape=(input_dim,), sparse=True, name="inputs")
    # Encoder Layers
    encode_layer1 = Dense(4 * encoding_dim, activation='relu', name="encode_layer1")
    encode_layer2 = Dense(2 * encoding_dim, activation='relu', name="encode_layer2")
    coding_layer = Dense(encoding_dim, activation='relu', name="coding_layer")

    encoding_1 = encode_layer1(inputs)
    encoding_2 = encode_layer2(encoding_1)
    the_code = coding_layer(encoding_2)

    # Decoder Layers
    decode_layer1 = DenseTied(coding_layer, activation='relu',name="decode_layer1")
    decode_layer2 = DenseTied(encode_layer2, activation='relu',name="decode_layer2")
    recon_layer = DenseTied(encode_layer1, activation='relu',name="reconstruction_layer")

    decoding_1 = decode_layer1(the_code)
    decoding_2 = decode_layer2(decoding_1)
    reconstruction = recon_layer(decoding_2)

    training_model = Model(inputs=inputs, outputs=reconstruction)
    embedding_model = Model(inputs=inputs, outputs=the_code)
    training_model.compile(optimizer='adam', loss='mse')
    return (embedding_model, training_model)

def train_autoencoder(model, data, epochs=1):
    model.fit(data, y=data, epochs=epochs, batch_size=data.shape[1], verbose=1)

def autoencoder(graph, embedding_dim, epochs=1):
    num_nodes = graph.n_nodes
    num_relations = graph.n_rels
    # get inputs for model
    A = graph.get_adjacency_matrix()
    # make model for embedding
    input_dim = A.shape[1]
    embedding_model, training_model = autoencoder_model(input_dim, embedding_dim)
    #train model
    train_autoencoder(training_model, A, epochs=epochs)
    #get embeddings
    embeddings = embedding_model.predict(A)
    return embeddings
