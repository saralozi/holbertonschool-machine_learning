#!/usr/bin/env python3
"""Convert Gensim Word2Vec model to Keras Embedding layer"""

from keras.layers import Embedding


def gensim_to_keras(model):
    """Convert a gensim Word2Vec model to a Keras Embedding layer"""

    weights = model.wv.vectors

    embedding = Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        weights=[weights],
        trainable=True
    )

    return embedding
