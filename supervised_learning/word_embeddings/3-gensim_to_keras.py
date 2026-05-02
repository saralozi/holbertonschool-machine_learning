#!/usr/bin/env python3
"""Convert Gensim Word2Vec model to Keras Embedding layer"""

import tensorflow as tf


def gensim_to_keras(model):
    """A function that converts a gensim Word2Vec
    model to a Keras Embedding layer"""

    keys = model.wv
    weights = keys.vectors

    return tf.keras.layers.Embedding(input_dim=weights.shape[0],
                                     output_dim=weights.shape[1],
                                     weights=[weights],
                                     trainable=True)
