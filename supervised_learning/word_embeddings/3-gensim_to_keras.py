#!/usr/bin/env python3
"""Convert Gensim Word2Vec model to Keras Embedding layer"""

import tensorflow as tf


def gensim_to_keras(model):
    """Convert a gensim Word2Vec model to a Keras Embedding layer"""

    weights = model.wv.vectors

    embedding = tf.keras.layers.Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        embeddings_initializer=tf.keras.initializers.Constant(weights),
        trainable=True
    )

    return embedding
