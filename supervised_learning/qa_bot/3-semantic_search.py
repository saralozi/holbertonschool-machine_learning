#!/usr/bin/env python3
"""Semantic search module."""

import os
import numpy as np
import tensorflow_hub as hub


def semantic_search(corpus_path, sentence):
    """
    Performs semantic search on a corpus of documents.

    Args:
        corpus_path (str): Path to the corpus of reference documents.
        sentence (str): Sentence to compare against the corpus.

    Returns:
        str: The reference text of the document most similar to sentence.
    """
    documents = []

    for filename in os.listdir(corpus_path):
        filepath = os.path.join(corpus_path, filename)

        if os.path.isfile(filepath):
            with open(filepath, "r", encoding="utf-8") as file:
                documents.append(file.read())

    model = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    embeddings = model(documents + [sentence])

    document_embeddings = embeddings[:-1]
    sentence_embedding = embeddings[-1]

    similarities = np.inner(document_embeddings, sentence_embedding)

    best_match_index = np.argmax(similarities)

    return documents[best_match_index]
