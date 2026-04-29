#!/usr/bin/env python3
"""Bag of Words"""


from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """Creats a bag of words embedding matrix"""

    vectorizer = CountVectorizer(vocabulary=vocab)

    embeddings = vectorizer.fit_transform(sentences).toarray()

    try:
        features = vectorizer.get_feature_names_out()
    except AttributeError:
        features = vectorizer.get_feature_names()

    return embeddings, features
