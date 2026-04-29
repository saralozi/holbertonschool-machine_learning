#!/usr/bin/env python3
"""TF-IDF"""

from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """Create a TF-IDF embedding matrix"""

    vectorizer = TfidfVectorizer(vocabulary=vocab)

    embeddings = vectorizer.fit_transform(sentences).toarray()

    try:
        features = vectorizer.get_feature_names_out()
    except AttributeError:
        features = vectorizer.get_feature_names()

    return embeddings, features
