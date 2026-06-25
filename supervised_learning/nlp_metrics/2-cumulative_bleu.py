#!/usr/bin/env python3
"""Cumulative N-gram BLEU score module."""

import math


def get_ngrams(words, n):
    """
    Creates n-grams from a list of words.

    Args:
        words (list): List of words.
        n (int): Size of the n-gram.

    Returns:
        list: List of n-grams as tuples.
    """
    return [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]


def cumulative_bleu(references, sentence, n):
    """
    Calculates the cumulative n-gram BLEU score for a sentence.

    Args:
        references (list): List of reference translations.
        sentence (list): Model proposed sentence.
        n (int): Largest n-gram size to use.

    Returns:
        float: The cumulative n-gram BLEU score.
    """
    sentence_length = len(sentence)

    if sentence_length == 0:
        return 0

    reference_lengths = [len(reference) for reference in references]

    closest_ref_length = min(
        reference_lengths,
        key=lambda ref_len: (abs(ref_len - sentence_length), ref_len)
    )

    if sentence_length > closest_ref_length:
        brevity_penalty = 1
    else:
        brevity_penalty = math.exp(1 - closest_ref_length / sentence_length)

    precision_product = 1

    for i in range(1, n + 1):
        sentence_ngrams = get_ngrams(sentence, i)

        if len(sentence_ngrams) == 0:
            return 0

        sentence_counts = {}

        for ngram in sentence_ngrams:
            sentence_counts[ngram] = sentence_counts.get(ngram, 0) + 1

        max_reference_counts = {}

        for reference in references:
            reference_ngrams = get_ngrams(reference, i)
            reference_counts = {}

            for ngram in reference_ngrams:
                reference_counts[ngram] = reference_counts.get(ngram, 0) + 1

            for ngram, count in reference_counts.items():
                max_reference_counts[ngram] = max(
                    max_reference_counts.get(ngram, 0),
                    count
                )

        clipped_count = 0

        for ngram, count in sentence_counts.items():
            clipped_count += min(count, max_reference_counts.get(ngram, 0))

        precision = clipped_count / len(sentence_ngrams)

        if precision == 0:
            return 0

        precision_product *= precision

    geo_mean = precision_product ** (1 / n)

    return brevity_penalty * geo_mean
