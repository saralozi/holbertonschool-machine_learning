#!/usr/bin/env python3
"""Unigram BLEU score module."""

import math


def uni_bleu(references, sentence):
    """
    Calculates the unigram BLEU score for a sentence.

    Args:
        references (list): List of reference translations.
        sentence (list): Model proposed sentence.

    Returns:
        float: The unigram BLEU score.
    """
    sentence_length = len(sentence)

    if sentence_length == 0:
        return 0

    reference_lengths = [len(reference) for reference in references]

    closest_ref_length = min(
        reference_lengths,
        key=lambda ref_len: (abs(ref_len - sentence_length), ref_len)
    )

    sentence_counts = {}

    for word in sentence:
        sentence_counts[word] = sentence_counts.get(word, 0) + 1

    max_reference_counts = {}

    for reference in references:
        reference_counts = {}

        for word in reference:
            reference_counts[word] = reference_counts.get(word, 0) + 1

        for word, count in reference_counts.items():
            max_reference_counts[word] = max(
                max_reference_counts.get(word, 0),
                count
            )

    clipped_count = 0

    for word, count in sentence_counts.items():
        clipped_count += min(count, max_reference_counts.get(word, 0))

    precision = clipped_count / sentence_length

    if sentence_length > closest_ref_length:
        brevity_penalty = 1
    else:
        brevity_penalty = math.exp(1 - closest_ref_length / sentence_length)

    return brevity_penalty * precision
