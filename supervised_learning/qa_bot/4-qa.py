#!/usr/bin/env python3
"""Multi-reference question answering module."""

import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


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


def answer_question(question, reference):
    """
    Finds a snippet of text within a reference document to answer a question.

    Args:
        question (str): The question to answer.
        reference (str): The reference document.

    Returns:
        str: The answer found in the reference document.
        None: If no valid answer is found.
    """
    tokenizer = BertTokenizer.from_pretrained(
        "bert-large-uncased-whole-word-masking-finetuned-squad"
    )

    model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")

    question_tokens = tokenizer.tokenize(question)
    reference_tokens = tokenizer.tokenize(reference)

    max_reference_len = 512 - len(question_tokens) - 3
    reference_tokens = reference_tokens[:max_reference_len]

    tokens = (
        ["[CLS]"] +
        question_tokens +
        ["[SEP]"] +
        reference_tokens +
        ["[SEP]"]
    )

    input_word_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_word_ids)

    input_type_ids = (
        [0] * (len(question_tokens) + 2) +
        [1] * (len(reference_tokens) + 1)
    )

    input_word_ids = tf.expand_dims(input_word_ids, 0)
    input_mask = tf.expand_dims(input_mask, 0)
    input_type_ids = tf.expand_dims(input_type_ids, 0)

    outputs = model([input_word_ids, input_mask, input_type_ids])

    start_scores = outputs[0][0]
    end_scores = outputs[1][0]

    start_index = tf.argmax(start_scores).numpy()
    end_index = tf.argmax(end_scores).numpy()

    if start_index == 0 or end_index == 0:
        return None

    if start_index > end_index:
        return None

    answer_tokens = tokens[start_index:end_index + 1]
    answer = tokenizer.convert_tokens_to_string(answer_tokens)

    if answer is None or answer.strip() == "":
        return None

    return answer


def question_answer(corpus_path):
    """
    Answers questions from multiple reference texts.

    Args:
        corpus_path (str): Path to the corpus of reference documents.
    """
    exit_words = {"exit", "quit", "goodbye", "bye"}

    while True:
        question = input("Q: ")

        if question.lower() in exit_words:
            print("A: Goodbye")
            break

        reference = semantic_search(corpus_path, question)
        answer = answer_question(question, reference)

        if answer is None:
            print("A: Sorry, I do not understand your question.")
        else:
            print("A: {}".format(answer))
