#!/usr/bin/env python3
"""Question Answering module using BERT."""

import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(question, reference):
    """
    Finds a snippet of text within a reference document to answer a question.

    Args:
        question (str): The question to answer.
        reference (str): The reference document.

    Returns:
        str: The answer found in the reference document.
        None: If no answer is found.
    """
    tokenizer = BertTokenizer.from_pretrained(
        "bert-large-uncased-whole-word-masking-finetuned-squad"
    )

    model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")

    question_tokens = tokenizer.tokenize(question)
    reference_tokens = tokenizer.tokenize(reference)

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

    if start_index > end_index:
        return None

    answer_tokens = tokens[start_index:end_index + 1]
    answer = tokenizer.convert_tokens_to_string(answer_tokens)

    if answer is None or answer.strip() == "":
        return None

    return answer
