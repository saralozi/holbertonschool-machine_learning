#!/usr/bin/env python3
"""A function that saves and loads a model from a JSON configuration file"""

import tensorflow.keras as K


def save_config(network, filename):
    """Saves a model's configuration (architecture) in JSON format"""

    json_config = network.to_json()
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(json_config)


def load_config(filename):
    """Loads a model from a JSON configuration file"""

    with open(filename, 'r', encoding='utf-8') as f:
        json_config = f.read()

    return K.models.model_from_json(json_config)
