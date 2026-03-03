#!/usr/bin/env python3
"""A function that flips an image horizontally"""


import tensorflow as tf


def flip_image(image):
    """Flip and image horizontally"""

    return tf.image.flip_left_right(image)
