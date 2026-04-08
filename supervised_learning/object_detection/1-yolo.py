#!/usr/bin/env python3
"""Implement a process_outputs mehtod in a Yolo class that converts raw
Darknet model predictions into scaled bounding boxes, confidence scores
and class probabilities relative to the original image size"""


import tensorflow as tf
import numpy as np


class Yolo:
    """Yolo class"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Class constructor for initialization"""

        self.model = tf.keras.models.load_model(model_path)
        self.class_names = []
        with open(classes_path, 'r') as f:
            for line in f:
                line = line.strip()
                self.class_names.append(line)
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """A function to process outputs"""

        image_height, image_height = image_size

        boxes = []
        box_confidences = []
        box_class_probs = []

        for idx, output in enumerate(outputs):

            grid_height, grid_width, nbr_anchor, _ = output.shape

            t_x = output[:, :, :, 0]
            t_y = output[:, :, :, 1]
            t_w = output[:, :, :, 2]
            t_h = output[:, :, :, 3]

            grid_x, grid_y = np.meshgrid(np.arange(grid_width),
                                         np.arange(grid_height))

            grid_x = np.expand_dims(grid_x, axis=-1)
            grid_y = np.expand_dims(grid_y, axis=-1)

            p_w = self.anchors[idx, :, 0]
            p_h = self.anchors[idx, :, 1]

            image_height, image_width = image_size

            b_x = ((1.0 / (1.0 + np.exp(-t_x))) + grid_x) / grid_width
            b_y = ((1.0 / (1.0 + np.exp(-t_y))) + grid_y) / grid_height
            b_w = p_w * np.exp(t_w)
            b_w /= self.model.input.shape[1]
            b_h = p_h * np.exp(t_h)
            b_h /= self.model.input.shape[2]

            x1 = (b_x - b_w / 2) * image_width
            y1 = (b_y - b_h / 2) * image_height
            x2 = (b_w / 2 + b_x) * image_width
            y2 = (b_h / 2 + b_y) * image_height

            box = np.zeros((grid_height, grid_width, nbr_anchor, 4))
            box[:, :, :, 0] = x1
            box[:, :, :, 1] = y1
            box[:, :, :, 2] = x2
            box[:, :, :, 3] = y2
            boxes.append(box)

            confidences = output[:, :, :, 4:5]
            sigmoid_confidence = 1 / (1 + np.exp(-confidences))
            class_probs = output[:, :, :, 5:]
            sigmoid_class_probs = 1 / (1 + np.exp(-class_probs))

            box_confidences.append(sigmoid_confidence)
            box_class_probs.append(sigmoid_class_probs)

        return boxes, box_confidences, box_class_probs
