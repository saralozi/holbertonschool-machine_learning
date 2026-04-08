#!/usr/bin/env python3
"""A class Yolo that uses the Yolo v3 algorithm
to perform object detection"""


from tensorflow import keras as K


class Yolo:
    """Yolo class"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Class constructor for initialization"""

        self.model = K.models.load_model(model_path)

        with open(classes_path, 'r') as classes_file:
            self.class_names = [
                line.strip() for line in classes_file.readlines()
                ]

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
