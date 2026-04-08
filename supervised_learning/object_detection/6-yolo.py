#!/usr/bin/env python3
"""Implement a show_boxes method in a Yolo class that displays
 an image with detected bounding boxes, class labels, and scores,
   and optionally saves it to a detections folder
   when the user presses s"""


import tensorflow.keras as K
import numpy as np
import glob
import cv2
import os


class Yolo:
    """Yolo class"""
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Class constructor for initialization"""

        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = f.read().split('\n')
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, x):
        """A function that returns the value of sigmoid(x) expression"""

        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """A function to process outputs"""

        image_height, image_width = image_size
        boxes, box_confidences, box_class_probs = [], [], []

        for output in range(len(outputs)):
            grid_h, grid_w, anchor_boxes = outputs[output].shape[:3]
            classes = outputs[output].shape[3] - 5
            input_width = int(self.model.input.shape[1])
            input_height = int(self.model.input.shape[2])
            boxes_tmp = np.zeros((grid_h, grid_w, anchor_boxes, 4))
            conf_tmp = np.zeros((grid_h, grid_w, anchor_boxes, 1))
            prob_tmp = np.zeros((grid_h, grid_w, anchor_boxes, classes))
            for row in range(grid_h):
                for col in range(grid_w):
                    for box in range(anchor_boxes):
                        tx, ty, tw, th = outputs[output][row][col][box][:4]
                        pw = self.anchors[output, box, 0]
                        ph = self.anchors[output, box, 1]

                        bx = self.sigmoid(tx) + col
                        by = self.sigmoid(ty) + row
                        bw = pw * np.exp(tw)
                        bh = ph * np.exp(th)
                        bx /= grid_w
                        by /= grid_h
                        bw /= input_width
                        bh /= input_height
                        x1 = (bx - bw / 2) * image_width
                        x2 = (bx + bw / 2) * image_width
                        y1 = (by - bh / 2) * image_height
                        y2 = (by + bh / 2) * image_height
                        boxes_tmp[row][col][box] = np.array([x1, y1, x2, y2])

                        pc = outputs[output][row][col][box][4]
                        conf_tmp[row, col, box, 0] = self.sigmoid(pc)

                        class_probs = outputs[output][row][col][box][5:]
                        prob_tmp[row, col, box] = self.sigmoid(class_probs)

            boxes.append(boxes_tmp)
            box_confidences.append(conf_tmp)
            box_class_probs.append(prob_tmp)

        return (boxes, box_confidences, box_class_probs)

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """A method to filter boxes of preprocess method"""

        filtered_boxes = []
        box_classes = []
        box_scores = []
        for i in range(len(boxes)):
            probs = box_confidences[i][:, :, :, :] *\
                          box_class_probs[i][:, :, :, :]

            predictions = np.amax(probs, axis=3)

            kept = np.argwhere(predictions[:, :, :] > self.class_t)

            for idx in kept:
                filtered_boxes.append(boxes[i][idx[0], idx[1], idx[2]])
                box_scores.append(predictions[idx[0], idx[1], idx[2]])
                box_classes.append(np.argmax(probs[idx[0], idx[1], idx[2]]))

        filtered_boxes = np.array(filtered_boxes)
        box_classes = np.array(box_classes)
        box_scores = np.array(box_scores)
        return (filtered_boxes, box_classes, box_scores)

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """A method to apply Non-max Suppression"""

        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []
        for label in range(len(self.class_names)):
            bound_tmp = []
            class_tmp = []
            score_tmp = []
            for i in range(len(box_classes)):
                if box_classes[i] == label:
                    bound_tmp.append(filtered_boxes[i])
                    class_tmp.append(box_classes[i])
                    score_tmp.append(box_scores[i])

            class_tmp = np.array(class_tmp)
            while len(class_tmp) > 0 and np.amax(class_tmp) > -1:
                index = np.argmax(score_tmp)

                box_predictions.append(bound_tmp[index])
                predicted_box_classes.append(class_tmp[index])
                predicted_box_scores.append(score_tmp[index])

                score_tmp[index] = -1
                class_tmp[index] = -1

                px1, py1, px2, py2 = bound_tmp[index]
                p_area = (px2 - px1) * (py2 - py1)

                for box in range(len(bound_tmp)):
                    if class_tmp[box] != -1:
                        bx1, by1, bx2, by2 = bound_tmp[box]
                        ox1 = px1 if px1 > bx1 else bx1
                        oy1 = py1 if py1 > by1 else by1
                        ox2 = px2 if px2 < bx2 else bx2
                        oy2 = py2 if py2 < by2 else by2

                        if ox2 - ox1 <= 0 or oy2 - oy1 <= 0:
                            continue

                        o_area = (ox2 - ox1) * (oy2 - oy1)
                        iou = o_area / p_area

                        if iou > self.nms_t:
                            class_tmp[box] = -1
                            score_tmp[box] = -1

        box_predictions = np.array(box_predictions)
        predicted_box_classes = np.array(predicted_box_classes)
        predicted_box_scores = np.array(predicted_box_scores)
        return (box_predictions, predicted_box_classes, predicted_box_scores)

    @staticmethod
    def load_images(folder_path):
        """A function to load images"""

        images = []
        image_paths = glob.glob(folder_path + "/*")
        for path in image_paths:
            images.append(cv2.imread(path))

        return (images, image_paths)

    def preprocess_images(self, images):
        """A function to preprocess images (resize & return to tuple)"""

        input_h = int(self.model.input.shape[2])
        input_w = int(self.model.input.shape[1])
        pimages = []
        image_shapes = []

        for image in images:
            original_height, original_width = image.shape[:2]
            image_shapes.append([original_height, original_width])

            resized = cv2.resize(
                image,
                (input_w, input_h),
                interpolation=cv2.INTER_CUBIC
            )
            normalized = resized / 255
            pimages.append(normalized)

        pimages = np.array(pimages)
        image_shapes = np.array(image_shapes)
        return (pimages, image_shapes)

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """Display image with all boundary boxes"""

        for box in range(len(boxes)):
            x1, y1, x2, y2 = boxes[box]
            image = cv2.rectangle(
                image,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (255, 0, 0),
                2
            )
            image = cv2.putText(
                image,
                "{} {:.2f}".format(self.class_names[box_classes[box]],
                                   box_scores[box]),
                (int(x1), int(y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA
            )
        cv2.imshow(file_name, image)
        key = cv2.waitKey(0)
        if key == ord('s'):
            if not os.path.exists("./detections"):
                os.mkdir("detections")
            cv2.imwrite("./detections/" + file_name, image)
        cv2.destroyAllWindows()
