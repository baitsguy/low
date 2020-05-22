from video_feed import VideoFeed
import cv2
import numpy as np
from collections import defaultdict


class ObjectDetector:


    def __init__(self) -> None:
        super().__init__()
        self.yolo_path = "yolo/final"
        # self.yolo_path = "yolo/v2"
        # self.yolo_path = "yolo/v1"
        self.yolo = cv2.dnn.readNet(self.yolo_path + "/yolov3.weights", self.yolo_path + "/yolov3.cfg")
        self.scale = 0.00392

        self.classes = None
        self.WIDTH = 470

        with open(self.yolo_path + "/yolov3.txt", 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]



        self.confidence_thresholds = defaultdict(lambda: 0.5)
        self.confidence_thresholds["cat"] = 0.5
        self.confidence_thresholds["waterbowl"] = 0.5
        self.confidence_thresholds["foodbowl"] = 0.5

        self.nms_thresholds = defaultdict(lambda: 0.5)
        self.nms_thresholds["cat"] = 0.01
        self.nms_thresholds["waterbowl"] = 0.5
        self.nms_thresholds["foodbowl"] = 0.5

    def get_output_layers(self, net):
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        return output_layers


    def get_objects(self, frame):
        width = frame.shape[1]
        height = frame.shape[0]
        blob = cv2.dnn.blobFromImage(frame, self.scale, (self.WIDTH, self.WIDTH), (0, 0, 0), True,
                                     crop=False)  # Taking the largest dimension
        self.yolo.setInput(blob)
        outs = self.yolo.forward(self.get_output_layers(self.yolo))

        boxes = defaultdict(list)
        confidences = defaultdict(list)

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                type = self.classes[class_id]
                confidence = scores[class_id]
                if confidence > self.confidence_thresholds[type]:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = center_x - w / 2
                    y = center_y - h / 2

                    # print(type + " " + str(confidence))
                    boxes[type].append([x, y, w, h])
                    confidences[type].append(float(confidence))

        # draw_outputs(type, boxes[type], confidences[type], confidence_thresholds[type], nms_thresholds[type])
        result = {}
        for type in self.classes:
            indices = cv2.dnn.NMSBoxes(boxes[type], confidences[type], self.confidence_thresholds[type], self.nms_thresholds[type])
            for i in indices:
                i = i[0]
                result[type] = result.get(type, [])
                b = {}
                b['confidence'] = confidences[type][i]
                box = boxes[type][i]
                b['x'] = box[0]
                b['y'] = box[1]
                b['w'] = box[2]
                b['h'] = box[3]
                result[type].append(b)
        return result