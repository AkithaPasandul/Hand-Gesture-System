import onnxruntime as ort
import numpy as np
import cv2

class YOLOv8HandDetector:
    def __init__(self, model_path, conf_threshold=0.5, iou_threshold=0.4):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def preprocess(self, frame):
        self.original_shape = frame.shape[:2]  # h, w

        img = cv2.resize(frame, (640, 640))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        return img

    def postprocess(self, output):
        predictions = np.squeeze(output)

        boxes = []
        scores = []

        for pred in predictions:
            conf = pred[4]
            if conf > self.conf_threshold:
                x, y, w, h = pred[0:4]

                x1 = x - w / 2
                y1 = y - h / 2
                x2 = x + w / 2
                y2 = y + h / 2

                boxes.append([x1, y1, x2, y2])
                scores.append(conf)

        if len(boxes) == 0:
            return []

        indices = cv2.dnn.NMSBoxes(
            boxes, scores,
            self.conf_threshold,
            self.iou_threshold
        )

        final_boxes = []
        if len(indices) > 0:
            for i in indices.flatten():
                box = boxes[i]
                final_boxes.append(self.rescale_box(box))

        return final_boxes

    def rescale_box(self, box):
        h, w = self.original_shape

        x1, y1, x2, y2 = box

        x1 = int(x1 * w / 640)
        y1 = int(y1 * h / 640)
        x2 = int(x2 * w / 640)
        y2 = int(y2 * h / 640)

        return [x1, y1, x2, y2]

    def detect(self, frame):
        input_tensor = self.preprocess(frame)
        output = self.session.run(None, {self.input_name: input_tensor})[0]
        return self.postprocess(output)
