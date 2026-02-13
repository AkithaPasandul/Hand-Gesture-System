import onnxruntime as ort
import numpy as np
import cv2

class HandLandmarkModel:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, hand_crop):
        img = cv2.resize(hand_crop, (256, 256))  
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    def predict(self, hand_crop):
        input_tensor = self.preprocess(hand_crop)
        output = self.session.run(None, {self.input_name: input_tensor})[0]

        landmarks = np.squeeze(output)

        # If model outputs 63 values
        if landmarks.size == 63:
            landmarks = landmarks.reshape(21, 3)
        else:
            print("Unexpected landmark output shape:", landmarks.shape)
            return None

        return landmarks
