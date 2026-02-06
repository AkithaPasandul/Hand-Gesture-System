import cv2
import numpy as np
import onnxruntime as ort

class ONNXHandLandmark:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def preprocess(self, frame):
        img = cv2.resize(frame, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return np.expand_dims(img, axis=0)

    def predict(self, frame):
        inp = self.preprocess(frame)
        output = self.session.run(
            [self.output_name],
            {self.input_name: inp}
        )[0]

        # Output shape: (1, 63)
        return output.flatten()
