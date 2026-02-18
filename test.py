import onnxruntime as ort

session = ort.InferenceSession("models/hand_landmark.onnx")

print("Input shape:", session.get_inputs()[0].shape)
print("Output shape:", session.get_outputs()[0].shape)
