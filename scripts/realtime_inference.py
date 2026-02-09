import cv2
import joblib
import numpy as np
from collections import deque

from utils.onnx_hand_landmarks import ONNXHandLandmark
from utils.feature_extractor import normalize_landmarks

# CONFIG
MODEL_PATH = "models/gesture_classifier.pkl"
ENCODER_PATH = "models/label_encoder.pkl"
SMOOTHING_WINDOW = 7
CONFIDENCE_THRESHOLD = 0.7

# LOAD MODEL
classifier = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

hand_model = ONNXHandLandmark("models/onnx/hand_landmark.onnx")

# CAMERA SETUP
cap = cv2.VideoCapture(0)
prediction_buffer = deque(maxlen=SMOOTHING_WINDOW)

print("Starting real-time gesture recognition...")
print("Press 'q' to quit")

# MAIN LOOP
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    landmarks = hand_model.predict(frame)
    display_text = "No Hand"

    if landmarks is not None:
        features = normalize_landmarks(landmarks).reshape(1, -1)

        probs = classifier.predict_proba(features)[0]
        pred_idx = np.argmax(probs)
        confidence = probs[pred_idx]

        if confidence >= CONFIDENCE_THRESHOLD:
            gesture = label_encoder.inverse_transform([pred_idx])[0]
            prediction_buffer.append(gesture)

            # Majority voting
            display_text = max(set(prediction_buffer), key=prediction_buffer.count)

    # UI
    cv2.putText(
        frame,
        f"Gesture: {display_text}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3
    )

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
