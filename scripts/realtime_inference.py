import cv2
import joblib
import json
import time
import numpy as np
from collections import deque

from utils.landmark_model import ONNXHandLandmark
from utils.feature_extractor import normalize_landmarks
from utils.action_executor import ActionExecutor

classifier = joblib.load("models/gesture_classifier.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

with open("configs/gesture_actions.json") as f:
    action_map = json.load(f)

executor = ActionExecutor(action_map)

hand_model = ONNXHandLandmark("models/onnx/hand_landmark.onnx")

cap = cv2.VideoCapture(0)

prediction_buffer = deque(maxlen=7)
last_executed = None
last_action_time = 0
cooldown = 1.0

prev_time = 0

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
        idx = np.argmax(probs)
        confidence = probs[idx]

        if confidence > 0.7:
            gesture = label_encoder.inverse_transform([idx])[0]
            prediction_buffer.append(gesture)
            display_text = max(set(prediction_buffer), key=prediction_buffer.count)

            if display_text != last_executed:
                now = time.time()
                if now - last_action_time > cooldown:
                    executor.execute(display_text)
                    last_executed = display_text
                    last_action_time = now

    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time

    cv2.putText(
        frame, 
        f"Gesture: {display_text}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, (0,255,0), 
        2
    )
    
    cv2.putText(
        frame, 
        f"FPS: {int(fps)}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.8, 
        (255,255,0), 
        2
    )

    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
