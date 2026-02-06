import cv2
import json
import os
import numpy as np

from utils.onnx_hand_landmarks import ONNXHandLandmark
from utils.feature_extractor import normalize_landmarks

# -----------------------------
# CONFIG
# -----------------------------
DATA_DIR = "data/raw"
GESTURE_FILE = "configs/gesture_list.json"
SAMPLES_PER_GESTURE = 200

os.makedirs(DATA_DIR, exist_ok=True)

# -----------------------------
# LOAD GESTURES
# -----------------------------
with open(GESTURE_FILE, "r") as f:
    gestures = json.load(f)["gestures"]

print("Gestures:", gestures)

# -----------------------------
# INIT MODEL & CAMERA
# -----------------------------
hand_model = ONNXHandLandmark("models/onnx/hand_landmark.onnx")
cap = cv2.VideoCapture(0)

current_gesture = None
collected = 0
features_buffer = []

print("\nControls:")
print("Press number key (1-9) to select gesture")
print("Press 's' to save samples")
print("Press 'q' to quit\n")

# -----------------------------
# MAIN LOOP
# -----------------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    landmarks = hand_model.predict(frame)

    if landmarks is not None and current_gesture is not None:
        features = normalize_landmarks(landmarks)
        features_buffer.append(features)
        collected += 1

    # -----------------------------
    # UI
    # -----------------------------
    cv2.putText(
        frame,
        f"Gesture: {current_gesture}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Samples: {collected}/{SAMPLES_PER_GESTURE}",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    cv2.imshow("Collect Hand Gesture Data", frame)

    key = cv2.waitKey(1) & 0xFF

    # -----------------------------
    # KEY CONTROLS
    # -----------------------------
    if key == ord('q'):
        break

    # Select gesture (1,2,3,4...)
    if ord('1') <= key <= ord(str(len(gestures))):
        idx = key - ord('1')
        current_gesture = gestures[idx]
        collected = 0
        features_buffer = []
        print(f"\nSelected gesture: {current_gesture}")

    # Save samples
    if key == ord('s') and current_gesture is not None:
        if len(features_buffer) >= SAMPLES_PER_GESTURE:
            save_path = os.path.join(DATA_DIR, f"{current_gesture}.npy")
            np.save(save_path, np.array(features_buffer))
            print(f"Saved {len(features_buffer)} samples to {save_path}")
            collected = 0
            features_buffer = []
        else:
            print("Not enough samples collected yet!")

# -----------------------------
# CLEANUP
# -----------------------------
cap.release()
cv2.destroyAllWindows()
