import cv2
import csv
import os
import json
import mediapipe as mp
import numpy as np

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "gesture_list.json")
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "landmarks", "static")

os.makedirs(DATASET_PATH, exist_ok=True)

# -----------------------------
# Load gesture mapping
# -----------------------------
with open(CONFIG_PATH, "r") as f:
    gesture_map = json.load(f)

# Reverse mapping for keyboard control
key_to_gesture = {
    '0': gesture_map["0"],
    '1': gesture_map["1"],
    '2': gesture_map["2"],
    '3': gesture_map["3"]
}

# -----------------------------
# MediaPipe setup
# -----------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# -----------------------------
# Camera
# -----------------------------
cap = cv2.VideoCapture(0)

print("Press keys to collect data:")
print("0 → open_palm | 1 → fist | 2 → thumbs_up | 3 → peace")
print("Press 'q' to quit")

# -----------------------------
# Main loop
# -----------------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            key = cv2.waitKey(1) & 0xFF
            if chr(key) in key_to_gesture:
                gesture_name = key_to_gesture[chr(key)]
                file_path = os.path.join(DATASET_PATH, f"{gesture_name}.csv")

                write_header = not os.path.exists(file_path)

                with open(file_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    if write_header:
                        header = [f"lm_{i}_{axis}" for i in range(21) for axis in ["x", "y", "z"]]
                        writer.writerow(header)
                    writer.writerow(landmarks)

                print(f"Saved sample for: {gesture_name}")

    cv2.imshow("Data Collection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
