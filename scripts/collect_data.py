import cv2
import numpy as np
import os
from utils.detector import YOLOv8HandDetector
from utils.landmark_model import HandLandmarkModel
from utils.feature_extractor import normalize_landmarks

DETECTOR_MODEL = "models/hand_detector.onnx"
LANDMARK_MODEL = "models/hand_landmarks.onnx"

SAVE_PATH = "data"
LABEL = input("Enter gesture label: ")

os.makedirs(SAVE_PATH, exist_ok=True)

detector = YOLOv8HandDetector(DETECTOR_MODEL)
landmark_model = HandLandmarkModel(LANDMARK_MODEL)

cap = cv2.VideoCapture(0)

samples = []

print("Press 's' to save sample. Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    boxes = detector.detect(frame)

    for box in boxes:
        x1, y1, x2, y2 = box
        hand_crop = frame[y1:y2, x1:x2]

        if hand_crop.size != 0:
            landmarks = landmark_model.predict(hand_crop)

            if landmarks is not None:
                features = normalize_landmarks(landmarks)

                cv2.putText(frame, f"Samples: {len(samples)}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)

                key = cv2.waitKey(1)

                if key == ord('s'):
                    samples.append(features)
                    print("Sample saved:", len(samples))

    cv2.imshow("Collecting Data", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

if len(samples) > 0:
    file_path = os.path.join(SAVE_PATH, f"{LABEL}.npy")
    np.save(file_path, np.array(samples))
    print("Saved to", file_path)
