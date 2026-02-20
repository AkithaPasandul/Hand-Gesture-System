import cv2
import time
from collections import deque, Counter
from utils.detector import YOLOv8HandDetector
from utils.landmark_model import HandLandmarkModel
from utils.feature_extractor import normalize_landmarks
from utils.classifier import GestureClassifier

# CONFIG
DETECTOR_MODEL_PATH = "models/hand_detector.onnx"
LANDMARK_MODEL_PATH = "models/hand_landmarks.onnx"
CLASSIFIER_PATH = "models/gesture_classifier.pkl"
landmark_model = HandLandmarkModel(LANDMARK_MODEL_PATH)
classifier = GestureClassifier(CLASSIFIER_PATH)

# INIT
detector = YOLOv8HandDetector(DETECTOR_MODEL_PATH)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open camera.")
    exit()

prev_time = 0
prediction_buffer = deque(maxlen=10)

# MAIN LOOP
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    start_time = time.time()

    # Detect hands
    boxes = detector.detect(frame)

    # Draw detections
    for box in boxes:
        x1, y1, x2, y2 = box

        # Safety clamp
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Crop hand
        hand_crop = frame[y1:y2, x1:x2]

        # Show cropped hand in corner (debug)
        if hand_crop.size != 0:
            landmarks = landmark_model.predict(hand_crop)
            
            if landmarks is not None:
                
                for lm in landmarks:
                    lx = int(x1 + lm[0] * (x2 - x1))
                    ly = int(y1 + lm[1] * (y2 - y1))                 
                    cv2.circle(frame, (lx, ly), 4, (0, 0, 255), -1)
                    
                features = normalize_landmarks(landmarks)
                gesture = classifier.predict(features)
                prediction_buffer.append(gesture)
                
                # Majority vote smoothing
                most_common = Counter(prediction_buffer).most_common(1)[0][0]
                gesture = most_common
                
                cv2.putText(
                    frame,
                    f"Gesture: {gesture}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 255),
                    2
                )
                    
    # FPS Calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (10, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.imshow("Hand Gesture System - YOLOv8", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
