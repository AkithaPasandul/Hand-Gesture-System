import cv2
from utils.landmark_model import ONNXHandLandmark

model = ONNXHandLandmark("models/hand_landmark.onnx")
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    landmarks = model.predict(frame)

    cv2.putText(
        frame,
        f"Landmarks extracted: {len(landmarks)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("ONNX Hand Landmark Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
