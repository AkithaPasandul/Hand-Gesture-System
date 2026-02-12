import numpy as np

def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks)

    # If flattened (63,), reshape it
    if landmarks.ndim == 1:
        landmarks = landmarks.reshape(21, 3)

    origin = landmarks[0]
    landmarks = landmarks - origin

    max_dist = np.max(np.linalg.norm(landmarks, axis=1))
    if max_dist > 0:
        landmarks = landmarks / max_dist

    return landmarks.flatten()
