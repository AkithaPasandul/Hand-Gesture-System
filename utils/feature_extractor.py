import numpy as np

def normalize_landmarks(landmarks):

    landmarks = np.array(landmarks)

    # Use the first landmark (wrist) as the origin
    origin = landmarks[0]
    landmarks = landmarks - origin

    # Scale by max distance (hand size normalization)
    max_dist = np.max(np.linalg.norm(landmarks, axis=1))
    if max_dist > 0:
        landmarks = landmarks / max_dist

    return landmarks.flatten()
