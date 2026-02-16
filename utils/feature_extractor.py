import numpy as np

def normalize_landmarks(landmarks):
    # Use wrist (index 0) as origin
    base = landmarks[0]

    normalized = landmarks - base

    # Scale by max distance (scale invariance)
    max_dist = np.max(np.linalg.norm(normalized, axis=1))

    if max_dist > 0:
        normalized = normalized / max_dist

    return normalized.flatten()
