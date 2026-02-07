import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_dataset(data_dir):
    X, y = [], []

    for file in os.listdir(data_dir):
        if file.endswith(".npy"):
            gesture = file.replace(".npy", "")
            data = np.load(os.path.join(data_dir, file))

            X.append(data)
            y.extend([gesture] * len(data))

    X = np.vstack(X)
    y = np.array(y)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    return X, y_encoded, encoder
