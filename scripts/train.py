import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATA_PATH = "data"
MODEL_PATH = "models/gesture_classifier.pkl"

X = []
y = []

for file in os.listdir(DATA_PATH):
    if file.endswith(".npy"):
        label = file.replace(".npy", "")
        data = np.load(os.path.join(DATA_PATH, file))

        for sample in data:
            X.append(sample)
            y.append(label)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

clf = RandomForestClassifier(n_estimators=200)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

joblib.dump(clf, MODEL_PATH)
print("Model saved to", MODEL_PATH)
