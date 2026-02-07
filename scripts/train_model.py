import joblib
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

from utils.dataset import load_dataset

# LOAD DATASET
X, y, encoder = load_dataset("data/raw")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Model
model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    max_iter=500,
    random_state=42
)

model.fit(X_train, y_train)

# EVALUATION
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

# SAVE MODEL
joblib.dump(model, "models/gesture_classifier.pkl")
joblib.dump(encoder, "models/label_encoder.pkl")

print("Model saved.")
