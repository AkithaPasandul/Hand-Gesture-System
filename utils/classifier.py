import joblib

class GestureClassifier:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def predict(self, features):
        probs = self.model.predict_proba([features])[0]
        max_prob = max(probs)
        class_id = self.model.classes_[probs.argmax()]
        
        return class_id, max_prob
