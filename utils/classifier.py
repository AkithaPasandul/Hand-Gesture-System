import joblib

class GestureClassifier:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def predict(self, features):
        prediction = self.model.predict([features])
        return prediction[0]
