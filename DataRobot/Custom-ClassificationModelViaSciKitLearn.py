import joblib
import pandas as pd

class CustomModel:
    def __init__(self):
        self.model = joblib.load('ModelClassificationModelViaSciKitLearn.joblib')

    def predict(self, data):
        # Assumes data is a pandas DataFrame
        return self.model.predict(data)