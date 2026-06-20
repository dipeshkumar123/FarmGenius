import os
import joblib
import numpy as np

class CropService:
    def __init__(self):
        self.model = None
        self.scaler = None
        self._load_model()

    def _load_model(self):
        model_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "crop_recommendation_model.pkl")
        if os.path.exists(model_path):
            try:
                # Load the dictionary containing model and scaler
                loaded = joblib.load(model_path)
                self.model = loaded.get('model')
                self.scaler = loaded.get('scaler')
            except Exception as e:
                print(f"Failed to load crop recommendation model: {e}")

    def predict_crop(self, n: float, p: float, k: float, ph: float, ec: float, s: float, cu: float, fe: float, mn: float, zn: float, b: float) -> str:
        if self.model is None or self.scaler is None:
            return "Model unavailable. Please contact an agronomist."
            
        try:
            # The model expects exactly 11 features in this order:
            # N, P, K, ph, EC, S, Cu, Fe, Mn, Zn, B
            features = np.array([[n, p, k, ph, ec, s, cu, fe, mn, zn, b]])
            
            # Scale the features
            scaled_features = self.scaler.transform(features)
            
            # Predict
            prediction = self.model.predict(scaled_features)
            
            # Prediction is typically an array of strings, e.g. ['rice']
            if len(prediction) > 0:
                crop = str(prediction[0]).capitalize()
                return crop
            return "Unknown"
        except Exception as e:
            return f"Error during prediction: {str(e)}"

crop_service = CropService()
