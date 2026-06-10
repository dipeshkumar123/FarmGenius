"""Retrain the crop recommendation model with current sklearn."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.crop_model import CropRecommendationModel
import logging
logging.basicConfig(level=logging.INFO)

# Delete old incompatible model first
old_model = os.path.join("models", "crop_recommendation_model.pkl")
if os.path.exists(old_model):
    os.remove(old_model)
    print(f"Deleted old model: {old_model}")

model = CropRecommendationModel()
result = model.train()
print(f"Training accuracy: {result['accuracy']:.4f}")
print(f"Best params: {result['best_params']}")

# Test prediction
test = {'N': 90, 'P': 40, 'K': 35, 'ph': 6.5, 'EC': 0.2, 'S': 15.0, 'Zn': 0.8, 'Fe': 20, 'Cu': 1.0, 'Mn': 2.0, 'B': 0.5}
pred = model.predict(test)
print("\nTop recommendations:")
for r in pred['top_recommendations'][:3]:
    print(f"  {r['crop']}: {r['confidence']:.2f}")
