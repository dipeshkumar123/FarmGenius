import json, pickle, sys

print("=== Verifying All Retrained Models ===\n")

# 1. Disease CNN Model
print("1. Disease CNN Model (disease_model.h5)")
try:
    from keras.models import load_model
    model = load_model("models/disease_model.h5")
    print(f"   OK - Loaded successfully, {model.count_params():,} parameters")
    print(f"   Input shape: {model.input_shape}, Output shape: {model.output_shape}")
except Exception as e:
    print(f"   FAILED: {e}")

# 2. Disease Class Map
print("\n2. Disease Class Map (disease_class_map.json)")
try:
    with open("models/disease_class_map.json") as f:
        class_map = json.load(f)
    classes = class_map["classes"]
    print(f"   OK - {len(classes)} classes")
    for c in classes:
        print(f"      - {c}")
except Exception as e:
    print(f"   FAILED: {e}")

# 3. Crop Model
print("\n3. Crop Recommendation Model (crop_recommendation_model.pkl)")
try:
    with open("models/crop_recommendation_model.pkl", "rb") as f:
        crop_data = pickle.load(f)
    print(f"   OK - Type: {type(crop_data).__name__}")
    if hasattr(crop_data, "predict"):
        print("   Has predict method: True")
except Exception as e:
    print(f"   FAILED: {e}")

# 4. FAQ Embeddings
print("\n4. FAQ Embeddings (faq_embeddings.pkl)")
try:
    with open("models/faq_embeddings.pkl", "rb") as f:
        faq_data = pickle.load(f)
    print(f"   OK - Type: {type(faq_data).__name__}")
    if isinstance(faq_data, dict):
        print(f"   Keys: {list(faq_data.keys())[:5]}")
    elif isinstance(faq_data, tuple):
        print(f"   Length: {len(faq_data)}")
except Exception as e:
    print(f"   FAILED: {e}")

# 5. Test disease model prediction
print("\n5. Test Disease Model Prediction")
try:
    import numpy as np
    from keras.applications.mobilenet_v2 import preprocess_input
    
    model = load_model("models/disease_model.h5")
    with open("models/disease_class_map.json") as f:
        class_map = json.load(f)
    
    # Create a dummy image
    dummy = np.random.randint(0, 255, (1, 224, 224, 3), dtype=np.uint8).astype(np.float32)
    dummy = preprocess_input(dummy)
    pred = model.predict(dummy, verbose=0)
    top_idx = np.argmax(pred[0])
    top_class = class_map["classes"][top_idx]
    top_conf = pred[0][top_idx]
    print(f"   OK - Prediction works (dummy: {top_class}, conf: {top_conf:.4f})")
except Exception as e:
    print(f"   FAILED: {e}")

# 6. Test crop model prediction
print("\n6. Test Crop Model Prediction")
try:
    import numpy as np
    with open("models/crop_recommendation_model.pkl", "rb") as f:
        crop_model = pickle.load(f)
    
    # Dummy soil features: N, P, K, ph, EC, S, Cu, Fe, Mn, Zn, B
    features = np.array([[80, 40, 40, 6.5, 0.5, 10, 2, 50, 20, 1, 0.5]])
    pred = crop_model.predict(features)
    proba = crop_model.predict_proba(features)
    print(f"   OK - Predicted: {pred[0]}, top prob: {max(proba[0]):.4f}")
except Exception as e:
    print(f"   FAILED: {e}")

print("\n=== All Model Verification Complete ===")
