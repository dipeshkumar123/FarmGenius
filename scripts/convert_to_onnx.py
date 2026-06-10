#!/usr/bin/env python
"""
Convert ML models to ONNX format for lightweight deployment.

Run locally after training:
    python scripts/convert_to_onnx.py

This converts:
  - disease_model.h5 (Keras/TF) → disease_model.onnx
  - crop_recommendation_model.pkl remains as pickle (scikit-learn loads natively)
  - faq_embeddings.pkl remains as pickle
"""

import os
import sys
import json
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def convert_disease_model():
    """Convert the Keras disease detection model (.h5) to ONNX format."""
    import subprocess
    import tempfile

    h5_path = os.path.join(MODELS_DIR, "disease_model.h5")
    onnx_path = os.path.join(MODELS_DIR, "disease_model.onnx")

    if not os.path.exists(h5_path):
        print(f"[SKIP] {h5_path} not found")
        return

    # Step 1: Load model and save as TF SavedModel
    print(f"[INFO] Loading Keras model from {h5_path}...")
    import tensorflow as tf
    model = tf.keras.models.load_model(h5_path)

    input_shape = model.input_shape  # e.g. (None, 224, 224, 3)
    print(f"[INFO] Model input shape: {input_shape}")

    saved_model_dir = os.path.join(tempfile.mkdtemp(), "saved_model")
    print(f"[INFO] Exporting as TF SavedModel to {saved_model_dir}...")
    model.export(saved_model_dir)

    # Step 2: Use tf2onnx CLI to convert SavedModel → ONNX
    # Monkey-patch np.cast (removed in NumPy 2.0) for tf2onnx compatibility
    print("[INFO] Converting SavedModel to ONNX via tf2onnx CLI...")
    patch_code = (
        "import numpy as np; "
        "np.cast = {t: lambda x, t=t: np.asarray(x, dtype=t) for t in "
        "[np.float16, np.float32, np.float64, np.int8, np.int16, np.int32, np.int64, np.uint8, np.bool_]}; "
    )
    cmd = [
        sys.executable, "-c",
        patch_code + "from tf2onnx.convert import main; main()",
        "--saved-model", saved_model_dir,
        "--output", onnx_path,
        "--opset", "13",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] tf2onnx failed:\n{result.stderr[-1000:]}")
        return
    print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)

    # Step 3: Verify the converted model
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    dummy = np.random.rand(1, *input_shape[1:]).astype(np.float32)
    result_val = sess.run(None, {input_name: dummy})
    print(f"[INFO] ONNX inference test passed. Output shape: {result_val[0].shape}")

    # File sizes
    h5_size = os.path.getsize(h5_path) / (1024 * 1024)
    onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"[INFO] Size: {h5_size:.1f} MB → {onnx_size:.1f} MB")
    print(f"[OK] Disease model converted to {onnx_path}")

    # Cleanup
    import shutil
    shutil.rmtree(os.path.dirname(saved_model_dir), ignore_errors=True)


def verify_pickle_models():
    """Verify that pickle models load correctly (no conversion needed)."""
    import pickle

    crop_path = os.path.join(MODELS_DIR, "crop_recommendation_model.pkl")
    faq_path = os.path.join(MODELS_DIR, "faq_embeddings.pkl")

    for path in [crop_path, faq_path]:
        if os.path.exists(path):
            with open(path, "rb") as f:
                obj = pickle.load(f)
            size = os.path.getsize(path) / (1024 * 1024)
            print(f"[OK] {os.path.basename(path)}: {size:.2f} MB — loaded successfully")
        else:
            print(f"[SKIP] {path} not found")


def main():
    print("=" * 60)
    print("FarmGenius — Model Conversion to ONNX")
    print("=" * 60)

    print("\n--- Disease Detection Model (Keras → ONNX) ---")
    convert_disease_model()

    print("\n--- Pickle Models (verify only) ---")
    verify_pickle_models()

    print("\n" + "=" * 60)
    print("Done! Models are ready for deployment.")
    print("Commit the models/ directory and push to deploy.")
    print("=" * 60)


if __name__ == "__main__":
    main()
