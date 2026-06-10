# FarmGenius ML Implementation Spec

This document details the Machine Learning implementation for FarmGenius, tailored to the constraints of smallholder farmers and low-connectivity environments. The ML pipeline provides robust NLP capabilities for local dialect queries and highly efficient on-device disease detection.

## 1. File Paths and Code for Each Task

### Task 1: Chatbot Service & Task 4: Groq Integration
We have replaced the broken `googletrans` dependency and legacy `chatbot.py` entirely. The system now uses a hybrid approach: local ML model first, falling back to Groq Llama-3.1-70B for translation and advanced response generation.

**File: `src/services/llm_service.py`**
Handles Groq API interactions and translation natively using Llama-3.1, along with graceful fallback messages per language.
*(See codebase for full implementation: `d:\Projects\FarmGenius\src\services\llm_service.py`)*

**File: `src/services/chatbot_service.py`**
Orchestrates the fallback. If the local MultinomialNB model confidence is ≥ 0.72, it provides a fast local response. Otherwise, it calls the `LLMService`.
*(See codebase for full implementation: `d:\Projects\FarmGenius\src\services\chatbot_service.py`)*

### Task 2: Retrain Chatbot on Real Farmer Queries
The previous chatbot was trained on generic curated FAQs. It is now retrained directly from the `FARMER_CORPUS.md`, capturing exact dialects ("jhulsa", "sundi") and Hinglish variations. Stop words are intentionally kept.

**File: `src/models/train_chatbot_farmer.py`**
Parses the markdown corpus, generates Hinglish augments, trains an n-gram `MultinomialNB`, and evaluates with `classification_report`.
*(See codebase for full implementation: `d:\Projects\FarmGenius\src\models\train_chatbot_farmer.py`)*

### Task 3: TFLite Disease Detection Model
The plant disease model has been optimized for Indian crops (Tomato, Potato, Rice, Wheat, Cotton, Chickpea, Maize) and heavily quantized for mobile delivery.

**File: `model/train_disease_model.ipynb`**
A Google Colab notebook providing end-to-end steps: Kaggle dataset downloading, filtering, transfer learning with MobileNetV2, fine-tuning, and INT8 quantization for sub-15MB deployment.
*(See codebase for full implementation: `d:\Projects\FarmGenius\model\train_disease_model.ipynb`)*

**File: `model/labels_indian_crops.txt`**
Includes the specific filtered class names, a "Healthy" class per crop, and an "Unknown — please retake photo" catch-all.
*(See codebase for full implementation: `d:\Projects\FarmGenius\model\labels_indian_crops.txt`)*

---

## 2. Requirements Additions

The backend and ML systems rely on distinct dependencies. Add these to their respective requirements files:

**`requirements-api.txt`** (for FastAPI deployment)
```text
groq==0.11.0          # For LLM fallback and translation
scikit-learn==1.3.2   # For running the local chatbot inference
pandas==2.1.3
requests==2.31.0
```

**`requirements-ml.txt`** (for local/Colab training)
```text
tensorflow==2.15.0    # For building/converting TFLite models
scikit-learn==1.3.2
pandas==2.1.3
kaggle==1.6.14        # For downloading PlantVillage
Pillow==10.1.0        # For image augmentations
```

---

## 3. How to Run Training

### Chatbot Training (Locally)
1. Ensure you have the `requirements-ml.txt` dependencies installed locally.
2. From the project root, run the chatbot training script:
   ```bash
   python src/models/train_chatbot_farmer.py
   ```
3. This script parses `FARMER_CORPUS.md` and generates `models/chatbot_farmer_v1.pkl`.

### Disease Model Training (Google Colab)
1. Upload `model/train_disease_model.ipynb` to [Google Colab](https://colab.research.google.com/).
2. You must have a Kaggle account to fetch the PlantVillage dataset. Place your `kaggle.json` inside your Colab environment or Google Drive.
3. Select Runtime > Change runtime type > Hardware accelerator: **T4 GPU** (Free tier).
4. Run all cells. At the end of the notebook, `disease_model_quant.tflite` (<15MB) and `labels_indian_crops.txt` will be available for download.

---

## 4. Model Accuracy Targets

For the models to be deemed production-ready, they must hit the following minimums on validation sets:

**Chatbot NLP Model:**
- Overall F1 Score: **> 0.85**
- Confidence Threshold limit for local inference: **0.72** (Queries below this fallback to Groq).

**Disease Detection TFLite Model:**
- Overall accuracy: **> 90%**
- Minimum acceptable accuracy *per class*: **80%**
- **Crucial Metric:** False positive rate for the "Healthy" class must be under 5% (we do not want to suggest applying chemicals to healthy crops).
- Inference speed on 2G Android devices (CPU): **< 800ms**.

---

## 5. Flutter App Model Update Steps

The `.tflite` model size is kept under 15MB to allow direct bundling.

### v1 Asset Bundling (Current)
1. Move the downloaded `disease_model_quant.tflite` and `labels_indian_crops.txt` to the Flutter `assets/models/` directory.
2. Ensure `pubspec.yaml` lists these assets:
   ```yaml
   flutter:
     assets:
       - assets/models/disease_model_quant.tflite
       - assets/models/labels_indian_crops.txt
   ```
3. Run `flutter clean` and `flutter pub get` to ensure assets are indexed.
4. Build the APK. The `tflite_flutter` package will load the model directly from the bundle.

### v2 OTA Updates (Future Strategy)
To avoid forcing 2G farmers to download a new 20-30MB APK for every model iteration:
1. The Flutter app will poll the Supabase DB at startup (if online) for the latest model hash.
2. If the hash differs from the local cache, the app downloads the new `.tflite` to the device's `ApplicationDocumentsDirectory`.
3. The inference service dynamically points the interpreter to the downloaded path instead of the static asset bundle.
