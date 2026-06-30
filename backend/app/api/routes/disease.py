from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
import base64
import groq
from app.core.config import settings
import json
import os
import io

router = APIRouter()

# Try loading the user's trained .onnx model
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ml_models", "disease_model.onnx")
LABELS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ml_models", "disease_class_map.json")

ort_session = None
class_names = {}

try:
    import onnxruntime as ort
    import numpy as np
    from PIL import Image
    if os.path.exists(MODEL_PATH):
        ort_session = ort.InferenceSession(MODEL_PATH)
        print("Successfully loaded trained user model: disease_model.onnx")
        
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, "r") as f:
            raw_labels = json.load(f)
            if "class_indices" in raw_labels:
                class_names = {str(v): k for k, v in raw_labels["class_indices"].items()}
            elif "classes" in raw_labels:
                class_names = {str(i): name for i, name in enumerate(raw_labels["classes"])}
            elif isinstance(raw_labels, dict):
                # Reverse if needed so key is string index
                first_val = list(raw_labels.values())[0]
                if isinstance(first_val, int):
                    class_names = {str(v): k for k, v in raw_labels.items()}
                else:
                    class_names = raw_labels
            elif isinstance(raw_labels, list):
                class_names = {str(i): name for i, name in enumerate(raw_labels)}
except Exception as e:
    print(f"Failed to load ONNX model: {e}")

@router.post("/detect")
async def detect_disease(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".jpg", ".png", ".jpeg")):
        raise HTTPException(status_code=400, detail="Only JPG and PNG images are allowed.")
    
    content = await file.read()
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size exceeds the 5MB limit.")
        
    # --- 1. First try the local trained ML Model ---
    if ort_session is not None and class_names:
        try:
            image = Image.open(io.BytesIO(content)).convert("RGB")
            image = image.resize((224, 224))
            # MobileNet preprocessing
            img_array = np.array(image, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            input_name = ort_session.get_inputs()[0].name
            output_name = ort_session.get_outputs()[0].name
            
            predictions = ort_session.run([output_name], {input_name: img_array})[0]
            predicted_class_idx = int(np.argmax(predictions[0]))
            confidence = float(predictions[0][predicted_class_idx])
            disease_name = class_names.get(str(predicted_class_idx), "Unknown Disease")
            
            # Fetch treatment advice from Groq using the predicted disease name (fast llama 3.1 8b)
            client = groq.Groq(api_key=settings.GROQ_API_KEY)
            prompt = f"""The user has identified the plant disease as '{disease_name}' with {confidence*100:.1f}% confidence.
            Respond ONLY with a valid JSON object matching this schema:
            {{
                "disease_name_hi": "Name of disease in Hindi",
                "organic_treatment": "Best organic treatment method",
                "chemical_treatment": "Best chemical treatment method with dosage",
                "dosage": "Specific dosage instructions"
            }}"""
            completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            result = json.loads(completion.choices[0].message.content)
            
            return {
                "disease_name": disease_name.replace("___", " ").replace("_", " "),
                "confidence": confidence,
                "disease_name_hi": result.get("disease_name_hi", ""),
                "organic_treatment": result.get("organic_treatment", "Consult local KVK."),
                "chemical_treatment": result.get("chemical_treatment", "Consult local KVK."),
                "dosage": result.get("dosage", ""),
                "source_url": "https://kvk.icar.gov.in/",
                "source_name": "FarmGenius AI (ONNX)"
            }
        except Exception as e:
            print(f"Error executing local model inference: {e}")

    # --- 2. Fallback to Groq Vision API ---
    try:
        base64_image = base64.b64encode(content).decode('utf-8')
        client = groq.Groq(api_key=settings.GROQ_API_KEY)
        
        prompt = """You are an expert agricultural AI. Analyze this plant leaf image and identify the disease.
        Respond ONLY with a valid JSON object matching this schema:
        {
            "disease_name": "Name of disease in English (or Healthy)",
            "confidence": 0.95,
            "disease_name_hi": "Name of disease in Hindi",
            "organic_treatment": "Best organic treatment method",
            "chemical_treatment": "Best chemical treatment method with dosage",
            "dosage": "Specific dosage instructions"
        }"""
        
        completion = client.chat.completions.create(
            model="llama-3.2-90b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(completion.choices[0].message.content)
        return {
            "disease_name": result.get("disease_name", "Unknown"),
            "confidence": result.get("confidence", 0.8),
            "disease_name_hi": result.get("disease_name_hi", ""),
            "organic_treatment": result.get("organic_treatment", "Consult local KVK."),
            "chemical_treatment": result.get("chemical_treatment", "Consult local KVK."),
            "dosage": result.get("dosage", ""),
            "source_url": "https://kvk.icar.gov.in/",
            "source_name": "FarmGenius AI (Groq Vision)"
        }
    except Exception as e:
        print(f"Error executing Groq Vision fallback: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Disease detection is temporarily unavailable. Error: {str(e)}"
        )


