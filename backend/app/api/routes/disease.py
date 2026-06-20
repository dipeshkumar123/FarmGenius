from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from app.models.schemas import DiseaseResponse
from app.core.security import get_current_user
import base64
import groq
from app.core.config import settings
import json
import os
import io

router = APIRouter()

# Try loading the user's trained .h5 model
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "models", "disease_model.h5")
LABELS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "model", "labels_indian_crops.txt")

keras_model = None
class_names = []

try:
    import tensorflow as tf
    import numpy as np
    from PIL import Image
    if os.path.exists(MODEL_PATH):
        keras_model = tf.keras.models.load_model(MODEL_PATH)
        print("Successfully loaded trained user model: disease_model.h5")
        
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, "r") as f:
            class_names = [line.strip() for line in f.readlines()]
except Exception as e:
    print(f"Failed to load user model: {e}")

@router.post("/detect", response_model=DiseaseResponse)
async def detect_disease(file: UploadFile = File(...), user_id: str = Depends(get_current_user)):
    if not file.filename.lower().endswith((".jpg", ".png", ".jpeg")):
        raise HTTPException(status_code=400, detail="Only JPG and PNG images are allowed.")
    
    content = await file.read()
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size exceeds the 5MB limit.")
        
    # --- 1. First try the local trained ML Model ---
    if keras_model is not None and class_names:
        try:
            image = Image.open(io.BytesIO(content)).convert("RGB")
            # Most MobileNet/ResNet expect 224x224
            image = image.resize((224, 224))
            img_array = np.array(image) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            predictions = keras_model.predict(img_array)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            disease_name = class_names[predicted_class_idx]
            
            # Fetch treatment advice from Groq using the predicted disease name
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
            
            return DiseaseResponse(
                disease_name=disease_name,
                confidence=confidence,
                disease_name_hi=result.get("disease_name_hi", ""),
                organic_treatment=result.get("organic_treatment", "Consult KVK."),
                chemical_treatment=result.get("chemical_treatment", "Consult KVK."),
                dosage=result.get("dosage", ""),
                source_url="https://kvk.icar.gov.in/",
                source_name="Trained AI Model (.h5)"
            )
        except Exception as e:
            print(f"Error executing local model inference: {e}")

    # --- 2. Fallback to Groq Vision API ---
    try:
        base64_image = base64.b64encode(content).decode('utf-8')
        client = groq.Groq(api_key=settings.GROQ_API_KEY)
        
        prompt = """Analyze this plant leaf image. Identify the crop and the disease (if any).
Respond ONLY with a valid JSON object matching this schema:
{
    "disease_name": "Name of disease in English",
    "disease_name_hi": "Name of disease in Hindi",
    "confidence": 0.0 to 1.0,
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
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        
        response_text = completion.choices[0].message.content
        result = json.loads(response_text)
        
        return DiseaseResponse(
            disease_name=result.get("disease_name", "Unknown"),
            confidence=float(result.get("confidence", 0.8)),
            disease_name_hi=result.get("disease_name_hi", ""),
            organic_treatment=result.get("organic_treatment", "Consult local KVK."),
            chemical_treatment=result.get("chemical_treatment", "Consult local KVK."),
            dosage=result.get("dosage", ""),
            source_url="https://kvk.icar.gov.in/",
            source_name="AI Vision (Groq Llama 3.2 90B)"
        )
    except Exception as e:
        print(f"Vision API Error: {e}")
        raise HTTPException(
            status_code=503,
            detail="Disease detection temporarily unavailable. Please try again in a moment or take a clearer photo of the affected leaf."
        )

