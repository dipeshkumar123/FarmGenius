from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from app.models.schemas import DiseaseResponse
from app.core.security import get_current_user

router = APIRouter()

@router.post("/detect", response_model=DiseaseResponse)
async def detect_disease(file: UploadFile = File(...), user_id: str = Depends(get_current_user)):
    if not file.filename.lower().endswith((".jpg", ".png", ".jpeg")):
        raise HTTPException(status_code=400, detail="Only JPG and PNG images are allowed.")
    
    content = await file.read()
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size exceeds the 5MB limit.")
        
    # Note: Primary disease detection is on-device via TFLite.
    # This endpoint acts as an online fallback/analytics hook.
    # In a full deployment, this would invoke a server-side TF model.
    # We return simulated data here.
    return DiseaseResponse(
        disease_name="Tomato___Early_blight",
        confidence=0.92,
        disease_name_hi="टमाटर अगेती झुलसा",
        organic_treatment="Spray 5% Neem seed kernel extract.",
        chemical_treatment="Apply Mancozeb or Copper fungicides.",
        dosage="2g per liter of water",
        source_url="https://kvk.icar.gov.in/",
        source_name="ICAR-KVK"
    )
