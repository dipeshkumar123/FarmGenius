from fastapi import APIRouter, Depends
from pydantic import BaseModel
from app.services.crop_service import crop_service
from app.core.security import get_current_user

router = APIRouter()

class CropRecommendationRequest(BaseModel):
    n: float
    p: float
    k: float
    ph: float
    ec: float
    s: float
    cu: float
    fe: float
    mn: float
    zn: float
    b: float

class CropRecommendationResponse(BaseModel):
    recommended_crop: str

@router.post("/recommend", response_model=CropRecommendationResponse)
async def recommend_crop(request: CropRecommendationRequest, user_id: str = Depends(get_current_user)):
    crop = crop_service.predict_crop(
        n=request.n, p=request.p, k=request.k, ph=request.ph, 
        ec=request.ec, s=request.s, cu=request.cu, fe=request.fe, 
        mn=request.mn, zn=request.zn, b=request.b
    )
    return CropRecommendationResponse(recommended_crop=crop)
