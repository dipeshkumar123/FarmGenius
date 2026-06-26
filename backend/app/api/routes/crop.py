from fastapi import APIRouter, Depends
from pydantic import BaseModel
from app.services.crop_service import crop_service
from app.core.security import get_current_user

from typing import List, Optional

router = APIRouter()

class CropRecommendationRequest(BaseModel):
    location: str
    soil_type: str
    water_availability: str
    farm_size: str
    season: str
    n: Optional[str] = None
    p: Optional[str] = None
    k: Optional[str] = None

class CropRecommendationItem(BaseModel):
    rank: int
    name: str
    emoji: str
    suitability: float
    expectedYield: str
    marketPrice: str
    profitEstimate: str
    season: str
    water: str
    duration: str

class CropRecommendationResponse(BaseModel):
    recommendations: List[CropRecommendationItem]

@router.post("/recommend", response_model=CropRecommendationResponse)
async def recommend_crop(request: CropRecommendationRequest, user_id: str = Depends(get_current_user)):
    recommendations = await crop_service.predict_crop(
        location=request.location,
        soil_type=request.soil_type,
        water_availability=request.water_availability,
        farm_size=request.farm_size,
        season=request.season,
        n=request.n,
        p=request.p,
        k=request.k
    )
    return CropRecommendationResponse(recommendations=recommendations)
