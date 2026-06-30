from fastapi import APIRouter, Depends
from typing import List
from app.models.schemas import SchemeResponse
from app.core.security import get_current_user_optional
from app.services.schemes_service import schemes_service

router = APIRouter()

@router.get("/", response_model=List[SchemeResponse])
async def get_schemes(crop: str, state: str, user_id: str = Depends(get_current_user_optional)):
    data = schemes_service.get_realtime_schemes(crop, state)
    return [SchemeResponse(**item) for item in data]
