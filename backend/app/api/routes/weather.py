from fastapi import APIRouter, Depends
from typing import List
from app.models.schemas import WeatherForecast
from app.services.weather_service import weather_service
from app.core.security import get_current_user

router = APIRouter()

@router.get("/", response_model=List[WeatherForecast])
async def get_weather(district: str, state: str, user_id: str = Depends(get_current_user)):
    data = await weather_service.get_weather(district, state)
    return [WeatherForecast(**item) for item in data]
