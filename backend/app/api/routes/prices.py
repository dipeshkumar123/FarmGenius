from fastapi import APIRouter, Depends
from app.models.schemas import PriceResponse
from app.services.price_service import price_service
from app.core.security import get_current_user_optional

router = APIRouter()

@router.get("/", response_model=PriceResponse)
async def get_prices(commodity: str, district: str, state: str, user_id: str = Depends(get_current_user_optional)):
    data = await price_service.get_prices(commodity, district, state)
    return PriceResponse(**data)
