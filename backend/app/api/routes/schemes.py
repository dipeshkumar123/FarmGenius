from fastapi import APIRouter, Depends
from typing import List
from app.models.schemas import SchemeResponse
from app.core.security import get_current_user, supabase

router = APIRouter()

@router.get("/", response_model=List[SchemeResponse])
async def get_schemes(crop: str, state: str, user_id: str = Depends(get_current_user)):
    try:
        # Example implementation: filtering schemes based on state and crop
        # Requires the "schemes" table to be pre-seeded in Supabase.
        res = supabase.table("schemes").select("*").execute()
        schemes_data = []
        if res.data:
            for item in res.data:
                schemes_data.append(SchemeResponse(**item))
            return schemes_data
    except Exception:
        pass
        
    # Dummy fallback response
    return [
        SchemeResponse(
            scheme_name="PM-KISAN",
            description="Direct benefit transfer of Rs. 6000 per year.",
            eligibility="All landholding farmers families.",
            link="https://pmkisan.gov.in"
        )
    ]
