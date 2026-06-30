from fastapi import APIRouter, Depends
from app.models.schemas import ChatRequest, ChatResponse
from app.services.chatbot_service import chatbot_service
from app.core.security import get_current_user_optional

router = APIRouter()

@router.post("/", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest, user_id: str = Depends(get_current_user_optional)):
    # Fix BOLA vulnerability: Use authenticated user_id if present, else fallback to req.farmer_id
    farmer_id = user_id if user_id else req.farmer_id
    
    # Supabase strictly requires farmer_id to be a valid UUID.
    # If the user logged in using the mock fallback, farmer_id might be a phone number.
    # Convert it to a deterministic UUID to prevent Supabase 22P02 errors.
    import uuid
    try:
        uuid.UUID(farmer_id)
    except ValueError:
        farmer_id = str(uuid.uuid5(uuid.NAMESPACE_OID, farmer_id))
        
    # Ensure this farmer exists in the database to satisfy the queries table foreign key constraint
    try:
        from app.core.security import supabase
        supabase.table("farmers").upsert({"id": farmer_id, "phone": user_id}).execute()
    except Exception as e:
        print(f"Error upserting mock farmer: {e}")
    # The LLM (Llama 3.3 70B) handles Hindi, Kannada, Telugu, Tamil, Marathi, English natively.
    # This removes the LibreTranslate round-trip that was failing silently on Vercel.
    res = await chatbot_service.get_response(req.query, req.language, farmer_id)

    return ChatResponse(**res)
