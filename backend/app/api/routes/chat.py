from fastapi import APIRouter, Depends
from app.models.schemas import ChatRequest, ChatResponse
from app.services.chatbot_service import chatbot_service
from app.core.security import get_current_user

router = APIRouter()

@router.post("/", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest, user_id: str = Depends(get_current_user)):
    # Fix BOLA vulnerability: Always use the authenticated user_id, NOT the one from the request body
    farmer_id = user_id

    # Pass the original language directly to the LLM system prompt.
    # The LLM (Llama 3.3 70B) handles Hindi, Kannada, Telugu, Tamil, Marathi, English natively.
    # This removes the LibreTranslate round-trip that was failing silently on Vercel.
    res = await chatbot_service.get_response(req.query, req.language, farmer_id)

    return ChatResponse(**res)
