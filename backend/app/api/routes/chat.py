from fastapi import APIRouter, Depends
from app.models.schemas import ChatRequest, ChatResponse
from app.services.chatbot_service import chatbot_service
from app.core.security import get_current_user

router = APIRouter()

@router.post("/", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest, user_id: str = Depends(get_current_user)):
    # Proceed using req.farmer_id or the authenticated user_id
    res = chatbot_service.get_response(req.query, req.language, req.farmer_id)
    return ChatResponse(**res)
