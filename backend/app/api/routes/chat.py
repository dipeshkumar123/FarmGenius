from fastapi import APIRouter, Depends
from app.models.schemas import ChatRequest, ChatResponse
from app.services.chatbot_service import chatbot_service
from app.core.security import get_current_user
from app.services.translation_service import translation_service

router = APIRouter()

@router.post("/", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest, user_id: str = Depends(get_current_user)):
    # Fix BOLA vulnerability: Always use the authenticated user_id, NOT the one from the request body
    farmer_id = user_id

    # Translate query to English if not English
    english_query = translation_service.translate(req.query, req.language, "en")
    
    # Get response using English query but ask LLM to format it carefully
    # Wait, llm_service uses tool calling and is async now!
    res = await chatbot_service.get_response(english_query, "en", farmer_id)
    
    # Translate response back to original language if necessary
    if req.language != "en":
        translated_res = translation_service.translate(res["response"], "en", req.language)
        res["response"] = translated_res
    
    return ChatResponse(**res)
