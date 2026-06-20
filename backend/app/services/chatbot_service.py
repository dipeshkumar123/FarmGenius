import os
from app.services.llm_service import llm_service
from app.core.security import supabase

class ChatbotService:
    def __init__(self):
        pass

    async def get_response(self, query: str, language: str, farmer_id: str) -> dict:
        confidence = 0.0
        response_text = ""
        source = ""
        category = "General"

        try:
            llm_res = await llm_service.get_response(query, language)
            response_text = llm_res["response"]
            confidence = llm_res["confidence"]
            source = llm_res["source"]
        except Exception as e:
            response_text = "Sorry, service is unavailable. Please contact your local KVK."
            source = "Error"
            confidence = 0.0

        # Background log to Supabase 'queries' table
        try:
            supabase.table("queries").insert({
                "farmer_id": farmer_id,
                "query_text": query,
                "language": language,
                "response": response_text,
                "category": category,
            }).execute()
        except Exception as e:
            # Silently catch insert errors to avoid disrupting user experience
            pass

        return {
            "response": response_text,
            "source": source,
            "confidence": confidence
        }

chatbot_service = ChatbotService()
