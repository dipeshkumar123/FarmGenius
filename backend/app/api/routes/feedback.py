from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Optional
import httpx
import os
from app.core.security import get_current_user, supabase

router = APIRouter()

class FeedbackRequest(BaseModel):
    query_id: str
    was_helpful: bool
    follow_up_text: Optional[str] = None
    language: str

@router.post("/")
async def submit_feedback(req: FeedbackRequest, user_id: str = Depends(get_current_user)):
    try:
        supabase.table("feedback").insert({
            "query_id": req.query_id,
            "was_helpful": req.was_helpful,
            "follow_up_action": req.follow_up_text,
        }).execute()
        
        # Trigger Slack webhook for negative feedback
        if not req.was_helpful:
            webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
            if webhook_url:
                async with httpx.AsyncClient() as client:
                    await client.post(webhook_url, json={
                        "text": f"Negative feedback from query {req.query_id} (Lang: {req.language}). Follow up: {req.follow_up_text}"
                    })
    except Exception:
        pass
    return {"status": "received"}
