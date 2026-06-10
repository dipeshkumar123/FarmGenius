from fastapi import Request, HTTPException
from supabase import create_client, Client
from app.core.config import settings
import time

# Initialize Supabase Admin Client
supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)

# Simple in-memory cache to prevent hammering Supabase Auth endpoint
_token_cache = {}

def validate_token(token: str) -> str:
    now = time.time()
    
    # Return from cache if validated within the last 5 minutes
    if token in _token_cache:
        cached = _token_cache[token]
        if now - cached['time'] < 300:
            return cached['user_id']
    
    try:
        res = supabase.auth.get_user(token)
        if not res or not res.user:
            raise Exception("Invalid token structure")
            
        user_id = res.user.id
        _token_cache[token] = {'user_id': user_id, 'time': now}
        return user_id
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

async def get_current_user(request: Request) -> str:
    # Extract JWT token from Authorization header
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
        
    token = auth_header.split(" ")[1]
    return validate_token(token)
