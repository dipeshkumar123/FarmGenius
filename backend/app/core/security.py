from fastapi import Request, HTTPException
import jwt
from supabase import create_client, Client
from app.core.config import settings
import time

# Initialize Supabase Admin Client
supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)

# Simple in-memory token validation cache (5 min TTL)
_token_cache: dict = {}
_CACHE_TTL = 300  # 5 minutes


def validate_token(token: str) -> str:
    """
    Validate a JWT token issued by this app's auth flow.
    Tokens are either:
      - A real Supabase JWT (signed with SUPABASE_JWT_SECRET)
      - A mock JWT (signed with SUPABASE_SERVICE_KEY, used when Supabase OTP fails)
    Strategy:
      1. Check in-memory cache (avoids repeated calls, 5 min TTL)
      2. Try decoding with SUPABASE_SERVICE_KEY (mock token path, always works)
      3. Try decoding with SUPABASE_JWT_SECRET if configured (real Supabase token path)
      4. Fallback: verify via Supabase Admin API (most reliable but slowest)
    """
    # 1. Check cache first
    now = time.time()
    cached = _token_cache.get(token)
    if cached and now - cached['timestamp'] < _CACHE_TTL:
        return cached['user_id']

    user_id = None

    # 2. Try decoding with SUPABASE_SERVICE_KEY (handles mock tokens from auth.py fallback)
    try:
        payload = jwt.decode(
            token,
            settings.SUPABASE_SERVICE_KEY,
            algorithms=["HS256"],
            options={"verify_aud": False},
        )
        user_id = payload.get("sub")
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        pass

    # 3. If that failed, try SUPABASE_JWT_SECRET (real Supabase tokens)
    if not user_id and settings.SUPABASE_JWT_SECRET:
        try:
            payload = jwt.decode(
                token,
                settings.SUPABASE_JWT_SECRET,
                algorithms=["HS256"],
                options={"verify_aud": False},
            )
            user_id = payload.get("sub")
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired. Please log in again.")
        except jwt.InvalidTokenError:
            pass

    # 4. Last resort: Supabase Admin API validation (for genuine Supabase session tokens)
    if not user_id:
        try:
            response = supabase.auth.get_user(token)
            if response and response.user:
                user_id = response.user.id
        except Exception:
            pass

    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid or expired token. Please log in again.")

    # Store in cache to avoid hammering validation on every request
    _token_cache[token] = {'user_id': user_id, 'timestamp': now}
    return user_id


async def get_current_user(request: Request) -> str:
    """Extract and validate JWT from Authorization header."""
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    token = auth_header.split(" ", 1)[1]
    return validate_token(token)
