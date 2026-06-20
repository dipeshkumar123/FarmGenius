from fastapi import Request, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Callable
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In-memory rate limit store: { user_identifier: [timestamps] }
rate_limit_store = {}

# Paths that are excluded from rate limiting
_RATE_LIMIT_EXCLUDED = {"/health", "/api/health"}

class RateLimitAndLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable):
        start_time = time.time()

        # Identity defaults to IP address, switches to JWT if present
        identifier = request.client.host if request.client else "unknown"
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            identifier = auth_header.split(" ")[1][:15]  # Truncated token as identifier

        # Rate Limiting Logic (Limit: 30 requests / minute)
        # IMPORTANT: Use JSONResponse — do NOT raise HTTPException inside
        # BaseHTTPMiddleware.  Starlette's base middleware wraps call_next in a
        # try/except that catches Exception and returns 500; FastAPI's exception
        # handlers are never invoked, so raise HTTPException becomes a 500.
        if request.url.path not in _RATE_LIMIT_EXCLUDED:
            now = time.time()
            if identifier not in rate_limit_store:
                rate_limit_store[identifier] = []

            # Evict timestamps older than 60 seconds
            rate_limit_store[identifier] = [t for t in rate_limit_store[identifier] if now - t < 60]

            if len(rate_limit_store[identifier]) >= 30:
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded. Please try again later."},
                )

            rate_limit_store[identifier].append(now)

        # Process the request
        response = await call_next(request)

        # Logging
        process_time = time.time() - start_time
        logger.info(
            f"Method: {request.method} Path: {request.url.path} "
            f"UserIdentifier: {identifier} ResponseTime: {process_time:.4f}s Status: {response.status_code}"
        )

        return response


def setup_middleware(app: FastAPI):
    # Setup CORS for the app. 
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173", 
            "http://localhost:3000",
            "https://frontend-pi-liart-51.vercel.app"
        ],
        allow_origin_regex=r"https://frontend-.*\.vercel\.app",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # Add custom rate limiting and logging
    app.add_middleware(RateLimitAndLoggingMiddleware)
