from fastapi import Request, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Callable
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In-memory rate limit store: { user_identifier: [timestamps] }
rate_limit_store = {}

class RateLimitAndLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable):
        start_time = time.time()
        
        # Identity defaults to IP address, switches to JWT if present
        identifier = request.client.host
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            identifier = auth_header.split(" ")[1][:15] # Truncated token as identifier

        # Rate Limiting Logic (Limit: 30 requests / minute)
        if request.url.path != "/health":
            now = time.time()
            if identifier not in rate_limit_store:
                rate_limit_store[identifier] = []
            
            # Evict timestamps older than 60 seconds
            rate_limit_store[identifier] = [t for t in rate_limit_store[identifier] if now - t < 60]
            
            if len(rate_limit_store[identifier]) >= 30:
                raise HTTPException(status_code=429, detail="Rate limit exceeded. Please try again later.")
            
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
    # Setup CORS for the app. Note: In real production, restrict origins to the specific app domain.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], 
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # Add custom rate limiting and logging
    app.add_middleware(RateLimitAndLoggingMiddleware)
