import uvicorn
import logging
import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from datetime import datetime

from src.api.routes import router

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create the FastAPI app
app = FastAPI(
    title="FarmAssist API",
    description="""
    ## Agricultural Support API
    
    This API provides a comprehensive suite of tools for farmers, including:
    
    ### Categories
    
    * **Chat**: Natural language interface for agricultural queries
    * **Crops**: Crop recommendation based on soil parameters
    * **Weather**: Current conditions, forecasts, and crop-specific advice
    * **Prices**: Commodity price information and trends
    * **Diseases**: Plant disease identification from images
    * **Users**: User profile management and history
    * **Alerts**: Price and weather alert configuration
    * **Language**: Language detection and translation support
    * **System**: System health and status
    
    The API supports image-based disease detection for Cashew, Cassava, Maize, and Tomato
    with detailed identification including disease type, severity, symptoms, and treatments.
    """,
    version="1.0.0",
    docs_url="/docs",  # Enable Swagger UI
    redoc_url="/redoc",  # Enable ReDoc
    openapi_url="/openapi.json"  # Enable OpenAPI schema
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the router with non-redundant tags
app.include_router(
    router,
    prefix="/api"
)

# Root endpoint
@app.get("/", tags=["System"])
async def root():
    """Get API status and information."""
    return {
        "status": "running",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "weather": "enabled",
            "price": "enabled",
            "disease": "enabled"
        }
    }

# Error handler for all exceptions
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"message": "An unexpected error occurred", "detail": str(exc)}
    )

def start():
    """Start the API server."""
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    debug = os.getenv("DEBUG", "False").lower() == "true"
    
    logger.info(f"Starting Farm Chatbot API on {host}:{port}")
    
    uvicorn.run(
        "src.api.app:app",
        host=host,
        port=port,
        reload=debug
    )

if __name__ == "__main__":
    start() 