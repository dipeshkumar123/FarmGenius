import uvicorn
import logging
import os
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

from src.routes import router
from src.db.connection import init_db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Lifespan: runs once on startup (init DB tables)
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing database tables...")
    init_db()
    yield


# Create the FastAPI app
app = FastAPI(
    title="FarmGenius API",
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
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
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

# ---------------------------------------------------------------------------
# Frontend serving — supports both dev (frontend/) and deploy (public/)
# ---------------------------------------------------------------------------
_project_root = Path(__file__).resolve().parent.parent
_frontend_dir = _project_root / "public"
if not _frontend_dir.exists():
    _frontend_dir = _project_root / "frontend"

@app.get("/", tags=["System"])
async def root():
    """Serve the frontend SPA."""
    index_file = _frontend_dir / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {
        "status": "running",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "weather": "enabled",
            "price": "enabled",
            "disease": "enabled"
        }
    }

# Mount frontend static files (CSS, JS, assets)
if _frontend_dir.exists():
    for sub in ("css", "js", "assets"):
        sub_dir = _frontend_dir / sub
        if sub_dir.exists():
            app.mount(f"/{sub}", StaticFiles(directory=sub_dir), name=sub)

# Error handler for all exceptions
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"message": "An unexpected error occurred", "detail": str(exc)}
    )

def start():
    """Start the API server (local development)."""
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    debug = os.getenv("DEBUG", "False").lower() == "true"
    
    logger.info(f"Starting FarmGenius API on {host}:{port}")
    
    uvicorn.run(
        "src.app:app",
        host=host,
        port=port,
        reload=debug
    )

if __name__ == "__main__":
    start() 