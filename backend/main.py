import sys
import os
import traceback

sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI

import_error_msg = None

try:
    from app.api.routes import chat, disease, prices, weather, schemes, feedback, auth, crop
    from app.api.middleware import setup_middleware
except Exception as e:
    import_error_msg = traceback.format_exc()

app = FastAPI(title="FarmGenius API", description="Backend services for the FarmGenius Android app.", root_path="/api")

if import_error_msg is None:
    # Initialize middlewares (CORS, Rate Limiting, Logging)
    setup_middleware(app)
    
    # Include Routers
    app.include_router(auth.router, prefix="/auth", tags=["Auth"])
    app.include_router(chat.router, prefix="/chat", tags=["Chat"])
    app.include_router(disease.router, prefix="/disease", tags=["Disease"])
    app.include_router(prices.router, prefix="/prices", tags=["Prices"])
    app.include_router(weather.router, prefix="/weather", tags=["Weather"])
    app.include_router(schemes.router, prefix="/schemes", tags=["Schemes"])
    app.include_router(feedback.router, prefix="/feedback", tags=["Feedback"])
    app.include_router(crop.router, prefix="/crop", tags=["Crop"])

@app.get("/health", tags=["Health"])
def health_check():
    if import_error_msg:
        return {"status": "error", "message": import_error_msg}
    return {"status": "ok", "message": "FarmGenius API is running gracefully."}




