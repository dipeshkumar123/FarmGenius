from fastapi import FastAPI
from app.api.routes import chat, disease, prices, weather, schemes, feedback
from app.api.middleware import setup_middleware

app = FastAPI(title="FarmGenius API", description="Backend services for the FarmGenius Android app.")

# Initialize middlewares (CORS, Rate Limiting, Logging)
setup_middleware(app)

# Include Routers
app.include_router(chat.router, prefix="/chat", tags=["Chat"])
app.include_router(disease.router, prefix="/disease", tags=["Disease"])
app.include_router(prices.router, prefix="/prices", tags=["Prices"])
app.include_router(weather.router, prefix="/weather", tags=["Weather"])
app.include_router(schemes.router, prefix="/schemes", tags=["Schemes"])
app.include_router(feedback.router, prefix="/feedback", tags=["Feedback"])

@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok", "message": "FarmGenius API is running gracefully."}
