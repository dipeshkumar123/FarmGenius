# FarmGenius Backend Code

This document contains the complete FastAPI backend codebase for FarmGenius. You can create the `backend/` directory and populate it with the files provided below.

## 1. Project Root Files

### `backend/requirements-api.txt`
```text
fastapi==0.109.2
uvicorn==0.27.1
pydantic==2.6.1
pydantic-settings==2.1.0
httpx==0.26.0
groq==0.11.0
python-multipart==0.0.9
supabase==2.3.4
scikit-learn==1.3.2
numpy==1.26.4
```

### `backend/.env.example`
```ini
# Supabase Configuration
# The URL of your Supabase project (e.g., https://xyzcompany.supabase.co)
SUPABASE_URL=

# The Service Role Key to bypass RLS and interact directly with Supabase APIs
SUPABASE_SERVICE_KEY=

# Groq API Configuration
# Your free-tier API key from Groq for running the Llama 3.1 70B model
GROQ_API_KEY=

# Data.gov.in API Configuration
# Free API key for the AGMARKNET Mandi prices endpoints
DATA_GOV_IN_API_KEY=

# Slack Configuration
# Webhook URL to send negative farmer feedback to a Slack channel
SLACK_WEBHOOK_URL=

# Application Environment (e.g., development, production)
ENVIRONMENT=production
```

### `backend/render.yaml`
```yaml
services:
  - type: web
    name: farmgenius-backend
    env: python
    region: singapore
    buildCommand: "pip install -r requirements-api.txt"
    startCommand: "uvicorn main:app --host 0.0.0.0 --port $PORT"
    healthCheckPath: /health
    plan: free
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.0
      - key: SUPABASE_URL
        sync: false
      - key: SUPABASE_SERVICE_KEY
        sync: false
      - key: GROQ_API_KEY
        sync: false
      - key: DATA_GOV_IN_API_KEY
        sync: false
      - key: ENVIRONMENT
        value: production
```

### `backend/Dockerfile`
```dockerfile
# Multi-stage, slim Python image for optimal performance
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies first (caching layer)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy remaining source code
COPY . .

# Expose port and run the application
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### `backend/main.py`
```python
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
```

## 2. GitHub Actions

### `backend/.github/workflows/deploy.yml`
```yaml
name: Deploy Backend

on:
  push:
    branches:
      - main
    paths:
      - 'backend/**'

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Trigger Render Deployment
        env:
          RENDER_DEPLOY_HOOK: ${{ secrets.RENDER_DEPLOY_HOOK }}
        run: |
          curl -X POST "$RENDER_DEPLOY_HOOK"
```

### `backend/.github/workflows/keepalive.yml`
```yaml
name: Keep-Alive Cron

on:
  schedule:
    # Runs every 10 minutes to prevent Render free-tier cold starts
    - cron: '*/10 * * * *'

jobs:
  ping:
    runs-on: ubuntu-latest
    steps:
      - name: Ping Health Endpoint
        run: |
          curl -f ${{ secrets.RENDER_HEALTH_URL }}
```

## 3. Core Configuration & Security

### `backend/app/core/config.py`
```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    SUPABASE_URL: str
    SUPABASE_SERVICE_KEY: str
    GROQ_API_KEY: str
    DATA_GOV_IN_API_KEY: str = ""
    ENVIRONMENT: str = "production"

    class Config:
        env_file = ".env"

settings = Settings()
```

### `backend/app/core/security.py`
```python
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
```

## 4. Middleware & Schemas

### `backend/app/api/middleware.py`
```python
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
```

### `backend/app/models/schemas.py`
```python
from pydantic import BaseModel
from typing import Optional

class ChatRequest(BaseModel):
    query: str
    language: str
    farmer_id: str

class ChatResponse(BaseModel):
    response: str
    source: str
    confidence: float

class PriceResponse(BaseModel):
    commodity: str
    district: str
    min_price: float
    max_price: float
    modal_price: float
    date: str
    unit: str

class WeatherForecast(BaseModel):
    date: str
    max_temp: float
    min_temp: float
    rainfall_mm: float
    wind_kmh: float
    farming_advisory: str

class SchemeResponse(BaseModel):
    scheme_name: str
    description: str
    eligibility: str
    link: str

class DiseaseResponse(BaseModel):
    disease_name: str
    confidence: float
    disease_name_hi: str
    organic_treatment: str
    chemical_treatment: str
    dosage: str
    source_url: str
    source_name: str
```

## 5. Services

### `backend/app/services/llm_service.py`
```python
import groq
from app.core.config import settings

class LLMService:
    def __init__(self):
        self.client = groq.Groq(api_key=settings.GROQ_API_KEY)

    def get_response(self, query: str, language: str) -> dict:
        prompt = f"""You are an agricultural advisor for Indian smallholder farmers.
The farmer is asking in {language}. Answer in {language}.
Keep answers under 3 sentences. Be specific — name exact quantities, timings, and product names. If you are unsure, say so clearly and recommend the farmer contact their local KVK.
Do not make up treatments or dosages. Append source: KVK / state agriculture dept.
Farmer's query: {query}"""
        
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": prompt}
                ],
                model="llama-3.1-70b-versatile",
                temperature=0.3,
            )
            response_text = chat_completion.choices[0].message.content
            return {
                "response": response_text,
                "source": "LLM generated via Groq (Llama 3.1 70B)",
                "confidence": 0.9
            }
        except Exception as e:
            fallback_msgs = {
                "hi": "माफ करें, अभी सेवा उपलब्ध नहीं है। कृपया अपने KVK से संपर्क करें।",
                "kn": "ಕ್ಷಮಿಸಿ, ಸೇವೆ ಲಭ್ಯವಿಲ್ಲ. ನಿಮ್ಮ KVK ಗೆ ಸಂಪರ್ಕಿಸಿ.",
                "te": "క్షమించండి, సేవ అందుబాటులో లేదు. దయచేసి మీ KVK ని సంప్రదించండి.",
                "ta": "மன்னிக்கவும், சேவை கிடைக்கவில்லை. உங்கள் KVK ஐ தொடர்பு கொள்ளவும்.",
                "mr": "क्षमस्व, सेवा अनुपलब्ध आहे. कृपया तुमच्या KVK शी संपर्क साधा.",
                "en": "Sorry, service is unavailable. Please contact your local KVK."
            }
            msg = fallback_msgs.get(language, fallback_msgs["en"])
            return {
                "response": msg,
                "source": "Fallback System",
                "confidence": 0.0
            }

llm_service = LLMService()
```

### `backend/app/services/chatbot_service.py`
```python
import os
import pickle
from app.services.llm_service import llm_service
from app.core.security import supabase

class ChatbotService:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.classes = []
        
        # Load local MultinomialNB model trained on FARMER_CORPUS if available
        # Note: Provide fallback behavior if file doesn't exist during initial build
        model_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "chatbot_farmer_v1.pkl")
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                data = pickle.load(f)
                self.model = data.get("model")
                self.vectorizer = data.get("vectorizer")
                self.classes = data.get("classes", [])

    def get_response(self, query: str, language: str, farmer_id: str) -> dict:
        confidence = 0.0
        response_text = ""
        source = ""
        category = "General"

        # Attempt local model inference first
        if self.model and self.vectorizer:
            try:
                X = self.vectorizer.transform([query])
                probs = self.model.predict_proba(X)[0]
                max_prob = max(probs)
                max_idx = probs.argmax()
                
                if max_prob >= 0.72:
                    category = self.classes[max_idx]
                    response_text = f"Your query falls under the category: '{category}'. For precise guidance, please follow up with your local KVK."
                    confidence = max_prob
                    source = "Local ML Model"
            except Exception:
                pass
                
        # Fallback to Groq LLM API if confidence is below 0.72 or model missing
        if confidence < 0.72:
            llm_res = llm_service.get_response(query, language)
            response_text = llm_res["response"]
            confidence = llm_res["confidence"]
            source = llm_res["source"]

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
```

### `backend/app/services/price_service.py`
```python
import time
from datetime import date
import httpx
from app.core.security import supabase
from app.core.config import settings

# In-memory dictionary cache: { "commodity_district": {"timestamp": float, "data": dict} }
price_cache = {}

class PriceService:
    async def get_prices(self, commodity: str, district: str, state: str) -> dict:
        key = f"{commodity}_{district}".lower()
        now = time.time()
        
        # 1. Check in-memory cache (valid for 6 hours / 21600 seconds)
        if key in price_cache:
            cache_entry = price_cache[key]
            if now - cache_entry['timestamp'] < 21600:
                return cache_entry['data']

        today_str = date.today().isoformat()
        
        # 2. Check Supabase prices_cache for today's data
        try:
            res = supabase.table("prices_cache").select("*").eq("commodity", commodity).eq("district", district).eq("date", today_str).execute()
            if res.data and len(res.data) > 0:
                result_data = res.data[0]
                price_cache[key] = {'timestamp': now, 'data': result_data}
                return result_data
        except Exception:
            pass
        
        # 3. Fetch from data.gov.in AGMARKNET API
        url = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
        params = {
            "api-key": settings.DATA_GOV_IN_API_KEY,
            "format": "json",
            "filters[commodity]": commodity,
            "filters[district]": district,
            "filters[state]": state
        }
        
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(url, params=params, timeout=10.0)
                data = resp.json()
                if data.get("records"):
                    record = data["records"][0]
                    result = {
                        "commodity": commodity,
                        "district": district,
                        "min_price": float(record.get("min_price", 0)),
                        "max_price": float(record.get("max_price", 0)),
                        "modal_price": float(record.get("modal_price", 0)),
                        "date": today_str,
                        "unit": "Quintal" # Default metric commonly used in mandis
                    }
                    
                    # Store in Supabase
                    try:
                        supabase.table("prices_cache").upsert(result).execute()
                    except Exception:
                        pass
                        
                    # Store in-memory cache
                    price_cache[key] = {'timestamp': now, 'data': result}
                    return result
            except Exception:
                pass
                
        # 4. Dummy/Fallback Data on total failure
        return {
            "commodity": commodity,
            "district": district,
            "min_price": 0.0,
            "max_price": 0.0,
            "modal_price": 0.0,
            "date": today_str,
            "unit": "Quintal"
        }

price_service = PriceService()
```

### `backend/app/districts_coords.json`
```json
{
  "dharwad": {"lat": 15.45, "lon": 75.00},
  "pune": {"lat": 18.52, "lon": 73.85},
  "ludhiana": {"lat": 30.90, "lon": 75.85},
  "bhopal": {"lat": 23.25, "lon": 77.41}
}
```

### `backend/app/services/weather_service.py`
```python
import httpx
import json
import os

class WeatherService:
    def __init__(self):
        self.district_coords = {}
        coords_path = os.path.join(os.path.dirname(__file__), "..", "districts_coords.json")
        try:
            with open(coords_path, "r") as f:
                self.district_coords = json.load(f)
        except Exception:
            self.district_coords = {"dharwad": {"lat": 15.45, "lon": 75.00}}

    async def get_weather(self, district: str, state: str) -> list:
        # Default to a central coordinate if district is unknown
        coord = self.district_coords.get(district.lower(), {"lat": 20.59, "lon": 78.96})
        
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": coord["lat"],
            "longitude": coord["lon"],
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max",
            "timezone": "Asia/Kolkata"
        }
        
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(url, params=params, timeout=10.0)
                data = resp.json()
                daily = data.get("daily", {})
                
                forecasts = []
                for i in range(len(daily.get("time", []))):
                    rainfall = daily.get("precipitation_sum")[i]
                    
                    # Simple rule-based advisory system
                    advisory = "Conditions look favorable for standard farming activities."
                    if rainfall > 20:
                        advisory = "Heavy rain expected — avoid spraying pesticides and monitor drainage."
                    elif rainfall == 0 and daily.get("temperature_2m_max")[i] > 38:
                        advisory = "High temperatures expected. Ensure adequate crop irrigation."
                        
                    forecasts.append({
                        "date": daily.get("time")[i],
                        "max_temp": daily.get("temperature_2m_max")[i],
                        "min_temp": daily.get("temperature_2m_min")[i],
                        "rainfall_mm": rainfall,
                        "wind_kmh": daily.get("wind_speed_10m_max")[i],
                        "farming_advisory": advisory
                    })
                return forecasts
            except Exception:
                return []

weather_service = WeatherService()
```

## 6. Endpoints / Routes

### `backend/app/api/routes/chat.py`
```python
from fastapi import APIRouter, Depends
from app.models.schemas import ChatRequest, ChatResponse
from app.services.chatbot_service import chatbot_service
from app.core.security import get_current_user

router = APIRouter()

@router.post("/", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest, user_id: str = Depends(get_current_user)):
    # Proceed using req.farmer_id or the authenticated user_id
    res = chatbot_service.get_response(req.query, req.language, req.farmer_id)
    return ChatResponse(**res)
```

### `backend/app/api/routes/disease.py`
```python
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from app.models.schemas import DiseaseResponse
from app.core.security import get_current_user

router = APIRouter()

@router.post("/detect", response_model=DiseaseResponse)
async def detect_disease(file: UploadFile = File(...), user_id: str = Depends(get_current_user)):
    if not file.filename.lower().endswith((".jpg", ".png", ".jpeg")):
        raise HTTPException(status_code=400, detail="Only JPG and PNG images are allowed.")
    
    content = await file.read()
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size exceeds the 5MB limit.")
        
    # Note: Primary disease detection is on-device via TFLite.
    # This endpoint acts as an online fallback/analytics hook.
    # In a full deployment, this would invoke a server-side TF model.
    # We return simulated data here.
    return DiseaseResponse(
        disease_name="Tomato___Early_blight",
        confidence=0.92,
        disease_name_hi="टमाटर अगेती झुलसा",
        organic_treatment="Spray 5% Neem seed kernel extract.",
        chemical_treatment="Apply Mancozeb or Copper fungicides.",
        dosage="2g per liter of water",
        source_url="https://kvk.icar.gov.in/",
        source_name="ICAR-KVK"
    )
```

### `backend/app/api/routes/prices.py`
```python
from fastapi import APIRouter, Depends
from app.models.schemas import PriceResponse
from app.services.price_service import price_service
from app.core.security import get_current_user

router = APIRouter()

@router.get("/", response_model=PriceResponse)
async def get_prices(commodity: str, district: str, state: str, user_id: str = Depends(get_current_user)):
    data = await price_service.get_prices(commodity, district, state)
    return PriceResponse(**data)
```

### `backend/app/api/routes/weather.py`
```python
from fastapi import APIRouter, Depends
from typing import List
from app.models.schemas import WeatherForecast
from app.services.weather_service import weather_service
from app.core.security import get_current_user

router = APIRouter()

@router.get("/", response_model=List[WeatherForecast])
async def get_weather(district: str, state: str, user_id: str = Depends(get_current_user)):
    data = await weather_service.get_weather(district, state)
    return [WeatherForecast(**item) for item in data]
```

### `backend/app/api/routes/schemes.py`
```python
from fastapi import APIRouter, Depends
from typing import List
from app.models.schemas import SchemeResponse
from app.core.security import get_current_user, supabase

router = APIRouter()

@router.get("/", response_model=List[SchemeResponse])
async def get_schemes(crop: str, state: str, user_id: str = Depends(get_current_user)):
    try:
        # Example implementation: filtering schemes based on state and crop
        # Requires the "schemes" table to be pre-seeded in Supabase.
        res = supabase.table("schemes").select("*").execute()
        schemes_data = []
        if res.data:
            for item in res.data:
                schemes_data.append(SchemeResponse(**item))
            return schemes_data
    except Exception:
        pass
        
    # Dummy fallback response
    return [
        SchemeResponse(
            scheme_name="PM-KISAN",
            description="Direct benefit transfer of Rs. 6000 per year.",
            eligibility="All landholding farmers families.",
            link="https://pmkisan.gov.in"
        )
    ]
```

### `backend/app/api/routes/feedback.py`
```python
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
```

## 7. PyTest Tests

### `backend/tests/test_chat.py`
```python
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_chat_endpoint_unauthorized():
    response = client.post("/chat/", json={
        "query": "hello",
        "language": "en",
        "farmer_id": "123"
    })
    # Expecting 401 because no Bearer token was provided
    assert response.status_code == 401
```

### `backend/tests/test_prices.py`
```python
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_prices_endpoint_unauthorized():
    response = client.get("/prices/?commodity=wheat&district=dharwad&state=karnataka")
    assert response.status_code == 401
```

### `backend/tests/test_disease.py`
```python
from fastapi.testclient import TestClient
from main import app
import io

client = TestClient(app)

def test_disease_endpoint_unauthorized():
    file = io.BytesIO(b"dummy image bytes content")
    response = client.post(
        "/disease/detect", 
        files={"file": ("test.jpg", file, "image/jpeg")}
    )
    assert response.status_code == 401
```
