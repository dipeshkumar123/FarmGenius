from fastapi import APIRouter, HTTPException, Depends, Request, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
import logging
import uuid
from datetime import datetime, timedelta
import os
from pathlib import Path
import tempfile
import shutil

from sqlalchemy.orm import Session

# Import the chatbot class
from src.services.chatbot import FarmChatbot
from src.services.disease_model import DiseaseModel
from src.services.price_model import CommodityPriceModel
from src.services.weather_model import WeatherModel
from src.services.crop_model import CropRecommendationModel
from src.services.user_model import UserModel
from src.services.auth import (
    hash_password, verify_password, create_access_token,
    get_current_user, require_user,
)
from src.utils.history import save_to_history
from src.db.connection import get_db
from src.db import crud as db_crud
from src.db.models import User

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize the chatbot instance
chatbot = FarmChatbot()
disease_model = DiseaseModel()
price_model = CommodityPriceModel()
weather_model = WeatherModel()
crop_model = CropRecommendationModel()
user_model = UserModel()

# Create a router
router = APIRouter()


# ═══════════════════════════════════════════════════════════════════════════════
# Auth request/response models
# ═══════════════════════════════════════════════════════════════════════════════

class RegisterRequest(BaseModel):
    email: str
    password: str
    display_name: Optional[str] = None

class LoginRequest(BaseModel):
    email: str
    password: str

class AuthResponse(BaseModel):
    token: str
    user: Dict[str, Any]

class ProfileUpdateRequest(BaseModel):
    display_name: Optional[str] = None
    language: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = None


# ═══════════════════════════════════════════════════════════════════════════════
# Auth endpoints
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/auth/register", response_model=AuthResponse, tags=["Auth"])
async def register(body: RegisterRequest, db: Session = Depends(get_db)):
    """Register a new user account."""
    email = body.email.strip().lower()
    if db_crud.get_user_by_email(db, email):
        raise HTTPException(status_code=409, detail="Email already registered")

    user_id = str(uuid.uuid4())
    display_name = body.display_name or email.split("@")[0]
    pw_hash = hash_password(body.password)

    user = db_crud.create_registered_user(
        db,
        user_id=user_id,
        email=email,
        display_name=display_name,
        password_hash=pw_hash,
    )
    token = create_access_token(user.user_id)
    return AuthResponse(token=token, user=user.to_dict())


@router.post("/auth/login", response_model=AuthResponse, tags=["Auth"])
async def login(body: LoginRequest, db: Session = Depends(get_db)):
    """Login with email & password and receive a JWT token."""
    email = body.email.strip().lower()
    user = db_crud.get_user_by_email(db, email)
    if not user or not user.password_hash:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    if not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    # Touch last_active
    user.last_active = datetime.utcnow()
    db.commit()

    token = create_access_token(user.user_id)
    return AuthResponse(token=token, user=user.to_dict())


@router.get("/auth/me", tags=["Auth"])
async def get_me(user: User = Depends(require_user)):
    """Get the currently authenticated user's profile."""
    return user.to_dict()


@router.put("/auth/profile", tags=["Auth"])
async def update_profile(
    body: ProfileUpdateRequest,
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    """Update the authenticated user's profile."""
    if body.display_name is not None:
        user.display_name = body.display_name
    if body.language is not None:
        user.language = body.language
    if body.preferences is not None:
        current = user.preferences or {}
        current.update(body.preferences)
        user.preferences = current
    user.last_active = datetime.utcnow()
    db.commit()
    db.refresh(user)
    return user.to_dict()


# Define request and response models
class QueryRequest(BaseModel):
    query: str
    user_id: Optional[str] = None
    target_lang: Optional[str] = None

class QueryResponse(BaseModel):
    response_id: str
    query: str
    response_text: str
    intent: str
    confidence: float
    source: str
    timestamp: str
    additional_data: Optional[Dict[str, Any]] = None

class ConversationHistoryRequest(BaseModel):
    user_id: str
    max_entries: Optional[int] = 10

class ConversationHistoryResponse(BaseModel):
    user_id: str
    entries: List[Dict[str, Any]]

class CropRecommendationRequest(BaseModel):
    N: int
    P: int
    K: int
    ph: float
    EC: Optional[float] = 0.0
    S: Optional[float] = 0.0
    Cu: Optional[int] = 0
    Fe: Optional[int] = 0
    Mn: Optional[int] = 0
    Zn: Optional[int] = 0
    B: Optional[int] = 0
    user_id: Optional[str] = None

class CropRecommendationResponse(BaseModel):
    response_id: str
    recommendation_text: str
    soil_params: Dict[str, Any]
    top_recommendations: List[Dict[str, Any]]
    confidence: float
    timestamp: str

class PriceRequest(BaseModel):
    commodity: str
    date: Optional[str] = None
    include_trends: Optional[bool] = False
    user_id: Optional[str] = None

class PriceResponse(BaseModel):
    response_id: str
    commodity: str
    price: Optional[float] = None
    unit: Optional[str] = None
    currency: Optional[str] = None
    date: Optional[str] = None
    found: bool
    message: Optional[str] = None
    trend_data: Optional[Dict[str, Any]] = None
    timestamp: str

class WeatherRequest(BaseModel):
    location: str
    user_id: Optional[str] = None

class ForecastRequest(BaseModel):
    location: str
    days: Optional[int] = 5
    user_id: Optional[str] = None

class CropWeatherRequest(BaseModel):
    location: str
    crop: str
    user_id: Optional[str] = None

class WeatherResponse(BaseModel):
    response_id: str
    location: str
    weather_data: Dict[str, Any]
    response_text: str
    timestamp: str

class ForecastResponse(BaseModel):
    """Response model for weather forecast."""
    response_id: str
    location: str
    forecast_data: Dict[str, Any]
    response_text: str
    timestamp: str

class CropWeatherResponse(BaseModel):
    response_id: str
    location: str
    crop: str
    weather_data: Dict[str, Any]
    crop_preferences: Dict[str, Any]
    advice: List[str]
    timestamp: str

class DiseaseIdentificationRequest(BaseModel):
    description: str
    crop: Optional[str] = None
    user_id: Optional[str] = None

class DiseaseManagementRequest(BaseModel):
    disease_id: str
    user_id: Optional[str] = None

class DiseaseByCropRequest(BaseModel):
    crop: str
    user_id: Optional[str] = None

class DiseaseResponse(BaseModel):
    response_id: str
    response_text: str
    found: bool
    disease_info: Optional[Dict[str, Any]] = None
    alternatives: Optional[List[Dict[str, Any]]] = None
    timestamp: str
    image_path: Optional[str] = None

class UserPreferencesRequest(BaseModel):
    language: Optional[str] = None
    region: Optional[str] = None
    crops: Optional[List[str]] = None
    notifications_enabled: Optional[bool] = None

class PriceAlertRequest(BaseModel):
    commodity: str
    target_price: float
    is_above: bool = True

class WeatherAlertRequest(BaseModel):
    location: str
    conditions: List[str]

class LanguageResponse(BaseModel):
    supported_languages: Dict[str, str]
    detected_language: Optional[str] = None

class VoiceQueryRequest(BaseModel):
    user_id: Optional[str] = None
    language: Optional[str] = None
    duration: Optional[float] = 5.0

class VoiceQueryResponse(BaseModel):
    success: bool
    text: Optional[str] = None
    response_text: Optional[str] = None
    audio_response: Optional[str] = None
    error: Optional[str] = None

class LanguageDetectionRequest(BaseModel):
    text: str

class TranslateRequest(BaseModel):
    text: str
    target_lang: str
    source_lang: Optional[str] = None

class TranslateResponse(BaseModel):
    translated_text: str
    source_lang: str
    target_lang: str
    success: bool

class DiseaseImageResponse(BaseModel):
    """Response model for disease image identification."""
    found: bool
    message: str
    results: List[Dict[str, Any]]
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


@router.post("/query", response_model=QueryResponse, tags=["Chat"])
async def process_query(
    request: QueryRequest,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """
    Process a natural language query and return an appropriate response.
    
    Accepts an optional Bearer token to associate the chat with the
    authenticated user. Falls back to request.user_id or 'guest'.
    """
    try:
        # Resolve effective user_id: authenticated user > body.user_id > guest
        effective_user_id = (
            current_user.user_id if current_user
            else request.user_id or "guest"
        )

        # Process the query through the chatbot
        response_data = chatbot.process_query(
            query=request.query,
            user_id=effective_user_id,
            target_lang=request.target_lang
        )
        
        # Create the API response
        response = QueryResponse(
            response_id=str(uuid.uuid4()),
            query=request.query,
            response_text=response_data.get('response_text', 'Sorry, I could not understand that.'),
            intent=response_data.get('intent', 'unknown'),
            confidence=response_data.get('confidence', 0.0),
            source=response_data.get('source', 'default'),
            timestamp=datetime.now().isoformat(),
            additional_data=response_data.get('additional_data')
        )

        # Persist to database
        if effective_user_id != "guest":
            try:
                db_crud.add_chat_entry(
                    db,
                    user_id=effective_user_id,
                    query=request.query,
                    response=response.response_text,
                    intent=response.intent,
                    confidence=response.confidence,
                    source=response.source,
                )
            except Exception as db_err:
                logger.warning(f"Failed to save chat to DB: {db_err}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


@router.post("/crops/recommend", response_model=CropRecommendationResponse, tags=["Crops"])
async def recommend_crops(request: CropRecommendationRequest):
    """
    Get crop recommendations based on soil parameters.
    
    Parameters:
    - N: Nitrogen content (mg/kg)
    - P: Phosphorus content (mg/kg)
    - K: Potassium content (mg/kg)
    - ph: Soil pH
    - EC: Electrical conductivity (optional)
    - S: Sulfur content (optional)
    - Cu: Copper content (optional)
    - Fe: Iron content (optional)
    - Mn: Manganese content (optional)
    - Zn: Zinc content (optional)
    - B: Boron content (optional)
    - user_id: Optional user identifier
    
    Returns:
    - Recommended crops with confidence scores
    - Soil parameter analysis
    - Detailed recommendation text
    """
    try:
        # Check if crop model is available
        if not chatbot.crop_model:
            raise HTTPException(
                status_code=503, 
                detail="Crop recommendation service is not available"
            )
            
        # Get soil parameters
        soil_params = {
            'N': request.N,
            'P': request.P,
            'K': request.K,
            'ph': request.ph,
            'EC': request.EC,
            'S': request.S,
            'Cu': request.Cu,
            'Fe': request.Fe,
            'Mn': request.Mn,
            'Zn': request.Zn,
            'B': request.B
        }
        
        # Get recommendations
        predictions = chatbot.crop_model.predict(soil_params)
        recommendations = predictions['top_recommendations']
        
        # Generate recommendation text
        from src.services.crop_model import generate_recommendation_text
        if recommendations and len(recommendations) > 0:
            top_crop = recommendations[0]['crop']
            confidence = recommendations[0]['confidence']
            recommendation_text = generate_recommendation_text(
                top_crop,
                confidence,
                soil_params
            )
        else:
            recommendation_text = "Unable to generate crop recommendations for the provided soil parameters."
        
        # Create the response
        response = CropRecommendationResponse(
            response_id=str(uuid.uuid4()),
            recommendation_text=recommendation_text,
            soil_params=soil_params,
            top_recommendations=recommendations,
            confidence=recommendations[0]['confidence'] if recommendations else 0.0,
            timestamp=datetime.now().isoformat()
        )
        
        # Save to conversation history if user_id is provided
        if request.user_id:
            # Create a synthetic query from the soil parameters
            query_parts = [
                f"What crops can I grow with these soil parameters?",
                f"N: {request.N}, P: {request.P}, K: {request.K}, pH: {request.ph}"
            ]
            if request.EC > 0:
                query_parts.append(f"EC: {request.EC}")
            if request.S > 0:
                query_parts.append(f"S: {request.S}")
                
            synthetic_query = " ".join(query_parts)
            
            # Add to conversation history
            chatbot.process_query(
                query=synthetic_query,
                user_id=request.user_id
            )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recommending crops: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error recommending crops: {str(e)}"
        )


@router.post("/prices/get", response_model=PriceResponse, tags=["Prices"])
async def get_commodity_price(request: PriceRequest):
    """
    Get current and historical commodity prices.
    
    Parameters:
    - commodity: Name of the commodity
    - date: Optional specific date for historical prices
    - include_trends: Whether to include price trends
    - user_id: Optional user identifier
    
    Returns:
    - Current price information
    - Price trends (if requested)
    - Market analysis
    """
    try:
        # Check if price model is available
        if not chatbot.price_model:
            raise HTTPException(
                status_code=503,
                detail="Price information service is not available"
            )
        
        # Get price data
        price_data = chatbot.price_model.get_price(
            commodity=request.commodity,
            date=request.date,
            include_trends=request.include_trends
        )
        
        # Create the response
        response = PriceResponse(
            response_id=str(uuid.uuid4()),
            commodity=request.commodity,
            price=price_data.get('price'),
            unit=price_data.get('unit'),
            currency=price_data.get('currency'),
            date=price_data.get('date'),
            found=price_data.get('found', False),
            message=price_data.get('message'),
            trend_data=price_data.get('trend_data') if request.include_trends else None,
            timestamp=datetime.now().isoformat()
        )
        
        # Save to conversation history if user_id is provided
        if request.user_id:
            # Create a query string
            date_str = f" on {request.date}" if request.date else ""
            trend_str = " trend" if request.include_trends else ""
            synthetic_query = f"What is the price of {request.commodity}{date_str}{trend_str}?"
            
            # Add to conversation history
            chatbot.process_query(
                query=synthetic_query,
                user_id=request.user_id
            )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting commodity price: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting commodity price: {str(e)}"
        )


@router.get("/prices/commodities", response_model=List[str], tags=["Prices"])
async def get_available_commodities():
    """
    Get a list of all available commodities for price queries.
    
    Returns:
    - List of commodity names
    """
    try:
        # Check if price model is available
        if not chatbot.price_model:
            raise HTTPException(
                status_code=503,
                detail="Price information service is not available"
            )
        
        # Get available commodities
        commodities = chatbot.get_available_commodities()
        
        return commodities
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting available commodities: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting available commodities: {str(e)}"
        )


@router.post("/weather/current", response_model=WeatherResponse, tags=["Weather"])
async def get_current_weather(request: WeatherRequest):
    """
    Get current weather conditions for a location.
    
    Parameters:
    - location: City or location name
    - user_id: Optional user identifier
    
    Returns:
    - Current weather conditions
    - Temperature
    - Humidity
    - Wind speed
    - Precipitation
    """
    try:
        # Check if weather model is available
        if not chatbot.weather_model:
            raise HTTPException(
                status_code=503,
                detail="Weather information service is not available"
            )
        
        # Get current weather
        weather_data = chatbot.weather_model.get_current_weather(request.location)
        
        # Format as text response
        response_text = chatbot.weather_model.format_weather_response(weather_data)
        
        # Create the response
        response = WeatherResponse(
            response_id=str(uuid.uuid4()),
            location=request.location,
            weather_data=weather_data,
            response_text=response_text,
            timestamp=datetime.now().isoformat()
        )
        
        # Save to conversation history if user_id is provided
        if request.user_id:
            synthetic_query = f"What is the current weather in {request.location}?"
            chatbot.process_query(
                query=synthetic_query,
                user_id=request.user_id
            )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting current weather: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting current weather: {str(e)}"
        )


@router.post("/weather/forecast", response_model=ForecastResponse, tags=["Weather"])
async def get_weather_forecast(request: ForecastRequest):
    """
    Get weather forecast for a location.
    
    Parameters:
    - location: City or location name
    - days: Number of days to forecast (default: 5)
    - user_id: Optional user identifier
    
    Returns:
    - Daily weather forecasts
    - Temperature trends
    - Precipitation probability
    - Wind conditions
    """
    try:
        # Check if weather model is available
        if not chatbot.weather_model:
            raise HTTPException(
                status_code=503,
                detail="Weather forecast service is not available"
            )
        
        # Get weather forecast
        forecast_result = chatbot.weather_model.get_weather_forecast(
            request.location,
            days=request.days
        )
        
        # Extract forecast data from the result
        forecast_data = forecast_result['forecast_data']
        
        # Format as text response
        response_text = chatbot.weather_model.format_forecast_response(
            forecast_data,
            days=request.days
        )
        
        # Create the response
        response = ForecastResponse(
            response_id=str(uuid.uuid4()),
            location=request.location,
            forecast_data=forecast_data,
            response_text=response_text,
            timestamp=datetime.now().isoformat()
        )
        
        # Save to conversation history if user_id is provided
        if request.user_id:
            synthetic_query = f"What is the weather forecast for {request.location}?"
            chatbot.process_query(
                query=synthetic_query,
                user_id=request.user_id
            )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting weather forecast: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting weather forecast: {str(e)}"
        )


@router.post("/weather/crop-advice", response_model=CropWeatherResponse, tags=["Weather"])
async def get_crop_weather_advice(request: CropWeatherRequest):
    """
    Get weather advice specific to a crop and location.
    
    Parameters:
    - location: The location to get weather for
    - crop: The crop to get advice for
    - user_id: Optional user identifier for personalized responses
    
    Returns:
    - Weather data and crop-specific advice
    """
    try:
        # Get weather data
        weather_data = weather_model.get_weather(request.location)
        
        # Get crop-specific advice
        crop_advice = weather_model.get_crop_weather_advice(
            location=request.location,
            crop=request.crop
        )
        
        # Create response
        response = CropWeatherResponse(
            response_id=str(uuid.uuid4()),
            location=request.location,
            crop=request.crop,
            weather_data=weather_data,
            crop_preferences=crop_advice.get('preferences', {}),
            advice=crop_advice.get('advice', []),
            timestamp=datetime.now().isoformat()
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting crop weather advice: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting crop weather advice: {str(e)}"
        )


@router.post("/diseases/identify-image", response_model=DiseaseImageResponse, tags=["Diseases"])
async def identify_disease_from_image(
    file: UploadFile = File(...),
    crop: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None)
):
    """Identify plant disease from an uploaded image using local model.
    
    This endpoint uses a trained deep learning model to identify plant diseases from images.
    The model supports various crops including Cashew, Cassava, Maize, and Tomato.
    
    Parameters:
    - file: The image file to analyze (supported formats: JPG, JPEG, PNG)
    - crop: Optional crop type to focus on (e.g., "Cashew", "Cassava", "Maize", "Tomato")
    - user_id: Optional user ID for history tracking
    
    Returns:
    - found: Whether a disease was identified
    - message: Status message
    - results: List of identified diseases with:
        - name: Disease name
        - confidence: Confidence score (0-1)
        - crop: Affected crop
        - type: Disease type (Fungal, Bacterial, Viral, Pest)
        - severity: Disease severity (low, medium, high)
        - symptoms: List of symptoms
        - treatment: List of treatment recommendations
    """
    try:
        # Save uploaded file temporarily
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Get disease identification
        result = disease_model.identify_disease_from_image(temp_path, crop)
        
        # Clean up temp file
        os.remove(temp_path)
        
        # Save to conversation history if user_id provided
        if user_id and result.get("found"):
            await save_to_history(
                user_id=user_id,
                query=f"Image upload: {file.filename}",
                response=result,
                intent="disease"
            )
        
        return DiseaseImageResponse(
            found=result.get("found", False),
            message=result.get("message", ""),
            results=result.get("results", []),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error processing disease image: {str(e)}")
        return DiseaseImageResponse(
            found=False,
            message=f"Error processing image: {str(e)}",
            results=[],
            timestamp=datetime.now().isoformat()
        )


@router.post("/history", response_model=ConversationHistoryResponse, tags=["Users"])
async def get_conversation_history(request: ConversationHistoryRequest, db: Session = Depends(get_db)):
    """
    Get the conversation history for a user.
    
    Parameters:
    - user_id: The user's identifier
    - max_entries: Maximum number of entries to return (default: 10)
    
    Returns:
    - user_id: The user's identifier
    - entries: List of conversation entries
    """
    try:
        # Try database first, fall back to legacy user_model
        try:
            entries = db_crud.get_chat_history(db, user_id=request.user_id, limit=request.max_entries)
            history = [e.to_dict() for e in entries]
        except Exception:
            history = chatbot.user_model.get_chat_history(
                user_id=request.user_id,
                limit=request.max_entries
            )
        
        return ConversationHistoryResponse(
            user_id=request.user_id,
            entries=history
        )
        
    except Exception as e:
        logger.error(f"Error getting conversation history: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting conversation history: {str(e)}"
        )


@router.get("/health", tags=["System"])
async def health_check():
    """
    Health check endpoint to verify API status.
    
    Returns:
    - API status
    - Timestamp
    - Service availability
    """
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@router.post("/users/{user_id}/preferences", response_model=Dict[str, Any], tags=["Users"])
async def update_user_preferences(user_id: str, preferences: UserPreferencesRequest, db: Session = Depends(get_db)):
    """
    Update user preferences.
    """
    try:
        preferences_dict = preferences.dict(exclude_none=True)
        
        if 'language' in preferences_dict:
            if not chatbot.language_model or not chatbot.language_model.is_language_supported(preferences_dict['language']):
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported language: {preferences_dict['language']}"
                )
        
        user = db_crud.update_user_preferences(db, user_id, preferences_dict)
        return {
            'success': True,
            'user_id': user_id,
            'preferences': user.preferences,
            'message': 'Preferences updated successfully'
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating user preferences: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error updating preferences: {str(e)}"
        )

@router.get("/users/{user_id}/preferences", response_model=Dict[str, Any], tags=["Users"])
async def get_user_preferences(user_id: str, db: Session = Depends(get_db)):
    """Get user preferences."""
    try:
        user = db_crud.get_user(db, user_id)
        if not user:
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")
            
        return {
            'success': True,
            'user_id': user_id,
            'preferences': user.preferences or {},
            'message': 'Preferences retrieved successfully'
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user preferences: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting preferences: {str(e)}"
        )

@router.get("/users/{user_id}/statistics", response_model=Dict[str, Any], tags=["Users"])
async def get_user_statistics(user_id: str, db: Session = Depends(get_db)):
    """Get user usage statistics."""
    try:
        stats = db_crud.get_user_statistics(db, user_id)
        if not stats.get('success'):
            raise HTTPException(status_code=404, detail=stats.get('message', f"User {user_id} not found"))
            
        return stats
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user statistics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting statistics: {str(e)}"
        )

@router.get("/users/{user_id}/history", response_model=Dict[str, Any], tags=["Users"])
async def get_user_history(user_id: str, query_type: Optional[str] = None, limit: int = 10, db: Session = Depends(get_db)):
    """Get user's query history from database (legacy, unprotected)."""
    try:
        entries = db_crud.get_chat_history(db, user_id=user_id, limit=limit)
        history = [e.to_dict() for e in entries]
        if query_type:
            history = [h for h in history if h.get('intent') == query_type]
        return {"history": history}
    except Exception as e:
        logger.error(f"Error getting user history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# Authenticated "my" endpoints — data scoped to the JWT user
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/my/history", tags=["Auth"])
async def my_history(
    limit: int = 50,
    query_type: Optional[str] = None,
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    """Get the authenticated user's chat history."""
    entries = db_crud.get_chat_history(db, user_id=user.user_id, limit=limit)
    history = [e.to_dict() for e in entries]
    if query_type:
        history = [h for h in history if h.get("intent") == query_type]
    return {"user_id": user.user_id, "history": history}


@router.delete("/my/history", tags=["Auth"])
async def clear_my_history(
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    """Clear the authenticated user's chat history."""
    count = db_crud.clear_chat_history(db, user_id=user.user_id)
    return {"cleared": count}


@router.get("/my/stats", tags=["Auth"])
async def my_stats(
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    """Get the authenticated user's activity statistics."""
    stats = db_crud.get_user_statistics(db, user_id=user.user_id)
    return stats

@router.post("/users/{user_id}/price-alerts", response_model=Dict[str, Any], tags=["Alerts"])
async def add_price_alert(user_id: str, alert: PriceAlertRequest):
    """
    Add a price alert for a user.
    
    Parameters:
    - user_id: User identifier
    - alert: Alert configuration
    
    Returns:
    - Alert status
    - Alert details
    """
    try:
        success = chatbot.user_model.add_price_alert(
            user_id,
            alert.commodity,
            alert.target_price,
            alert.is_above
        )
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"User {user_id} not found"
            )
            
        return {
            "success": True,
            "message": "Price alert added successfully"
        }
        
    except Exception as e:
        logger.error(f"Error adding price alert: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error adding price alert: {str(e)}"
        )

@router.post("/users/{user_id}/weather-alerts", response_model=Dict[str, Any], tags=["Alerts"])
async def add_weather_alert(user_id: str, alert: WeatherAlertRequest):
    """
    Add a weather alert for a user.
    
    Parameters:
    - user_id: User identifier
    - alert: Alert configuration
    
    Returns:
    - Alert status
    - Alert details
    """
    try:
        success = chatbot.user_model.add_weather_alert(
            user_id,
            alert.location,
            alert.conditions
        )
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"User {user_id} not found"
            )
            
        return {
            "success": True,
            "message": "Weather alert added successfully"
        }
        
    except Exception as e:
        logger.error(f"Error adding weather alert: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error adding weather alert: {str(e)}"
        )

@router.get("/languages", response_model=LanguageResponse, tags=["Language"])
async def get_supported_languages():
    """
    Get list of supported languages.
    
    Returns:
    - List of supported languages
    - Language codes
    - Language names
    """
    try:
        # Get supported languages
        supported_languages = chatbot.get_supported_languages()
        
        return LanguageResponse(
            supported_languages=supported_languages
        )
        
    except Exception as e:
        logger.error(f"Error getting supported languages: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting supported languages: {str(e)}"
        )

@router.post("/languages/detect", response_model=LanguageResponse, tags=["Language"])
async def detect_language(request: LanguageDetectionRequest):
    """
    Detect the language of the input text.
    
    Parameters:
    - text: Text to detect language
    
    Returns:
    - Detected language
    - Confidence score
    - Supported languages list
    """
    try:
        if not chatbot.language_model:
            raise HTTPException(
                status_code=503,
                detail="Language detection service is not available"
            )
            
        # Detect language
        detection_result = chatbot.language_model.detect_language(request.text)
        
        return LanguageResponse(
            supported_languages=chatbot.get_supported_languages(),
            detected_language=detection_result['detected_language']
        )
        
    except Exception as e:
        logger.error(f"Error detecting language: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error detecting language: {str(e)}"
        )

@router.post("/translate", response_model=TranslateResponse, tags=["Language"])
async def translate_text(request: TranslateRequest):
    """
    Translate text from one language to another.
    
    Parameters:
    - text: Text to translate
    - target_lang: Target language code (e.g., 'en', 'hi', 'es')
    - source_lang: Optional source language code (auto-detected if not provided)
    
    Returns:
    - Translated text
    """
    try:
        if not chatbot.language_model:
            raise HTTPException(
                status_code=503,
                detail="Translation service is not available"
            )
        
        # Detect source language if not provided
        source_lang = request.source_lang
        if not source_lang:
            detection = chatbot.language_model.detect_language(request.text)
            source_lang = detection.get('detected_language', 'en')
        
        # If source and target are the same, return as-is
        if source_lang == request.target_lang:
            return TranslateResponse(
                translated_text=request.text,
                source_lang=source_lang,
                target_lang=request.target_lang,
                success=True
            )
        
        # Translate
        translated = chatbot.language_model.translate(
            request.text,
            target_lang=request.target_lang,
            source_lang=source_lang
        )
        
        return TranslateResponse(
            translated_text=translated,
            source_lang=source_lang,
            target_lang=request.target_lang,
            success=True
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error translating text: {str(e)}")
        return TranslateResponse(
            translated_text=request.text,
            source_lang=request.source_lang or 'unknown',
            target_lang=request.target_lang,
            success=False
        )

@router.post("/prices/trends", response_model=Dict[str, Any], tags=["Prices"])
async def get_price_trends(
    commodity: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    include_analysis: bool = True,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get price trends for a commodity over a specified period.
    
    Parameters:
    - commodity: The commodity to get trends for
    - start_date: Optional start date (YYYY-MM-DD)
    - end_date: Optional end date (YYYY-MM-DD)
    - include_analysis: Whether to include trend analysis
    - user_id: Optional user identifier for personalized responses
    
    Returns:
    - Price trend data and analysis
    """
    try:
        # Get price trends
        trend_data = price_model.get_price_trends(
            commodity=commodity,
            start_date=start_date,
            end_date=end_date
        )
        
        # Add analysis if requested
        if include_analysis and trend_data.get('prices'):
            analysis = price_model.analyze_price_trends(trend_data)
            trend_data['analysis'] = analysis
            
        return {
            "response_id": str(uuid.uuid4()),
            "commodity": commodity,
            "trend_data": trend_data,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting price trends: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting price trends: {str(e)}"
        ) 