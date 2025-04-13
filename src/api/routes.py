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

# Import the chatbot class
from src.models.chatbot import FarmChatbot
from src.models.disease_model import DiseaseModel
from src.models.price_model import CommodityPriceModel
from src.models.weather_model import WeatherModel
from src.models.crop_model import CropRecommendationModel
from src.models.user_model import UserModel
from src.utils.history import save_to_history

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

class DiseaseImageResponse(BaseModel):
    """Response model for disease image identification."""
    found: bool
    message: str
    results: List[Dict[str, Any]]
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


@router.post("/query", response_model=QueryResponse, tags=["Chat"])
async def process_query(request: QueryRequest):
    """
    Process a natural language query and return an appropriate response.
    
    This endpoint handles various types of queries including:
    - Weather information
    - Crop recommendations
    - Disease identification
    - Price information
    - General farming advice
    
    Parameters:
    - query: The user's question or request
    - user_id: Optional user identifier for personalized responses
    - target_lang: Optional target language for response translation
    
    Returns:
    - response_id: Unique identifier for the response
    - response_text: The generated response
    - intent: Detected intent of the query
    - confidence: Confidence score of the response
    - source: Source of the information
    - timestamp: Response generation time
    - additional_data: Any extra relevant information
    """
    try:
        # Process the query through the chatbot
        response_data = chatbot.process_query(
            query=request.query,
            user_id=request.user_id,
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
        from src.models.crop_model import generate_recommendation_text
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
async def get_conversation_history(request: ConversationHistoryRequest):
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
        # Get conversation history from user model
        history = chatbot.user_model.get_chat_history(
            user_id=request.user_id,
            limit=request.max_entries
        )
        
        # Create the API response
        response = ConversationHistoryResponse(
            user_id=request.user_id,
            entries=history
        )
        
        return response
        
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
async def update_user_preferences(user_id: str, preferences: UserPreferencesRequest):
    """
    Update user preferences.
    
    Parameters:
    - user_id: User identifier
    - preferences: User preference settings
    
    Returns:
    - Updated preferences
    - Success status
    """
    try:
        # Convert Pydantic model to dict
        preferences_dict = preferences.dict(exclude_none=True)
        
        # Validate language if provided
        if 'language' in preferences_dict:
            if not chatbot.language_model or not chatbot.language_model.is_language_supported(preferences_dict['language']):
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported language: {preferences_dict['language']}"
                )
        
        # Update preferences
        result = chatbot.user_model.update_preferences(user_id, preferences_dict)
        
        if not result['success']:
            raise HTTPException(
                status_code=404,
                detail=result['message']
            )
            
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating user preferences: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error updating preferences: {str(e)}"
        )

@router.get("/users/{user_id}/preferences", response_model=Dict[str, Any], tags=["Users"])
async def get_user_preferences(user_id: str):
    """
    Get user preferences.
    
    Parameters:
    - user_id: User identifier
    
    Returns:
    - User preference settings
    """
    try:
        preferences = chatbot.user_model.get_preferences(user_id)
        
        if not preferences:
            raise HTTPException(
                status_code=404,
                detail=f"User {user_id} not found"
            )
            
        return {
            'success': True,
            'user_id': user_id,
            'preferences': preferences,
            'message': 'Preferences retrieved successfully'
        }
        
    except Exception as e:
        logger.error(f"Error getting user preferences: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting preferences: {str(e)}"
        )

@router.get("/users/{user_id}/statistics", response_model=Dict[str, Any], tags=["Users"])
async def get_user_statistics(user_id: str):
    """
    Get user usage statistics.
    
    Parameters:
    - user_id: User identifier
    
    Returns:
    - Query counts
    - Feature usage
    - Activity timeline
    """
    try:
        statistics = chatbot.user_model.get_user_statistics(user_id)
        
        if not statistics:
            raise HTTPException(
                status_code=404,
                detail=f"User {user_id} not found"
            )
            
        return {
            "success": True,
            "statistics": statistics
        }
        
    except Exception as e:
        logger.error(f"Error getting user statistics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting statistics: {str(e)}"
        )

@router.get("/users/{user_id}/history", response_model=Dict[str, Any], tags=["Users"])
async def get_user_history(user_id: str, query_type: Optional[str] = None, limit: int = 10):
    """
    Get user's query history.
    
    Args:
        user_id (str): User identifier
        query_type (str, optional): Type of queries to retrieve
        limit (int): Maximum number of entries to return
        
    Returns:
        Dict[str, Any]: User's query history
    """
    try:
        history = user_model.get_query_history(
            user_id=user_id,
            query_type=query_type,
            limit=limit
        )
        return {"history": history}
    except Exception as e:
        logger.error(f"Error getting user history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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