import logging
import os
import re
import json
from datetime import datetime
from dotenv import load_dotenv
from typing import Optional, Dict, Any, List
import uuid

from src.models.faq_model import FAQModel
from src.models.crop_model import CropRecommendationModel, generate_recommendation_text, get_crop_description
from src.models.price_model import CommodityPriceModel
from src.models.weather_model import WeatherModel
from src.models.disease_model import DiseaseModel
from src.models.user_model import UserModel
from src.models.language_model import LanguageModel
from src.models.voice_model import VoiceModel
from src.models.deepseek_model import DeepSeekModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FarmChatbot:
    """
    Main chatbot class that orchestrates the different specialized models.
    This is the primary entry point for interacting with the chatbot.
    """
    
    def __init__(self):
        """Initialize the chatbot and its component models."""
        # Load environment variables
        load_dotenv()
        
        # Dictionary to store the conversation history
        self.conversation_history = {}
        
        # Get model paths from environment
        faq_model_path = os.getenv("FAQ_MODEL_PATH", "models/faq_model.pkl")
        crop_model_path = os.getenv("CROP_MODEL_PATH", "models/crop_recommendation_model.pkl")
        
        # Check the use of live data for each component
        use_live_weather = os.getenv("USE_LIVE_WEATHER_DATA", "False").lower() == "true"
        use_live_price = os.getenv("USE_LIVE_PRICE_DATA", "False").lower() == "true"
        use_live_disease = os.getenv("USE_LIVE_DISEASE_DATA", "False").lower() == "true"

        # Initialize DeepSeek model
        try:
            self.deepseek_model = DeepSeekModel()
            logger.info("DeepSeek model initialized")
        except Exception as e:
            logger.error(f"Error initializing DeepSeek model: {str(e)}")
            self.deepseek_model = None

        # Initialize voice model
        try:
            self.voice_model = VoiceModel()
            logger.info("Voice model initialized")
        except Exception as e:
            logger.error(f"Error initializing voice model: {str(e)}")
            self.voice_model = None

        # Initialize language model
        try:
            self.language_model = LanguageModel()
            logger.info("Language model initialized")
        except Exception as e:
            logger.error(f"Error initializing language model: {str(e)}")
            self.language_model = None

        # Initialize FAQ model
        try:
            self.faq_model = FAQModel()
            logger.info("FAQ model initialized")
        except Exception as e:
            logger.error(f"Error initializing FAQ model: {str(e)}")
            self.faq_model = None
        
        # Initialize crop recommendation model
        try:
            self.crop_model = CropRecommendationModel(model_path=crop_model_path)
            logger.info(f"Crop recommendation model initialized from {crop_model_path}")
        except Exception as e:
            logger.error(f"Error initializing crop model: {str(e)}")
            self.crop_model = None
        
        # Initialize price information model
        try:
            price_api_key = os.getenv("PRICE_API_KEY") if use_live_price else None
            price_api_url = os.getenv("PRICE_API_URL") if use_live_price else None
            
            self.price_model = CommodityPriceModel(
                use_live_data=use_live_price,
                api_key=price_api_key,
                api_url=price_api_url
            )
            logger.info(f"Price information model initialized (live data: {use_live_price})")
        except Exception as e:
            logger.error(f"Error initializing price model: {str(e)}")
            self.price_model = None
        
        # Initialize weather model
        try:
            # Weather model now gets configuration from environment variables
            self.weather_model = WeatherModel()
            logger.info("Weather model initialized")
        except Exception as e:
            logger.error(f"Error initializing weather model: {str(e)}")
            self.weather_model = None
        
        # Initialize disease model
        try:
            self.disease_model = DiseaseModel()
            logger.info("Disease model initialized")
        except Exception as e:
            logger.error(f"Error initializing disease model: {str(e)}")
            self.disease_model = None
        
        # Initialize user model
        self.user_model = UserModel()
        
        logger.info("FarmChatbot initialized")
    
    def _detect_intent(self, query):
        """
        Detect the user's intent from the query.
        
        Args:
            query (str): User query
        
        Returns:
            str: Detected intent (faq, crop, price, weather, disease, greeting, unknown)
        """
        # Convert query to lowercase for case-insensitive matching
        query_lower = query.lower()
        
        # Check for greetings
        greeting_patterns = [
            r'\bhello\b', r'\bhi\b', r'\bhey\b', r'\bgreetings\b', 
            r'\bgood morning\b', r'\bgood afternoon\b', r'\bgood evening\b'
        ]
        for pattern in greeting_patterns:
            if re.search(pattern, query_lower):
                return 'greeting'
        
        # Check for crop-related queries
        crop_patterns = [
            r'\bcrop recommendation\b', r'\bwhat crop\b', r'\bwhich crop\b',
            r'\bsuitable crop\b', r'\bplant in\b', r'\bgrow in\b', r'\bsoil\b',
            r'\bcrops should i plant\b', r'\bwhat to plant\b', r'\bplanting\b'
        ]
        for pattern in crop_patterns:
            if re.search(pattern, query_lower):
                return 'crop'
        
        # Check for price-related queries
        price_patterns = [
            r'\bprice\b', r'\bcost\b', r'\bmarket\b', r'\bsell\b', 
            r'\brate\b', r'\bvalue\b', r'\bexpense\b', r'\bworth\b',
            r'\bhow much is\b', r'\bhow much are\b', r'\btrading at\b',
            r'\bcurrent price\b', r'\bmarket price\b'
        ]
        for pattern in price_patterns:
            if re.search(pattern, query_lower):
                return 'price'
        
        # Check for weather-related queries
        weather_patterns = [
            r'\bweather\b', r'\btemperature\b', r'\brain\b', r'\bforecast\b', 
            r'\bhumidity\b', r'\bclimatic\b', r'\bprecipitation\b',
            r'\bhumid\b', r'\bhot\b', r'\bcold\b', r'\bcloudy\b',
            r'\bsunny\b', r'\bstorm\b', r'\bthunder\b', r'\bwind\b',
            r'\bclimate\b', r'\bmeteorological\b'
        ]
        for pattern in weather_patterns:
            if re.search(pattern, query_lower):
                return 'weather'
        
        # Check for disease-related queries
        disease_patterns = [
            r'\bdisease\b', r'\bpest\b', r'\binfect\b', r'\bsymptom\b', 
            r'\bspots\b', r'\blesion\b', r'\bwilt\b', r'\brust\b',
            r'\bmildew\b', r'\brot\b', r'\bblight\b', r'\binsect\b',
            r'\bbug\b', r'\baphid\b', r'\bbeetle\b', r'\bcaterpillar\b',
            r'\bmoth\b', r'\bworm\b', r'\bmosaic\b', r'\byellow\b',
            r'\btreat\b', r'\bcure\b', r'\bcontrol\b', r'\bprevent\b',
            r'\bmanagement\b', r'\bprotection\b'
        ]
        for pattern in disease_patterns:
            if re.search(pattern, query_lower):
                return 'disease'
        
        # Default to FAQ
        return 'faq'
    
    def _get_greeting_response(self):
        """Generate a greeting response."""
        greetings = [
            "Hello! I'm your farming assistant. How can I help you today?",
            "Hi there! Ask me anything about farming, crops, weather, or market prices.",
            "Greetings! I'm here to help with your agricultural queries.",
            "Welcome! How can I assist you with your farming needs today?"
        ]
        
        import random
        return self._get_base_response(
            response_text=random.choice(greetings),
            intent='greeting',
            confidence=1.0,
            source='static'
        )

    def _get_base_response(self, response_text: str, intent: str, confidence: float, source: str, additional_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a base response dictionary.
        
        Args:
            response_text (str): Text response
            intent (str): Detected intent
            confidence (float): Confidence score
            source (str): Source of the response
            additional_data (dict, optional): Additional data for the response
            
        Returns:
            dict: Response data
        """
        # Generate a response ID - in real implementation, this would likely be a UUID
        import random
        response_id = str(random.randint(1000, 9999))
        
        # Create the response data structure
        response_data = {
            "response_text": response_text,
            "confidence": confidence,
            "source": source,
            "intent": intent,
            "response_id": response_id,
            "timestamp": datetime.now().isoformat(),
            "found": True  # Add default found attribute to all responses
        }
        
        # Add any additional data
        if additional_data:
            response_data.update(additional_data)
            
        return response_data
    
    def _extract_commodity_info(self, query):
        """
        Extract commodity information from a query.
        
        Args:
            query (str): User query
        
        Returns:
            dict: Extracted commodity information
        """
        commodity_info = {}
        
        # Common agricultural commodities
        commodities = [
            'wheat', 'rice', 'corn', 'soybeans', 'cotton', 
            'sugarcane', 'coffee', 'tea', 'potatoes', 'tomatoes'
        ]
        
        # Try to find mentioned commodities
        found_commodities = []
        for commodity in commodities:
            if re.search(r'\b' + re.escape(commodity) + r'\b', query.lower()):
                found_commodities.append(commodity)
        
        if found_commodities:
            commodity_info['commodity'] = found_commodities[0]  # Use the first found commodity
            
            # Try to extract a date
            date_patterns = [
                r'on ([0-9]{4}-[0-9]{1,2}-[0-9]{1,2})',
                r'for ([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})',
                r'at ([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})',
                r'in ([A-Z][a-z]+ [0-9]{4})'
            ]
            
            for pattern in date_patterns:
                date_match = re.search(pattern, query)
                if date_match:
                    commodity_info['date'] = date_match.group(1)
                    break
            
            # Check if trend analysis is requested
            trend_patterns = [r'\btrend\b', r'\bchange\b', r'\bmovement\b', r'\bfluctuation\b']
            commodity_info['include_trends'] = any(re.search(pattern, query.lower()) for pattern in trend_patterns)
        
        return commodity_info
    
    def _extract_soil_params(self, query):
        """
        Extract soil parameters from a query.
        
        Args:
            query (str): User query
        
        Returns:
            dict: Extracted soil parameters
        """
        soil_params = {}
        
        # Extract NPK values
        npk_pattern = r'NPK.*?(\d+)[^\d]*(\d+)[^\d]*(\d+)'
        npk_match = re.search(npk_pattern, query)
        if npk_match:
            soil_params['N'] = int(npk_match.group(1))
            soil_params['P'] = int(npk_match.group(2))
            soil_params['K'] = int(npk_match.group(3))
        
        # Extract individual values if not already found
        if 'N' not in soil_params:
            n_pattern = r'[^a-zA-Z]N[^a-zA-Z]*?(\d+)'
            n_match = re.search(n_pattern, query)
            if n_match:
                soil_params['N'] = int(n_match.group(1))
        
        if 'P' not in soil_params:
            p_pattern = r'[^a-zA-Z]P[^a-zA-Z]*?(\d+)'
            p_match = re.search(p_pattern, query)
            if p_match:
                soil_params['P'] = int(p_match.group(1))
        
        if 'K' not in soil_params:
            k_pattern = r'[^a-zA-Z]K[^a-zA-Z]*?(\d+)'
            k_match = re.search(k_pattern, query)
            if k_match:
                soil_params['K'] = int(k_match.group(1))
        
        # Extract pH value
        ph_pattern = r'pH[^0-9]*([0-9]*\.?[0-9]+)'
        ph_match = re.search(ph_pattern, query)
        if ph_match:
            soil_params['ph'] = float(ph_match.group(1))
        
        # Set default values if nothing was extracted
        if len(soil_params) == 0:
            soil_params = {
                'N': 100,
                'P': 50,
                'K': 100,
                'ph': 6.5
            }
            
        return soil_params
    
    def _get_price_information(self, query):
        """
        Get price information based on the query.
        
        Args:
            query (str): User query
        
        Returns:
            dict: Price information response
        """
        if not self.price_model:
            return {
                "response_text": "I'm sorry, the price information service is not available.",
                "confidence": 0.0,
                "source": "error",
                "intent": "price"
            }
        
        try:
            # Extract commodity information
            commodity_info = self._extract_commodity_info(query)
            
            if not commodity_info.get('commodity'):
                return {
                    "response_text": "Please specify which commodity's price you'd like to know.",
                    "confidence": 0.3,
                    "source": "error",
                    "intent": "price"
                }
            
            # Get price data
            price_data = self.price_model.get_price(
                commodity=commodity_info['commodity'],
                date=commodity_info.get('date'),
                include_trends=commodity_info.get('include_trends', False)
            )
            
            if not price_data.get('found', False):
                return {
                    "response_text": f"Sorry, I couldn't find price information for {commodity_info['commodity']}.",
                    "confidence": 0.3,
                    "source": "error",
                    "intent": "price"
                }
            
            # Format response text
            response_text = f"The price of {commodity_info['commodity'].capitalize()} is "
            response_text += f"{price_data['price']} {price_data['currency']} per {price_data['unit']} "
            response_text += f"on {price_data['date']} "
            response_text += "This information is based on real-time market data."
            
            return {
                "response_text": response_text,
                "confidence": 0.9,
                "source": "price_model",
                "intent": "price",
                "price_data": price_data
            }
            
        except Exception as e:
            logger.error(f"Error getting price information: {str(e)}")
            return {
                "response_text": "I'm sorry, I couldn't retrieve the price information. Please try again.",
                "confidence": 0.0,
                "source": "error",
                "intent": "price"
            }
    
    def _get_crop_recommendation(self, query):
        """
        Get crop recommendations based on the query.
        
        Args:
            query (str): User query
        
        Returns:
            dict: Crop recommendation response
        """
        if not self.crop_model:
            return {
                "response_text": "I'm sorry, the crop recommendation service is not available.",
                "confidence": 0.0,
                "source": "error",
                "intent": "crop"
            }
        
        try:
            # Extract soil parameters
            soil_params = self._extract_soil_params(query)
            
            if not soil_params:
                return {
                    "response_text": "Please provide soil parameters (NPK values, pH) for crop recommendations.",
                    "confidence": 0.3,
                    "source": "error",
                    "intent": "crop"
                }
            
            # Get recommendations
            recommendations = self.crop_model.predict(soil_params)
            
            if not recommendations.get('top_recommendations'):
                return {
                    "response_text": "Unable to generate crop recommendations for the provided soil parameters.",
                    "confidence": 0.3,
                    "source": "error",
                    "intent": "crop"
                }
            
            # Format response text
            top_crop = recommendations['top_recommendations'][0]
            crop_info = get_crop_description(top_crop['crop'])
            response_text = f"{top_crop['crop']} is recommended for your soil conditions with "
            response_text += f"{top_crop['confidence']:.1%} confidence. "
            response_text += f"Based on your soil's NPK values of {soil_params['N']}-{soil_params['P']}-{soil_params['K']} "
            response_text += f"and pH of {soil_params['ph']}, {top_crop['crop']} grows best in "
            response_text += f"{crop_info.get('ideal_soil', 'well-prepared soil')}."
            
            return {
                "response_text": response_text,
                "confidence": top_crop['confidence'],
                "source": "crop_model",
                "intent": "crop",
                "recommendations": recommendations
            }
            
        except Exception as e:
            logger.error(f"Error getting crop recommendations: {str(e)}")
            return {
                "response_text": "I'm sorry, I couldn't generate crop recommendations. Please try again.",
                "confidence": 0.0,
                "source": "error",
                "intent": "crop"
            }
    
    def _get_weather_information(self, query):
        """
        Get weather information based on the query.
        
        Args:
            query (str): User query
        
        Returns:
            dict: Weather information response
        """
        if not self.weather_model:
            return {
                "response_text": "I'm sorry, the weather information service is not available.",
                "confidence": 0.0,
                "source": "error",
                "intent": "weather"
            }
        
        try:
            # Extract location
            location = self.weather_model._extract_location_from_query(query)
            
            if not location:
                return {
                    "response_text": "Please specify a location for weather information.",
                    "confidence": 0.3,
                    "source": "error",
                    "intent": "weather"
                }
            
            # Get weather data
            weather_data = self.weather_model.get_weather(location)
            current_weather = weather_data.get('current', {})
            
            if not current_weather:
                return {
                    "response_text": f"I'm sorry, I couldn't retrieve the weather information for {location}.",
                    "confidence": 0.3,
                    "source": "error",
                    "intent": "weather"
                }
            
            # Ensure location is set in current weather data
            current_weather['location'] = location
            
            # Format response text using the weather model's formatter
            response_text = self.weather_model.format_weather_response(current_weather)
            
            return {
                "response_text": response_text,
                "confidence": 0.9,
                "source": "weather_model",
                "intent": "weather",
                "weather_data": weather_data
            }
            
        except Exception as e:
            logger.error(f"Error getting weather information: {str(e)}")
            return {
                "response_text": f"I'm sorry, I couldn't retrieve the weather information. Error: {str(e)}",
                "confidence": 0.0,
                "source": "error",
                "intent": "weather"
            }
    
    def _get_disease_information(self, query):
        """Get disease information from query."""
        try:
            # Extract crop name if present
            crop = None
            crop_match = re.search(r'in\s+(\w+)', query.lower())
            if crop_match:
                crop = crop_match.group(1)
            
            # Check if query contains image-related keywords
            if any(word in query.lower() for word in ['image', 'picture', 'photo']):
                return {
                    'success': False,
                    'response_text': 'Please use the image upload endpoint to identify diseases from images.',
                    'intent': 'disease',
                    'confidence': 0.8,
                    'source': 'disease_model'
                }
            
            # For text queries, provide general guidance
            response_text = (
                "To identify plant diseases accurately, please upload a clear image of the affected plant part. "
                "The system supports disease detection for the following crops: "
                f"{', '.join(sorted(set(name.split()[0] for name in self.disease_model.class_names)))}"
            )
            
            return {
                'success': True,
                'response_text': response_text,
                'intent': 'disease',
                'confidence': 0.8,
                'source': 'disease_model'
            }
            
        except Exception as e:
            logger.error(f"Error getting disease information: {str(e)}")
            return {
                'success': False,
                'response_text': 'Error processing disease query. Please try again.',
                'intent': 'disease',
                'confidence': 0.0,
                'source': 'error'
            }

    def identify_disease(self, description: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Identify disease based on description.
        
        Args:
            description (str): Disease description
            user_id (str, optional): User identifier
            
        Returns:
            Dict[str, Any]: Disease identification results
        """
        try:
            # For text queries, provide guidance to use image upload
            response = {
                'response_id': str(uuid.uuid4()),
                'response_text': (
                    "For accurate disease identification, please upload a clear image of the affected plant part. "
                    "The system uses advanced image recognition to detect diseases in: "
                    f"{', '.join(sorted(set(name.split()[0] for name in self.disease_model.class_names)))}"
                ),
                'found': True,
                'disease_info': None,
                'alternatives': None,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save to conversation history if user_id provided
            if user_id:
                self._save_to_history(user_id, description, response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in disease identification: {str(e)}")
            raise Exception(f"Error identifying disease: {str(e)}")
    
    def process_query(self, query: str, user_id: Optional[str] = None, target_lang: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a user query and return a response.
        
        Args:
            query (str): User's query
            user_id (str, optional): User identifier
            target_lang (str, optional): Target language code
            
        Returns:
            dict: Response data
        """
        try:
            # Get user preferences if available
            preferences = {}
            if user_id:
                user_data = self.user_model.get_user(user_id)
                if user_data and 'preferences' in user_data:
                    preferences = user_data['preferences']
                    
            # Detect intent
            intent = self._detect_intent(query)
            
            # First try to get a response from specialized models
            response = None
            
            # Process based on intent
            if intent == 'greeting':
                response = self._get_greeting_response()
            elif intent == 'faq':
                faq_response = self.faq_model.get_answer(query)
                response = self._format_faq_response(faq_response, query)
            elif intent == 'price':
                response = self._get_price_information(query)
            elif intent == 'crop':
                response = self._get_crop_recommendation(query)
            elif intent == 'weather':
                response = self._get_weather_information(query)
            elif intent == 'disease':
                response = self._get_disease_information(query)
            
            # If no response or low quality response, try DeepSeek
            if not response or not response.get('found', False) or self._should_use_deepseek(response):
                logger.info("Using DeepSeek model for response generation")
                response = self.deepseek_model.generate_response(query, {
                    'intent': intent,
                    'preferences': preferences,
                    'previous_responses': self._get_recent_responses(query)
                })
            
            # Add found attribute if not present
            if 'found' not in response:
                response['found'] = True
            
            # Translate the response if needed
            if target_lang and target_lang != 'en':
                if response.get('response_text'):
                    translated_text = self.language_model.translate(response['response_text'], target_lang)
                    response['response_text'] = translated_text
                    response['language'] = target_lang
            
            # Save to conversation history if user is identified
            if user_id:
                self._save_to_history(user_id, query, response)
                
            # Schedule model retraining if appropriate
            self._schedule_retraining(query, response)
                
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            error_response = {
                "response_text": "I'm sorry, but I encountered an error processing your request.",
                "confidence": 0.0,
                "source": "error",
                "intent": "error",
                "found": False,
                "error": str(e)
            }
            return error_response
        
    def _get_personalized_response(self, query: str, intent: str, preferences: Dict) -> Dict[str, Any]:
        """
        Get a personalized response based on user preferences and intent.
        
        Args:
            query (str): User's query
            intent (str): Identified intent
            preferences (dict): User preferences
            
        Returns:
            dict: Personalized response data
        """
        # Get response based on intent
        if intent == 'greeting':
            response_data = self._get_greeting_response()
        elif intent == 'price':
            response_data = self._get_price_information(query)
        elif intent == 'crop':
            response_data = self._get_crop_recommendation(query)
        elif intent == 'weather':
            response_data = self._get_weather_information(query)
        elif intent == 'disease':
            response_data = self._get_disease_information(query)
        else:
            # Try FAQ model for unknown intents
            if self.faq_model:
                faq_response = self.faq_model.get_answer(query)
                if faq_response['found']:
                    response_data = {
                        "response_text": faq_response['answer'],
                        "confidence": faq_response['confidence'],
                        "source": "faq_model",
                        "intent": intent
                    }
                else:
                    response_data = self._get_base_response(
                        "I'm not sure about that. Could you please rephrase your question?",
                        intent,
                        0.3,
                        'static'
                    )
            else:
                response_data = self._get_base_response(
                    "I'm not sure about that. Could you please rephrase your question?",
                    intent,
                    0.3,
                    'static'
                )
        
        # Add personalization based on preferences
        if preferences:
            # Add region-specific information if available
            if 'region' in preferences and preferences['region'] != 'unknown':
                response_data['region'] = preferences['region']
                
            # Add crop-specific information if user has preferred crops
            if 'crops' in preferences and preferences['crops']:
                response_data['user_crops'] = preferences['crops']
                
            # Add language preference
            if 'language' in preferences:
                response_data['language'] = preferences['language']
                
            # Add notification preferences
            if 'notifications_enabled' in preferences:
                response_data['notifications_enabled'] = preferences['notifications_enabled']
                
        return response_data
        
    def get_crop_recommendation(self, soil_params: Dict[str, float], user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get personalized crop recommendations.
        
        Args:
            soil_params (dict): Soil parameters
            user_id (str, optional): User identifier
            
        Returns:
            dict: Recommendation data
        """
        # Get base recommendations
        recommendations = self.crop_model.predict(soil_params)
        
        # Get user preferences if available
        if user_id:
            user_data = self.user_model.get_user(user_id)
            if user_data and 'preferences' in user_data:
                preferences = user_data['preferences']
                
                # Filter recommendations based on user's preferred crops
                if 'crops' in preferences and preferences['crops']:
                    preferred_crops = set(crop.lower() for crop in preferences['crops'])
                    recommendations['top_recommendations'] = [
                        rec for rec in recommendations['top_recommendations']
                        if rec['crop'].lower() in preferred_crops
                    ]
                    
                # Add region-specific information
                if 'region' in preferences and preferences['region'] != 'unknown':
                    recommendations['region'] = preferences['region']
                    
            # Update user history
            self.user_model.add_query_history(
                user_id,
                'crop_recommendations',
                {
                    'soil_params': soil_params,
                    'recommendations': recommendations
                }
            )
            
        return recommendations
        
    def get_price_info(self, commodity: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get personalized price information.
        
        Args:
            commodity (str): Commodity name
            user_id (str, optional): User identifier
            
        Returns:
            dict: Price information
        """
        # Get base price information
        price_data = self.price_model.get_price(commodity)
        
        # Get user preferences if available
        if user_id:
            user_data = self.user_model.get_user(user_id)
            if user_data and 'preferences' in user_data:
                preferences = user_data['preferences']
                
                # Add region-specific information
                if 'region' in preferences and preferences['region'] != 'unknown':
                    price_data['region'] = preferences['region']
                    
                # Check price alerts
                if 'price_alerts' in preferences:
                    price_data['alerts'] = [
                        alert for alert in preferences['price_alerts']
                        if alert['commodity'].lower() == commodity.lower() and alert['active']
                    ]
                    
            # Update user history
            self.user_model.add_query_history(
                user_id,
                'price_checks',
                {
                    'commodity': commodity,
                        'price_data': price_data
                    }
            )
            
        return price_data
        
    def get_weather_info(self, location: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get personalized weather information.
        
        Args:
            location (str): Location name
            user_id (str, optional): User identifier
            
        Returns:
            dict: Weather information
        """
        # Get base weather information
        weather_data = self.weather_model.get_weather(location)
        
        # Get user preferences if available
        if user_id:
            user_data = self.user_model.get_user(user_id)
            if user_data and 'preferences' in user_data:
                preferences = user_data['preferences']
                
                # Add region-specific information
                if 'region' in preferences and preferences['region'] != 'unknown':
                    weather_data['region'] = preferences['region']
                    
                # Check weather alerts
                if 'weather_alerts' in preferences:
                    weather_data['alerts'] = [
                        alert for alert in preferences['weather_alerts']
                        if alert['location'].lower() == location.lower() and alert['active']
                    ]
                    
            # Update user history
            self.user_model.add_query_history(
                user_id,
                'weather_checks',
                {
                    'location': location,
                    'weather_data': weather_data
                }
            )
            
        return weather_data
        
    def _save_to_history(self, user_id, query, response):
        """
        Save a query and its response to the conversation history.
        
        Args:
            user_id (str): User identifier
            query (str): User's query
            response (dict): Response data
        """
        # Initialize user's conversation history if not exists
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []
            
        # Create a history entry
        history_entry = {
            'query': query,
            'response': response,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add to conversation history
        self.conversation_history[user_id].append(history_entry)
        
        # Limit history size (optional)
        max_history = 100
        if len(self.conversation_history[user_id]) > max_history:
            self.conversation_history[user_id] = self.conversation_history[user_id][-max_history:]
            
        logger.debug(f"Added entry to conversation history for user {user_id}")

    def get_conversation_history(self, user_id, max_entries=10):
        """
        Get conversation history for a user.
        
        Args:
            user_id (str): User identifier
            max_entries (int, optional): Maximum number of entries to return
            
        Returns:
            dict: Conversation history
        """
        try:
            user_data = self.user_model.get_user(user_id)
            if not user_data or 'history' not in user_data:
                return {
                    "user_id": user_id,
                    "history": [],
                    "found": False,
                    "message": "No history found for this user"
                }
                
            # Get most recent entries
            history = user_data['history'][-max_entries:]
            
            return {
                "user_id": user_id,
                "history": history,
                "count": len(history),
                "found": True,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting conversation history: {str(e)}")
            return {
                "user_id": user_id,
                "history": [],
                "found": False,
                "error": str(e)
            }
    
    def get_available_commodities(self):
        """
        Get a list of available commodities.
        
        Returns:
            list: List of commodity names
        """
        if self.price_model:
            return self.price_model.get_available_commodities()
        return []

    def _format_faq_response(self, faq_response, query):
        """
        Format the response from the FAQ model.
        
        Args:
            faq_response (dict): Response from the FAQ model
            query (str): User query
            
        Returns:
            dict: Formatted response
        """
        if faq_response and faq_response.get('found_answer', False):
            # Get the top result
            top_result = faq_response['results'][0]
            
            return {
                "response_text": top_result['answer'],
                "confidence": top_result['score'],
                "intent": "faq",
                "source": "faq_model"
            }
        else:
            # No answer found
            return {
                "response_text": "I'm sorry, I don't have information on that topic. Could you try rephrasing your question?",
                "confidence": 0.0,
                "intent": "unknown",
                "source": "fallback"
            }

    def get_supported_languages(self) -> Dict[str, str]:
        """
        Get list of supported languages.
        
        Returns:
            dict: Dictionary of language codes and names
        """
        if self.language_model:
            return self.language_model.get_supported_languages()
        return {'en': 'English'}

    def process_voice_query(self, audio_file: str, user_id: Optional[str] = None, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a voice query from the user.
        
        Args:
            audio_file (str): Path to the audio file
            user_id (str, optional): User identifier
            language (str, optional): Language code for speech recognition
            
        Returns:
            dict: Response data including audio file path
        """
        try:
            if not self.voice_model:
                return {
                    "success": False,
                    "error": "Voice processing is not available"
                }
            
            # Convert speech to text
            success, text = self.voice_model.speech_to_text(audio_file, language or 'en')
            if not success:
                return {
                    "success": False,
                    "error": text
                }
            
            # Process the text query
            response_data = self.process_query(text, user_id, language)
            
            # Convert response to speech
            audio_response = self.voice_model.text_to_speech(
                response_data['response_text'],
                language or 'en'
            )
            
            if not audio_response:
                return {
                    "success": False,
                    "error": "Failed to generate audio response"
                }
            
            # Add audio file path to response
            response_data['audio_response'] = audio_response
            
            return response_data
            
        except Exception as e:
            logger.error(f"Error processing voice query: {str(e)}")
            return {
                "success": False,
                "error": f"Error processing voice query: {str(e)}"
            }
            
    def cleanup_voice_files(self, audio_files: List[str]) -> bool:
        """
        Clean up voice-related temporary files.
        
        Args:
            audio_files (List[str]): List of audio file paths to clean up
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.voice_model:
            return False
            
        success = True
        for audio_file in audio_files:
            if not self.voice_model.cleanup_audio_file(audio_file):
                success = False
                logger.error(f"Failed to clean up audio file: {audio_file}")
                
        return success

    def _should_use_deepseek(self, response_data: Dict[str, Any]) -> bool:
        """
        Determine if DeepSeek should be used as a fallback.
        
        Args:
            response_data (Dict[str, Any]): Response from local model
            
        Returns:
            bool: True if DeepSeek should be used
        """
        if not self.deepseek_model:
            return False
            
        # Check confidence score
        if response_data.get('confidence', 0) < 0.7:  # Increased threshold
            logger.info(f"Using DeepSeek due to low confidence: {response_data.get('confidence', 0):.2f}")
            return True
            
        # Check if response is too generic
        response_text = response_data.get('response_text', '').lower()
        generic_phrases = [
            "i'm sorry, i don't understand",
            "could you please rephrase",
            "i'm not sure about that",
            "i don't have information on that",
            "please provide more details",
            "i need more information",
            "that's not clear",
            "i'm not certain",
            "i don't know",
            "i can't help with that"
        ]
        
        if any(phrase in response_text for phrase in generic_phrases):
            logger.info("Using DeepSeek due to generic response")
            return True
            
        # Check response quality
        if len(response_text.split()) < 20:  # Too short
            logger.info("Using DeepSeek due to short response")
            return True
            
        # Check for technical terms
        technical_terms = [
            'temperature', 'humidity', 'ph', 'nutrients', 'organic',
            'sustainable', 'climate', 'season', 'rotation', 'irrigation',
            'fertilizer', 'pest', 'disease', 'harvest', 'soil'
        ]
        
        if not any(term in response_text for term in technical_terms):
            logger.info("Using DeepSeek due to lack of technical terms")
            return True
            
        return False

    def _get_deepseek_response(self, query: str, intent: str, preferences: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get response from DeepSeek API.
        
        Args:
            query (str): User's query
            intent (str): Detected intent
            preferences (Dict[str, Any]): User preferences
            
        Returns:
            Dict[str, Any]: Response data
        """
        if not self.deepseek_model:
            return {
                "response_text": "I'm sorry, advanced AI assistance is not available at the moment.",
                "confidence": 0.0,
                "source": "error",
                "intent": intent,
                "found": False
            }
            
        # Prepare context for DeepSeek
        context = {
            "intent": intent,
            "user_preferences": preferences,
            "previous_responses": self._get_recent_responses(query)
        }
        
        try:
            # Get response from DeepSeek
            response = self.deepseek_model.generate_response(query, context)
            
            # Analyze response quality
            if response.get('success', False):
                quality_score = self.deepseek_model.analyze_response_quality(response)
                
                # Format the response in the expected structure
                formatted_response = {
                    "response_text": response.get('text', 'I apologize, but I could not generate a proper response.'),
                    "confidence": quality_score,
                    "source": "deepseek_model",
                    "intent": intent,
                    "found": True,
                    "quality_score": quality_score
                }
                
                # If quality is low, mark for retraining
                if quality_score < 0.6:
                    self._schedule_retraining(query, response)
                    
                return formatted_response
            else:
                # If DeepSeek failed, return an error
                return {
                    "response_text": "I apologize, but I could not process your request properly.",
                    "confidence": 0.0,
                    "source": "error",
                    "intent": intent,
                    "found": False,
                    "error": response.get('error', 'Unknown error')
                }
        except Exception as e:
            logger.error(f"Error in DeepSeek response: {str(e)}")
            return {
                "response_text": "I apologize, but I encountered an error while processing your request.",
                "confidence": 0.0,
                "source": "error",
                "intent": intent,
                "found": False,
                "error": str(e)
            }

    def _get_recent_responses(self, current_query: str, max_responses: int = 3) -> str:
        """
        Get recent responses from conversation history.
        
        Args:
            current_query (str): Current query
            max_responses (int): Maximum number of responses to include
            
        Returns:
            str: Formatted recent responses
        """
        recent_responses = []
        
        # Get responses from all users' histories
        for user_id, history in self.conversation_history.items():
            for entry in history[-max_responses:]:
                if entry['query'] != current_query:  # Exclude current query
                    recent_responses.append(f"Q: {entry['query']}\nA: {entry['response'].get('response_text', '')}")
                    
        return "\n\n".join(recent_responses[-max_responses:])

    def _schedule_retraining(self, query: str, response: Dict[str, Any]):
        """
        Schedule model retraining based on low quality responses.
        
        Args:
            query (str): Original query
            response (Dict[str, Any]): Generated response
        """
        if not self.deepseek_model:
            return
            
        # Prepare feedback data
        feedback = {
            "quality_score": response.get('quality_score', 0),
            "reason": "Low quality response",
            "timestamp": datetime.now().isoformat()
        }
        
        # Send feedback for retraining
        self.deepseek_model.retrain_on_feedback(query, response, feedback)
        
        logger.info(f"Scheduled retraining for query: {query[:50]}...")

    def _extract_location(self, query: str) -> str:
        """
        Extract location from query string.
        
        Args:
            query (str): Query string containing location
            
        Returns:
            str: Extracted location or default location
        """
        try:
            if hasattr(self, 'weather_model'):
                return self.weather_model._extract_location_from_query(query)
            return "New Delhi, India"  # Default location
        except Exception as e:
            logger.error(f"Error extracting location: {str(e)}")
            return "New Delhi, India"  # Default location on error


def main():
    """Test the chatbot with sample queries."""
    chatbot = FarmChatbot()
    
    test_queries = [
        "Hello, how are you?",
        "What is crop rotation?",
        "What crops should I plant in sandy soil with NPK 120-60-80 and pH 6.5?",
        "Recommend crops for soil with N 150, P 30, K 180",
        "What is the current price of wheat?",
        "How much does corn cost per bushel?",
        "Show me the price trend for rice over the past week",
        "What will the weather be like tomorrow?",
        "How can I prevent soil erosion?"
    ]
    
    for query in test_queries:
        response = chatbot.process_query(query)
        print(f"\nQuery: {query}")
        print(f"Intent: {response['intent']}")
        print(f"Response: {response['response_text']}")
        print(f"Confidence: {response['confidence']}")
        print(f"Source: {response['source']}")


if __name__ == "__main__":
    main() 