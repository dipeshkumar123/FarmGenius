import os
import logging
import pandas as pd
import json
import random
from datetime import datetime, timedelta
import requests
from typing import Dict, List, Any, Optional, Tuple
import re
from dotenv import load_dotenv

from src.utils.file_utils import get_project_root, ensure_directory_exists, save_processed_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WeatherModel:
    """
    Model to provide weather information for agricultural purposes.
    Uses synthetic data to simulate weather conditions.
    """
    
    def __init__(self):
        """Initialize the weather information model."""
        # Load environment variables
        load_dotenv()
        
        # Configure settings from environment
        self.use_live_data = os.getenv("USE_LIVE_WEATHER_DATA", "False").lower() == "true"
        self.api_key = os.getenv("WEATHER_API_KEY")
        self.api_url = os.getenv("WEATHER_API_URL")
        
        logger.info(f"Weather model initialized with live data: {self.use_live_data}")
        if self.use_live_data:
            logger.info("Using WeatherAPI.com for live weather data")
        
        # Cache for weather data
        self.weather_cache = {}
        self.cache_timestamp = None
        self.cache_expiry = timedelta(hours=1)  # Cache expires after 1 hour for weather data
        
        # Default locations for demo
        self.default_locations = [
            'New Delhi, India',
            'Mumbai, India',
            'Bangalore, India',
            'Chennai, India',
            'Hyderabad, India',
            'Kolkata, India',
            'Pune, India',
            'Jaipur, India',
            'Lucknow, India',
            'Kanpur, India'
        ]
        
        # Weather condition mappings with probabilities
        self.weather_conditions = {
            'Clear': 0.3,
            'Partly Cloudy': 0.2,
            'Cloudy': 0.2,
            'Overcast': 0.1,
            'Light Rain': 0.1,
            'Moderate Rain': 0.05,
            'Heavy Rain': 0.03,
            'Thunderstorm': 0.01,
            'Foggy': 0.01
        }
        
        # Crop-specific weather advice
        self.crop_weather_advice = {
            'wheat': {
                'ideal_temp_range': (15, 24),
                'ideal_rainfall': (450, 650),  # mm per growing season
                'humidity_preference': 'moderate',
                'sensitivity': 'Sensitive to high humidity during flowering and ripening; drought resistant.'
            },
            'rice': {
                'ideal_temp_range': (20, 35),
                'ideal_rainfall': (1250, 2000),  # mm per growing season
                'humidity_preference': 'high',
                'sensitivity': 'Requires high water availability; sensitive to drought.'
            },
            'corn': {
                'ideal_temp_range': (18, 32),
                'ideal_rainfall': (500, 800),  # mm per growing season
                'humidity_preference': 'moderate',
                'sensitivity': 'Sensitive to drought during silking and pollination.'
            },
            'soybeans': {
                'ideal_temp_range': (20, 30),
                'ideal_rainfall': (450, 700),  # mm per growing season
                'humidity_preference': 'moderate',
                'sensitivity': 'Sensitive to excess moisture and drought during flowering and pod fill.'
            },
            'potatoes': {
                'ideal_temp_range': (15, 20),
                'ideal_rainfall': (500, 700),  # mm per growing season
                'humidity_preference': 'moderate',
                'sensitivity': 'Sensitive to frost and drought; excessive heat reduces tuber development.'
            },
            'tomatoes': {
                'ideal_temp_range': (20, 27),
                'ideal_rainfall': (400, 600),  # mm per growing season
                'humidity_preference': 'low to moderate',
                'sensitivity': 'Sensitive to frost; high humidity can increase disease pressure.'
            }
        }
        
        logger.info("Weather model initialized")
    
    def _is_cache_valid(self, location, forecast=False):
        """Check if the cache is still valid for a location."""
        if not self.cache_timestamp:
            return False
        
        cache_key = self._get_cache_key(location, forecast)
        if cache_key not in self.weather_cache:
            return False
        
        time_elapsed = datetime.now() - self.cache_timestamp
        return time_elapsed < self.cache_expiry
    
    def _get_cache_key(self, location, forecast=False):
        """Generate a cache key for weather data."""
        prefix = "forecast_" if forecast else "current_"
        return f"{prefix}{location.lower().replace(' ', '_')}"
    
    def _normalize_location(self, location: str) -> str:
        """
        Normalize location names to match API requirements.
        """
        # Common location name variations
        location_variations = {
            'banglore': 'Bangalore, India',
            'bangalore': 'Bangalore, India',
            'bengaluru': 'Bangalore, India',
            'delhi': 'New Delhi, India',
            'new delhi': 'New Delhi, India',
            'mumbai': 'Mumbai, India',
            'bombay': 'Mumbai, India',
            'chennai': 'Chennai, India',
            'madras': 'Chennai, India',
            'hyderabad': 'Hyderabad, India',
            'kolkata': 'Kolkata, India',
            'calcutta': 'Kolkata, India',
            'pune': 'Pune, India',
            'jaipur': 'Jaipur, India',
            'lucknow': 'Lucknow, India',
            'kanpur': 'Kanpur, India'
        }
        
        # Convert to lowercase for comparison
        location_lower = location.lower().strip()
        
        # Check if location exists in variations
        if location_lower in location_variations:
            return location_variations[location_lower]
        
        # If location contains 'india', add it if missing
        if 'india' not in location_lower and any(city in location_lower for city in location_variations.keys()):
            return f"{location}, India"
        
        return location
    
    def _try_fetch_live_data(self, location, forecast=False):
        """
        Try to fetch live weather data from the API.
        """
        try:
            # Normalize location name
            normalized_location = self._normalize_location(location)
            
            # Construct API URL
            endpoint = "forecast.json" if forecast else "current.json"
            url = f"{self.api_url}/{endpoint}"
            
            # Prepare parameters
            params = {
                "key": self.api_key,
                "q": normalized_location,
                "aqi": "no"
            }
            
            if forecast:
                params["days"] = 7
            
            # Make API request
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"API request failed with status {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching live weather data: {str(e)}")
            return None
    
    def _parse_current_api_response(self, api_data):
        """
        Parse current weather API response to our standard format.
        
        Args:
            api_data (dict): API response data
            
        Returns:
            dict: Standardized weather data
        """
        try:
            current = api_data.get('current', {})
            location = api_data.get('location', {})
            
            # Extract the data we need
            return {
                'location': f"{location.get('name', 'Unknown')}, {location.get('country', 'Unknown')}",
                'condition': current.get('condition', {}).get('text', 'Unknown'),
                'temperature_c': current.get('temp_c', 0),
                'temperature_f': current.get('temp_f', 32),
                'humidity': current.get('humidity', 0),
                'wind_speed_kmh': current.get('wind_kph', 0),
                'wind_direction': current.get('wind_dir', 'N'),
                'precipitation_mm': current.get('precip_mm', 0),
                'data_source': 'live_api',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        except Exception as e:
            logger.error(f"Error parsing API response: {str(e)}")
            return None
    
    def _parse_forecast_api_response(self, api_data):
        """
        Parse forecast API response to our standard format.
        
        Args:
            api_data (dict): API response data
            
        Returns:
            dict: Standardized forecast data
        """
        try:
            location = api_data.get('location', {})
            forecast = api_data.get('forecast', {})
            forecast_days = forecast.get('forecastday', [])
            
            daily_forecast = []
            
            for day_data in forecast_days:
                day = day_data.get('date')
                day_obj = day_data.get('day', {})
                
                daily_forecast.append({
                    'date': day,
                    'condition': day_obj.get('condition', {}).get('text', 'Unknown'),
                    'temp_c_max': day_obj.get('maxtemp_c', 0),
                    'temp_c_min': day_obj.get('mintemp_c', 0),
                    'humidity': day_obj.get('avghumidity', 0),
                    'wind_speed_kmh': day_obj.get('maxwind_kph', 0),
                    'precipitation_chance': day_obj.get('daily_chance_of_rain', 0),
                    'precipitation_mm': day_obj.get('totalprecip_mm', 0)
                })
            
            return {
                'location': f"{location.get('name', 'Unknown')}, {location.get('country', 'Unknown')}",
                'forecast_days': daily_forecast,
                'data_source': 'live_api',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        except Exception as e:
            logger.error(f"Error parsing forecast API response: {str(e)}")
            return None
    
    def _generate_synthetic_weather(self, location: str) -> Dict[str, Any]:
        """Generate synthetic weather data with realistic patterns."""
        # Generate temperature based on time of day and season
        current_time = datetime.now()
        hour = current_time.hour
        month = current_time.month
        
        # Base temperature varies by season
        if month in [12, 1, 2]:  # Winter
            base_temp = random.uniform(10, 20)
        elif month in [3, 4, 5]:  # Spring
            base_temp = random.uniform(20, 30)
        elif month in [6, 7, 8]:  # Summer
            base_temp = random.uniform(25, 35)
        else:  # Fall
            base_temp = random.uniform(15, 25)
        
        # Temperature varies by time of day
        if 5 <= hour < 12:  # Morning
            temp_c = base_temp + random.uniform(0, 5)
        elif 12 <= hour < 17:  # Afternoon
            temp_c = base_temp + random.uniform(5, 10)
        else:  # Evening/Night
            temp_c = base_temp + random.uniform(-5, 0)
        
        # Select weather condition based on probabilities
        conditions = list(self.weather_conditions.keys())
        probabilities = list(self.weather_conditions.values())
        condition = random.choices(conditions, probabilities)[0]
        
        # Adjust humidity based on condition
        if condition in ['Light Rain', 'Moderate Rain', 'Heavy Rain', 'Thunderstorm']:
            humidity = random.randint(80, 95)
        elif condition == 'Foggy':
            humidity = random.randint(70, 85)
        else:
            humidity = random.randint(40, 70)
        
        return {
            'location': self._normalize_location(location),
            'condition': condition,
            'temperature_c': round(temp_c, 1),
            'temperature_f': round((temp_c * 9/5) + 32, 1),
            'humidity': humidity,
            'wind_speed_kmh': round(random.uniform(0, 30), 1),
            'wind_direction': random.choice(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']),
            'precipitation_mm': round(random.uniform(0, 10), 1) if 'Rain' in condition else 0,
            'data_source': 'synthetic',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

    def _generate_synthetic_forecast(self, location: str, days: int = 5) -> Dict[str, Any]:
        """Generate synthetic forecast data with realistic patterns."""
        daily_forecast = []
        current_time = datetime.now()
        
        # Generate base temperature trend
        base_temp = random.uniform(20, 30)
        temp_trend = random.uniform(-0.5, 0.5)  # Slight daily variation
        
        for i in range(days):
            date = (current_time + timedelta(days=i)).strftime('%Y-%m-%d')
            
            # Temperature varies by day with a slight trend
            daily_temp = base_temp + (temp_trend * i) + random.uniform(-2, 2)
            
            # Select weather condition based on probabilities
            conditions = list(self.weather_conditions.keys())
            probabilities = list(self.weather_conditions.values())
            condition = random.choices(conditions, probabilities)[0]
            
            # Adjust humidity based on condition
            if condition in ['Light Rain', 'Moderate Rain', 'Heavy Rain', 'Thunderstorm']:
                humidity = random.randint(80, 95)
            elif condition == 'Foggy':
                humidity = random.randint(70, 85)
            else:
                humidity = random.randint(40, 70)
            
            daily_forecast.append({
                'date': date,
                'condition': condition,
                'temp_c_max': round(daily_temp + random.uniform(2, 5), 1),
                'temp_c_min': round(daily_temp - random.uniform(2, 5), 1),
                'humidity': humidity,
                'wind_speed_kmh': round(random.uniform(0, 30), 1),
                'precipitation_chance': random.randint(0, 100) if 'Rain' in condition else random.randint(0, 30),
                'precipitation_mm': round(random.uniform(0, 15), 1) if 'Rain' in condition else 0
            })
        
        return {
            'location': self._normalize_location(location),
            'forecast_days': daily_forecast,
            'data_source': 'synthetic',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

    def get_current_weather(self, location: str) -> Dict[str, Any]:
        """
        Get current weather for a location.
        
        Args:
            location (str): Location name
            
        Returns:
            dict: Weather data
        """
        try:
            # Check cache first
            if self._is_cache_valid(location):
                return self.weather_cache[self._get_cache_key(location)]
            
            # Try to get live data if enabled
            if self.use_live_data and self.api_key:
                api_data = self._try_fetch_live_data(location)
                if api_data:
                    weather_data = self._parse_current_api_response(api_data)
                    if weather_data:
                        # Update cache
                        self.weather_cache[self._get_cache_key(location)] = weather_data
                        self.cache_timestamp = datetime.now()
                        return weather_data
            
            # Fallback to synthetic data
            weather_data = self._generate_synthetic_weather(location)
            self.weather_cache[self._get_cache_key(location)] = weather_data
            self.cache_timestamp = datetime.now()
            return weather_data
            
        except Exception as e:
            logger.error(f"Error getting weather data for {location}: {str(e)}")
            raise

    def get_weather_forecast(self, location: str, days: int = 5) -> Dict[str, Any]:
        """
        Get weather forecast for a location.
        
        Args:
            location (str): Location name
            days (int): Number of days to forecast
            
        Returns:
            dict: Forecast data
        """
        try:
            # Validate days parameter
            days = min(max(1, days), 7)  # Limit to 1-7 days
            
            # Check cache first
            cache_key = self._get_cache_key(location, forecast=True)
            if self._is_cache_valid(location, forecast=True):
                cached_data = self.weather_cache[cache_key]
                if len(cached_data['forecast_days']) >= days:
                    cached_data['forecast_days'] = cached_data['forecast_days'][:days]
                    return {'forecast_data': cached_data}
            
            # Try to get live data if enabled
            if self.use_live_data and self.api_key:
                api_data = self._try_fetch_live_data(location, forecast=True)
                if api_data:
                    forecast_data = self._parse_forecast_api_response(api_data)
                    if forecast_data:
                        # Ensure we have exactly the requested number of days
                        if len(forecast_data['forecast_days']) > days:
                            forecast_data['forecast_days'] = forecast_data['forecast_days'][:days]
                        elif len(forecast_data['forecast_days']) < days:
                            # Generate additional synthetic days if needed
                            while len(forecast_data['forecast_days']) < days:
                                last_day = forecast_data['forecast_days'][-1]
                                next_date = (datetime.strptime(last_day['date'], '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
                                forecast_data['forecast_days'].append(self._generate_synthetic_forecast(location, 1)['forecast_days'][0])
                                forecast_data['forecast_days'][-1]['date'] = next_date
                        
                        # Update cache
                        self.weather_cache[cache_key] = forecast_data
                        self.cache_timestamp = datetime.now()
                        return {'forecast_data': forecast_data}
            
            # Fallback to synthetic data
            forecast_data = self._generate_synthetic_forecast(location, days)
            self.weather_cache[cache_key] = forecast_data
            self.cache_timestamp = datetime.now()
            return {'forecast_data': forecast_data}
            
        except Exception as e:
            logger.error(f"Error getting forecast data for {location}: {str(e)}")
            raise

    def get_crop_weather_advice(self, location: str, crop: str) -> Dict[str, Any]:
        """
        Get crop-specific weather advice for a location.
        
        Args:
            location (str): Location name
            crop (str): Crop name
            
        Returns:
            dict: Weather advice data
        """
        try:
            # Normalize crop name
            crop = crop.lower().strip()
            
            # Get current weather
            weather = self.get_current_weather(location)
            
            # Get crop preferences
            crop_info = self.crop_weather_advice.get(crop)
            if not crop_info:
                return {
                    'found': False,
                    'message': f"No weather advice available for crop: {crop}",
                    'weather_data': weather
                }
            
            # Analyze conditions
            temp_c = weather['temperature_c']
            ideal_temp_min, ideal_temp_max = crop_info['ideal_temp_range']
            
            advice = []
            
            # Temperature advice
            if temp_c < ideal_temp_min:
                advice.append(f"Current temperature ({temp_c}°C) is below ideal range ({ideal_temp_min}-{ideal_temp_max}°C)")
            elif temp_c > ideal_temp_max:
                advice.append(f"Current temperature ({temp_c}°C) is above ideal range ({ideal_temp_min}-{ideal_temp_max}°C)")
            else:
                advice.append(f"Temperature ({temp_c}°C) is within ideal range")
            
            # Humidity advice
            humidity = weather['humidity']
            if crop_info['humidity_preference'] == 'high' and humidity < 70:
                advice.append("Humidity is lower than preferred. Consider irrigation or humidity management.")
            elif crop_info['humidity_preference'] == 'low to moderate' and humidity > 70:
                advice.append("Humidity is higher than preferred. Ensure good air circulation.")
            
            return {
                'found': True,
                'crop': crop,
                'weather_data': weather,
                'crop_preferences': crop_info,
                'advice': advice,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            logger.error(f"Error getting crop weather advice for {crop} at {location}: {str(e)}")
            raise
    
    def get_weather_for_crop(self, location, crop_name):
        """
        Get weather information with agricultural advice for a specific crop.
        
        Args:
            location (str): Location name
            crop_name (str): Name of the crop
        
        Returns:
            dict: Weather data with crop-specific advice
        """
        # Get current weather
        current_weather = self.get_current_weather(location)
        
        # Get forecast
        forecast = self.get_weather_forecast(location, days=7)
        
        # Get crop advice
        crop_advice = self.get_crop_weather_advice(
            location, 
            crop_name
        )
        
        return {
            'current_weather': current_weather,
            'forecast': forecast,
            'crop_advice': crop_advice
        }
    
    def format_weather_response(self, weather_data, include_forecast=False):
        """Format weather data into a readable response."""
        try:
            response = []
            
            # Add location and current conditions
            response.append(f"Weather in {weather_data.get('location', 'Unknown')}:")
            response.append(f"Current conditions: {weather_data.get('condition', 'Unknown')}")
            response.append(f"Temperature: {weather_data.get('temperature_c', 0)}°C ({weather_data.get('temperature_f', 32)}°F)")
            response.append(f"Humidity: {weather_data.get('humidity', 0)}%")
            response.append(f"Wind: {weather_data.get('wind_speed_kmh', 0)} km/h from {weather_data.get('wind_direction', 'N')}")
            
            if weather_data.get('precipitation_mm', 0) > 0:
                response.append(f"Precipitation: {weather_data.get('precipitation_mm', 0)} mm")
            
            # Add forecast if requested
            if include_forecast and 'forecast' in weather_data:
                response.append("\nForecast:")
                for day in weather_data['forecast'][:3]:  # Show next 3 days
                    response.append(f"\n{day['date']}:")
                    response.append(f"  {day['condition']}")
                    response.append(f"  Temperature: {day['temp_c_min']}°C to {day['temp_c_max']}°C")
                    if day['precipitation_chance'] > 30:
                        response.append(f"  Rain chance: {day['precipitation_chance']}%")
            
            return "\n".join(response)
            
        except Exception as e:
            logger.error(f"Error formatting weather response: {str(e)}")
            return "Sorry, I couldn't format the weather information properly."

    def format_forecast_response(self, forecast_data, days=3):
        """Format forecast data into a readable response."""
        try:
            response = []
            response.append(f"Weather forecast for {forecast_data['location']}:")
            
            for day in forecast_data['forecast_days'][:days]:
                response.append(f"\n{day['date']}:")
                response.append(f"  {day['condition']}")
                response.append(f"  Temperature: {day['temp_c_min']}°C to {day['temp_c_max']}°C")
                response.append(f"  Humidity: {day['humidity']}%")
                if day['precipitation_chance'] > 30:
                    response.append(f"  Rain chance: {day['precipitation_chance']}%")
                if day['precipitation_mm'] > 0:
                    response.append(f"  Expected rainfall: {day['precipitation_mm']} mm")
            
            return "\n".join(response)
            
        except Exception as e:
            logger.error(f"Error formatting forecast response: {str(e)}")
            return "Sorry, I couldn't format the forecast information properly."
    
    def format_crop_weather_advice(self, crop_weather_data):
        """
        Format crop weather advice into a natural language response.
        
        Args:
            crop_weather_data (dict): Crop weather data
        
        Returns:
            str: Formatted crop weather advice
        """
        current = crop_weather_data['current_weather']
        crop_advice = crop_weather_data['crop_advice']
        
        response = f"Weather advisory for {crop_advice['crop']} in {current['location']}:\n\n"
        response += f"{crop_advice['overall_recommendation']}\n\n"
        
        response += f"• {crop_advice['temperature_advice']}\n"
        response += f"• {crop_advice['humidity_advice']}\n"
        response += f"• {crop_advice['precipitation_advice']}\n"
        
        if crop_advice['extreme_conditions']:
            response += "\nWarnings:\n"
            for warning in crop_advice['extreme_conditions']:
                response += f"• {warning}\n"
        
        response += f"\nCrop sensitivity: {crop_advice['crop_sensitivity']}"
        
        return response.strip()
    
    def get_weather_from_query(self, query):
        """
        Extract information and generate a weather response from a query.
        
        Args:
            query (str): User query
        
        Returns:
            dict: Response information
        """
        # Extract location from query
        location = self._extract_location_from_query(query)
        
        # Determine if forecast is requested
        forecast_patterns = [
            r'\bforecast\b', r'\btomorrow\b', r'\bnext week\b', 
            r'\bupcoming\b', r'\bexpect\b', r'\bweek\b'
        ]
        is_forecast = any(re.search(pattern, query, re.IGNORECASE) for pattern in forecast_patterns)
        
        # Determine if crop-specific info is requested
        crop_match = None
        for crop in self.crop_weather_advice.keys():
            if re.search(r'\b' + re.escape(crop) + r'\b', query, re.IGNORECASE):
                crop_match = crop
                break
        
        # Get the appropriate weather data
        if crop_match:
            weather_data = self.get_weather_for_crop(location, crop_match)
            response_text = self.format_crop_weather_advice(weather_data)
            data_type = "crop_weather"
        elif is_forecast:
            weather_data = self.get_weather_forecast(location)
            response_text = self.format_forecast_response(weather_data)
            data_type = "forecast"
        else:
            weather_data = self.get_current_weather(location)
            response_text = self.format_weather_response(weather_data, include_forecast=False)
            data_type = "current"
        
        return {
            "response_text": response_text,
            "confidence": 0.9,
            "source": "weather_model",
            "data_type": data_type,
            "location": location,
            "weather_data": weather_data
        }

    def get_weather(self, location: str) -> Dict[str, Any]:
        """
        Get comprehensive weather information for a location.
        
        Args:
            location (str): Location name
            
        Returns:
            dict: Weather information including current conditions and forecast
        """
        # Get current weather
        current_weather = self.get_current_weather(location)
        
        # Get 5-day forecast
        forecast = self.get_weather_forecast(location, days=5)
        
        # Combine the data
        weather_data = {
            'location': location,
            'current': current_weather,
            'forecast': forecast,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_source': 'weather_model'
        }
        
        return weather_data

    def _extract_location_from_query(self, query: str) -> str:
        """
        Extract location from query string.
        
        Args:
            query (str): Query string containing location
            
        Returns:
            str: Extracted location or default location
        """
        try:
            # Common patterns for location in weather queries
            patterns = [
                r'(?:in|at|for|near)\s+([A-Za-z\s,]+?)(?:\s+(?:weather|temperature|forecast|rain|humidity)|$)',
                r'([A-Za-z\s,]+?)(?:\s+(?:weather|temperature|forecast|rain|humidity))',
                r'weather\s+(?:in|at|for|near)\s+([A-Za-z\s,]+)',
                r'weather\s+of\s+([A-Za-z\s,]+)',
                r'weather\s+(?:like|is)\s+(?:in|at|for|near)\s+([A-Za-z\s,]+)',
                r'what\'s\s+(?:the\s+)?weather\s+(?:like\s+)?(?:in|at|for|near)\s+([A-Za-z\s,]+)',
                r'how\'s\s+(?:the\s+)?weather\s+(?:like\s+)?(?:in|at|for|near)\s+([A-Za-z\s,]+)'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, query, re.IGNORECASE)
                if match:
                    location = match.group(1).strip()
                    return self._normalize_location(location)
            
            return "New Delhi, India"  # Default location
            
        except Exception as e:
            logger.error(f"Error extracting location from query: {str(e)}")
            return "New Delhi, India"  # Default location on error


def main():
    """Test the weather model."""
    model = WeatherModel()
    
    # Test current weather
    location = "Mumbai, India"
    print(f"\nCurrent weather in {location}:")
    current_weather = model.get_current_weather(location)
    print(json.dumps(current_weather, indent=2))
    
    # Test weather forecast
    print(f"\nWeather forecast for {location}:")
    forecast = model.get_weather_forecast(location, days=3)
    print(json.dumps(forecast, indent=2))
    
    # Test crop weather advice
    crop_name = "rice"
    print(f"\nWeather advice for {crop_name} in {location}:")
    crop_weather = model.get_weather_for_crop(location, crop_name)
    print(json.dumps(crop_weather['crop_advice'], indent=2))
    
    # Test query interpretation
    test_queries = [
        "What's the weather in Mumbai?",
        "Will it rain tomorrow in Delhi?",
        "Weather forecast for Bangalore next week",
        "Is the weather good for growing wheat in Pune?",
        "Will there be a thunderstorm in Chennai today?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = model.get_weather_from_query(query)
        print(f"Response: {result['response_text']}")
        print(f"Location: {result['location']}")
        print(f"Data type: {result['data_type']}")


if __name__ == "__main__":
    main() 