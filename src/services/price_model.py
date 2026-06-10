import os
import logging
import pandas as pd
import json
from datetime import datetime, timedelta
import random
import requests
from typing import Dict, List, Any, Optional, Tuple
from dotenv import load_dotenv
import numpy as np
import time

from src.utils.file_utils import load_csv_data, get_project_root, ensure_directory_exists, save_processed_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CommodityPriceModel:
    """
    Model to provide agricultural commodity price information.
    This can use both synthetic data and real-time API data.
    """

    def __init__(self, use_live_data: bool = True, api_key: Optional[str] = None, api_url: Optional[str] = None):
        """
        Initialize the commodity price model.

        Args:
            use_live_data (bool): Whether to use live API data when available
            api_key (str, optional): API key for price data service
            api_url (str, optional): URL for the price data service API
        """
        # Load environment variables if not explicitly provided
        load_dotenv()
        
        # Set up API configuration
        self.use_live_data = use_live_data
        self.api_key = api_key or os.getenv("PRICE_API_KEY")
        self.api_url = api_url or "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
        
        # Log the configuration
        if self.use_live_data and self.api_key:
            logger.info(f"Price model initialized with live API access: {self.api_url}")
        else:
            logger.info("Price model initialized with synthetic data only")

        # Cache for price data
        self.cache = {}
        self.cache_timeout = 3600  # 1 hour
        self.cache_timestamp = None
        self.cache_expiry = {}
        self.cache_duration = timedelta(hours=1)

        # Default commodity list
        self.commodities = [
            'wheat', 'rice', 'corn', 'soybeans', 'cotton',
            'sugarcane', 'coffee', 'tea', 'potatoes', 'tomatoes',
            'onions', 'apples', 'bananas', 'oranges', 'milk',
            'cattle', 'chicken', 'eggs', 'wool', 'rubber'
        ]
        
        # Mapping of commodity names to API commodity names
        self.commodity_mapping = {
            'wheat': 'WHEAT',
            'rice': 'RICE',
            'corn': 'MAIZE',
            'maize': 'MAIZE', 
            'soybeans': 'SOYABEAN',
            'soybean': 'SOYABEAN',
            'potatoes': 'POTATO',
            'potato': 'POTATO',
            'tomatoes': 'TOMATO',
            'tomato': 'TOMATO',
            'onions': 'ONION',
            'onion': 'ONION',
            'garlic': 'GARLIC',
            'cotton': 'COTTON',
            'coconut': 'COCONUT',
            'banana': 'BANANA'
            # Add more mappings as needed
        }
        
        # Initialize synthetic price data for fallback
        self._init_synthetic_data()

    def _init_synthetic_data(self):
        """Initialize synthetic price data for demo purposes."""
        self.synthetic_prices = {}
        
        # Generate random price data for the commodities
        base_prices = {
            'wheat': 250,     # per quintal
            'rice': 1800,     # per quintal
            'corn': 1400,     # per quintal
            'soybeans': 3600, # per quintal
            'cotton': 5500,   # per quintal
            'sugarcane': 280, # per quintal
            'coffee': 350,    # per kg
            'tea': 250,       # per kg
            'potatoes': 20,   # per kg
            'tomatoes': 25,   # per kg
            'onions': 22,     # per kg
            'apples': 90,     # per kg
            'bananas': 40,    # per dozen
            'oranges': 70,    # per kg
            'milk': 45,       # per liter
            'cattle': 35000,  # per animal
            'chicken': 200,   # per kg
            'eggs': 5,        # per egg
            'wool': 500,      # per kg
            'rubber': 150     # per kg
        }
        
        # Units for each commodity
        self.commodity_units = {
            'wheat': 'quintal',
            'rice': 'quintal',
            'corn': 'quintal',
            'soybeans': 'quintal',
            'cotton': 'quintal',
            'sugarcane': 'quintal',
            'coffee': 'kg',
            'tea': 'kg',
            'potatoes': 'kg',
            'tomatoes': 'kg',
            'onions': 'kg',
            'apples': 'kg',
            'bananas': 'dozen',
            'oranges': 'kg',
            'milk': 'liter',
            'cattle': 'animal',
            'chicken': 'kg',
            'eggs': 'unit',
            'wool': 'kg',
            'rubber': 'kg'
        }
        
        # Generate synthetic time series (last 30 days) for each commodity
        today = datetime.now()
        for commodity, base_price in base_prices.items():
            self.synthetic_prices[commodity] = {}
            
            # Variance factors
            daily_variance = 0.02  # 2% daily price movement
            trend_factor = random.uniform(-0.1, 0.1)  # -10% to +10% monthly trend
            
            # Generate daily prices for last 30 days
            for day in range(30):
                date = today - timedelta(days=day)
                date_str = date.strftime('%Y-%m-%d')
                
                # Calculate price with trend and random movement
                days_factor = day / 30.0  # 0.0 to 1.0
                trend_adjustment = 1.0 + (trend_factor * days_factor)
                daily_adjustment = 1.0 + random.uniform(-daily_variance, daily_variance)
                
                # Final price (future dates have higher trend influence)
                price = base_price * trend_adjustment * daily_adjustment
                
                # Store in synthetic data
                self.synthetic_prices[commodity][date_str] = round(price, 2)

    def _try_fetch_live_data(self, commodity, date=None):
        """
        Try to fetch live price data from the data.gov.in API.
        
        Args:
            commodity (str): Commodity name
            date (str, optional): Date for historical price
            
        Returns:
            dict: Price data or None if fetch failed
        """
        if not self.use_live_data or not self.api_key:
            return None
            
        try:
            # Check cache first (only for today's price)
            if date is None or date == datetime.now().strftime('%Y-%m-%d'):
                if commodity in self.cache:
                    cache_age = datetime.now() - self.cache_timestamp
                    if cache_age < self.cache_duration:
                        logger.info(f"Returning cached price data for {commodity}")
                        return self.cache[commodity]
            
            # Map the commodity name to the API's expected format
            api_commodity = self.commodity_mapping.get(commodity.lower(), commodity)
            
            # Build the API request
            api_url = self.api_url
            
            # Set up query parameters based on the data.gov.in API format
            params = {
                "api-key": self.api_key,
                "format": "json",
                "limit": 10,  # Limit the number of results
                "filters[commodity]": api_commodity
            }
            
            # Add date filter if specified
            if date:
                params["filters[arrival_date]"] = date
                
            # Make the API request
            logger.info(f"Fetching price data from API for {api_commodity}")
            response = requests.get(api_url, params=params)
            
            # Log the URL for debugging
            logger.info(f"API request URL: {response.url}")
            
            # Check if request was successful
            if response.status_code == 200:
                data = response.json()
                
                # Parse the API response
                price_data = self._parse_api_response(data, commodity)
                
                # Cache the result if it's for today
                if date is None or date == datetime.now().strftime('%Y-%m-%d'):
                    self.cache[commodity] = price_data
                    self.cache_timestamp = datetime.now()
                    
                return price_data
            else:
                logger.warning(f"API request failed with status {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching price data: {str(e)}")
            return None
            
    def _parse_api_response(self, api_data, commodity):
        """
        Parse the API response to extract price information.
        
        Args:
            api_data (dict): Raw API response
            commodity (str): Requested commodity
            
        Returns:
            dict: Extracted price data or None if no valid data
        """
        try:
            if not api_data or 'records' not in api_data:
                logger.warning(f"No valid data structure in API response for {commodity}")
                return None
                
            records = api_data.get('records', [])
            
            if not records or len(records) == 0:
                logger.warning(f"No price records found in API response for {commodity}")
                return None
                
            # Get the most recent record
            latest_record = records[0]
            
            # Try different field names that might exist in the API response
            possible_price_fields = ['modal_price', 'min_price', 'max_price', 'price', 'retail_price']
            price_value = None
            
            for field in possible_price_fields:
                if field in latest_record and latest_record[field]:
                    try:
                        price_value = float(latest_record[field])
                        break
                    except (ValueError, TypeError):
                        continue
            
            if not price_value:
                logger.warning(f"Could not extract price value from API response for {commodity}")
                return None
                
            # Get the date (try different possible fields)
            date_value = None
            date_fields = ['arrival_date', 'date', 'recorded_date']
            
            for field in date_fields:
                if field in latest_record and latest_record[field]:
                    date_value = latest_record[field]
                    break
            
            # Extract other useful fields
            market = latest_record.get('market', 'Unknown')
            state = latest_record.get('state', 'Unknown')
            unit = latest_record.get('unit', 'quintal')
            
            return {
                'price': price_value,
                'date': date_value or datetime.now().strftime('%Y-%m-%d'),
                'market': market,
                'state': state,
                'unit': unit,
                'currency': 'INR',
                'raw_data': latest_record
            }
            
        except Exception as e:
            logger.error(f"Error parsing API response for {commodity}: {str(e)}")
            return None

    def get_price(self, commodity: str, date: Optional[str] = None, include_trends: bool = False) -> Dict[str, Any]:
        """
        Get the price for a commodity.
        
        Args:
            commodity (str): Name of the commodity
            date (str, optional): Date for historical price (YYYY-MM-DD)
            include_trends (bool): Whether to include trend data
            
        Returns:
            dict: Price information including:
                - price: Current price
                - unit: Unit of measurement
                - currency: Currency (default: INR)
                - date: Price date
                - found: Whether price was found
                - message: Status message
                - trend_data: Optional trend information
        """
        try:
            # Normalize commodity name
            commodity = commodity.lower()
            
            # Try to get live price data first
            price_data = self._try_fetch_live_data(commodity, date)
            
            # If live data not available, use synthetic data
            if not price_data:
                logger.info(f"Using synthetic data for {commodity}")
                price_data = self._generate_synthetic_price_data(commodity, date)
            
            # Get trend data if requested
            trend_data = None
            if include_trends:
                if date:
                    end_date = datetime.strptime(date, '%Y-%m-%d')
                else:
                    end_date = datetime.now()
                start_date = end_date - timedelta(days=30)
                trend_data = self._get_trend_data(commodity, start_date.strftime('%Y-%m-%d'))
            
            # Format the response
            response = {
                'found': True if price_data else False,
                'commodity': commodity,
                'price': price_data.get('price') if price_data else None,
                'unit': price_data.get('unit', self.commodity_units.get(commodity, 'unit')),
                'currency': price_data.get('currency', 'INR'),
                'date': price_data.get('date', datetime.now().strftime('%Y-%m-%d')),
                'message': 'Price data found' if price_data else 'Price data not available',
                'trend_data': trend_data if include_trends else None
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error getting price for {commodity}: {str(e)}")
            return {
                'found': False,
                'commodity': commodity,
                'price': None,
                'unit': self.commodity_units.get(commodity, 'unit'),
                'currency': 'INR',
                'date': date or datetime.now().strftime('%Y-%m-%d'),
                'message': f'Error getting price data: {str(e)}',
                'trend_data': None
            }

    def _get_trend_data(self, commodity, date=None):
        """Get price trend data."""
        try:
            # Set default dates if None
            if date is None:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)
            else:
                try:
                    end_date = datetime.strptime(date, "%Y-%m-%d")
                    start_date = end_date - timedelta(days=30)
                except ValueError:
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=30)
            
            # Format dates for API
            start_date_str = start_date.strftime("%Y-%m-%d")
            end_date_str = end_date.strftime("%Y-%m-%d")
            
            if self.use_live_data:
                logger.info(f"Fetching trend data from API for {commodity} from {start_date_str} to {end_date_str}")
                return self._fetch_live_trend_data(commodity, start_date_str, end_date_str)
            else:
                return self._generate_synthetic_trends(commodity, start_date_str, end_date_str)
                
        except Exception as e:
            logger.error(f"Error fetching trend data: {str(e)}")
            return self._generate_synthetic_trends(commodity, start_date_str, end_date_str)
            
    def _fetch_live_price_data(self, commodity, date=None):
        """Fetch price data from the API."""
        try:
            params = {
                'api-key': self.api_key,
                'format': 'json',
                'limit': 10,
                'filters[commodity]': commodity.upper()
            }
            
            if date:
                params['filters[arrival_date]'] = date
                
            response = requests.get(self.api_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if not data.get('records'):
                return {
                    'found': False,
                    'message': f"No price data found for {commodity}"
                }
                
            # Get the most recent record
            record = data['records'][0]
            return {
                'found': True,
                'price': float(record.get('modal_price', 0)),
                'unit': 'kg',
                'currency': 'INR',
                'date': record.get('arrival_date'),
                'message': None
            }
            
        except requests.exceptions.Timeout:
            logger.error("API request timed out")
            return self._generate_synthetic_price_data(commodity, date)
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching live price data: {str(e)}")
            return self._generate_synthetic_price_data(commodity, date)
            
    def _generate_synthetic_price_data(self, commodity, date=None):
        """Generate synthetic price data for testing."""
        try:
            # Generate a random price between 1000 and 5000
            price = random.uniform(1000, 5000)
            
            # Use current date if not specified
            if not date:
                date = datetime.now().strftime("%Y-%m-%d")
                
            return {
                'found': True,
                'price': price,
                'unit': 'kg',
                'currency': 'INR',
                'date': date,
                'message': "Using synthetic data"
            }
        except Exception as e:
            logger.error(f"Error generating synthetic price data: {str(e)}")
            return {
                'found': False,
                'message': f"Error generating price data: {str(e)}",
                'price': None,
                'unit': None,
                'currency': None,
                'date': None
            }

    def get_price_trend(self, commodity: str, duration: int = 30) -> Dict[str, Any]:
        """
        Get price trends for a commodity over a specified duration.
        
        Args:
            commodity (str): Commodity name
            duration (int): Duration in days
            
        Returns:
            dict: Price trend data with found attribute
        """
        try:
            logger.info(f"Attempting to get price trend for {commodity} over {duration} days")
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=duration)
            
            # Format dates
            start_date_str = start_date.strftime("%Y-%m-%d")
            end_date_str = end_date.strftime("%Y-%m-%d")
            
            # Try to get live data first
            if self.use_live_data:
                trend_data = self._get_live_trends(commodity, start_date_str, end_date_str)
                if trend_data and trend_data.get("prices"):
                    # Calculate trend statistics
                    statistics = self._calculate_trend_statistics(trend_data["prices"])
                    
                    return {
                        "commodity": commodity,
                        "start_date": start_date_str,
                        "end_date": end_date_str,
                        "prices": trend_data["prices"],
                        "statistics": statistics,
                        "days": duration,
                        "data_source": "live",
                        "found": True
                    }
            
            # Fall back to synthetic data
            trend_data = self._get_synthetic_trends(commodity, start_date_str, end_date_str)
            if trend_data and trend_data.get("prices"):
                # Calculate trend statistics
                statistics = self._calculate_trend_statistics(trend_data["prices"])
                
                return {
                    "commodity": commodity,
                    "start_date": start_date_str,
                    "end_date": end_date_str,
                    "prices": trend_data["prices"],
                    "statistics": statistics,
                    "days": duration,
                    "data_source": "synthetic",
                    "found": True
                }
                
            # No data available
            return {
                "commodity": commodity,
                "start_date": start_date_str,
                "end_date": end_date_str,
                "prices": [],
                "statistics": {},
                "days": duration,
                "data_source": "none",
                "found": False,
                "message": f"No price trend data available for {commodity}"
            }
            
        except Exception as e:
            logger.error(f"Error getting price trend for {commodity}: {str(e)}")
            return {
                "commodity": commodity,
                "found": False,
                "error": str(e),
                "message": f"Error retrieving price trend data for {commodity}"
            }
            
    def _calculate_trend_statistics(self, prices: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate statistics for price trend data.
        
        Args:
            prices (List[Dict]): List of price data points
            
        Returns:
            dict: Statistics about the price trend
        """
        if not prices:
            return {}
            
        # Extract price values
        price_values = [p["price"] for p in prices]
        
        # Calculate statistics
        mean_price = sum(price_values) / len(price_values)
        min_price = min(price_values)
        max_price = max(price_values)
        
        # Calculate trend (first and last prices)
        first_price = price_values[0] if len(price_values) > 0 else 0
        last_price = price_values[-1] if len(price_values) > 0 else 0
        
        price_change = last_price - first_price
        percent_change = (price_change / first_price) * 100 if first_price != 0 else 0
        
        # Calculate volatility (standard deviation)
        variance = sum((p - mean_price) ** 2 for p in price_values) / len(price_values)
        std_dev = variance ** 0.5
        
        # Determine trend direction
        if percent_change > 5:
            trend = "strong_up"
        elif percent_change > 1:
            trend = "up"
        elif percent_change < -5:
            trend = "strong_down"
        elif percent_change < -1:
            trend = "down"
        else:
            trend = "stable"
        
        return {
            "mean": mean_price,
            "min": min_price,
            "max": max_price,
            "price_change": price_change,
            "percent_change": percent_change,
            "volatility": std_dev,
            "trend": trend
        }

    def get_price_trends(self, commodity: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Get price trends for a commodity over a date range.
        
        Args:
            commodity (str): The commodity to get trends for
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            
        Returns:
            dict: Price trend data and analysis
        """
        commodity = commodity.lower()
        
        # Try to get live trend data first if configured
        if self.use_live_data:
            logger.info(f"Fetching trend data from API for {commodity.upper()} from {start_date} to {end_date}")
            trend_data = self._get_live_trends(commodity, start_date, end_date)
            if trend_data and 'prices' in trend_data and trend_data['prices']:
                logger.info(f"Successfully fetched live trend data for {commodity}")
                return trend_data
            logger.warning(f"No live trend data found for {commodity}, falling back to synthetic data")
                
        # Fall back to synthetic data
        logger.info(f"Generating synthetic trend data for {commodity}")
        return self._get_synthetic_trends(commodity, start_date, end_date)

    def analyze_price_trends(self, price_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze price trends and provide insights.
        
        Args:
            price_data (Dict[str, Any]): Price trend data
            
        Returns:
            Dict[str, Any]: Trend analysis
        """
        try:
            # Convert price data to DataFrame
            df = pd.DataFrame(price_data["prices"])
            
            # Calculate basic statistics
            stats = {
                "mean": float(df["price"].mean()),
                "std": float(df["price"].std()),
                "min": float(df["price"].min()),
                "max": float(df["price"].max()),
                "current": float(df["price"].iloc[-1])
            }
            
            # Calculate trend
            prices = df["price"].values
            x = np.arange(len(prices))
            slope, intercept = np.polyfit(x, prices, 1)
            trend = "increasing" if slope > 0 else "decreasing"
            
            # Calculate volatility
            returns = df["price"].pct_change().dropna()
            volatility = float(returns.std() * np.sqrt(252))  # Annualized volatility
            
            # Identify patterns
            patterns = self._identify_patterns(df)
            
            return {
                "statistics": stats,
                "trend": trend,
                "slope": float(slope),
                "volatility": volatility,
                "patterns": patterns
            }
            
        except Exception as e:
            logger.error(f"Error analyzing price trends: {str(e)}")
            return {
                "error": "Failed to analyze price trends",
                "details": str(e)
            }

    def _get_live_price(self, commodity: str, date: Optional[str] = None) -> Dict[str, Any]:
        """Get live price data from API."""
        try:
            logger.info(f"Attempting to fetch live data for {commodity}")
            logger.info(f"Fetching price data from API for {commodity.upper()}")
            
            # Prepare API request
            params = {
                "api-key": self.api_key,
                "format": "json",
                "limit": 10,
                "filters[commodity]": commodity.upper()
            }
            
            if date:
                params["filters[arrival_date]"] = date
                
            # Make API request
            response = requests.get(self.api_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if not data.get("records"):
                logger.warning(f"No price records found in API response for {commodity}")
                return self._get_synthetic_price(commodity, date)
                
            # Process response
            latest_record = data["records"][0]
            price_data = {
                "commodity": commodity,
                "price": float(latest_record.get("modal_price", 0)),
                "unit": latest_record.get("unit", "quintal"),
                "currency": "INR",
                "date": latest_record.get("arrival_date", datetime.now().strftime("%d/%m/%Y")),
                "found": True,
                "message": None,
                "trend_data": None
            }
            
            # Cache the result
            cache_key = f"{commodity}_{date}" if date else commodity
            self._cache_price_data(cache_key, price_data)
            
            logger.info(f"Successfully fetched live price data for {commodity}")
            return price_data
            
        except Exception as e:
            logger.error(f"Error fetching live price data: {str(e)}")
            return self._get_synthetic_price(commodity, date)
            
    def _get_live_trends(self, commodity: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """Get live price trend data from API."""
        try:
            # Handle None values for dates
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
                logger.info(f"No start_date provided, using default: {start_date}")
                
            if end_date is None:
                end_date = datetime.now().strftime("%Y-%m-%d")
                logger.info(f"No end_date provided, using default: {end_date}")
                
            logger.info(f"Fetching trend data from API for {commodity.upper()} from {start_date} to {end_date}")
            
            # Prepare API request
            params = {
                "api-key": self.api_key,
                "format": "json",
                "limit": 100,
                "filters[commodity]": commodity.upper()
            }
            
            # Only add date filter if both dates are provided
            if start_date and end_date:
                # Format dates as required by the API
                start_date_formatted = datetime.strptime(start_date, "%Y-%m-%d").strftime("%d/%m/%Y")
                end_date_formatted = datetime.strptime(end_date, "%Y-%m-%d").strftime("%d/%m/%Y")
                params["filters[arrival_date]"] = f"{start_date_formatted},{end_date_formatted}"
            
            # Make API request
            response = requests.get(self.api_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if not data.get("records"):
                logger.warning(f"No trend records found for {commodity}")
                return self._get_synthetic_trends(commodity, start_date, end_date)
                
            # Process response
            prices = []
            for record in data["records"]:
                # Convert API date format (DD/MM/YYYY) to ISO format (YYYY-MM-DD)
                api_date = record.get("arrival_date")
                if api_date:
                    try:
                        date_obj = datetime.strptime(api_date, "%d/%m/%Y")
                        formatted_date = date_obj.strftime("%Y-%m-%d")
                    except Exception:
                        formatted_date = api_date
                else:
                    formatted_date = datetime.now().strftime("%Y-%m-%d")
                
                prices.append({
                    "date": formatted_date,
                    "price": float(record.get("modal_price", 0)),
                    "unit": record.get("unit", "quintal"),
                    "currency": "INR"
                })
                
            # Sort prices by date
            prices.sort(key=lambda x: x["date"])
                
            return {
                "commodity": commodity,
                "prices": prices,
                "start_date": start_date,
                "end_date": end_date,
                "data_source": "live"
            }
            
        except Exception as e:
            logger.error(f"Error fetching live trend data: {str(e)}")
            return self._get_synthetic_trends(commodity, start_date, end_date)
            
    def _get_synthetic_price(self, commodity: str, date: Optional[str] = None) -> Dict[str, Any]:
        """Generate synthetic price data for testing."""
        import random
        
        base_prices = {
            "wheat": 2500,
            "rice": 3000,
            "corn": 2000,
            "soybeans": 4000,
            "cotton": 6000
        }
        
        base_price = base_prices.get(commodity.lower(), 3000)
        price = base_price + random.uniform(-100, 100)
        
        return {
            "commodity": commodity,
            "price": price,
            "unit": "quintal",
            "currency": "INR",
            "date": date or datetime.now().strftime("%d/%m/%Y"),
            "found": True,
            "message": None,
            "trend_data": None,
            "data_source": "synthetic"
        }
        
    def _get_synthetic_trends(self, commodity, start_date, end_date):
        """Generate synthetic price trend data for testing."""
        try:
            # Handle None values for dates
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
                logger.info(f"No start_date provided, using default: {start_date}")
                
            if end_date is None:
                end_date = datetime.now().strftime("%Y-%m-%d")
                logger.info(f"No end_date provided, using default: {end_date}")
            
            # Parse dates
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
            
            # Ensure end date is not before start date
            if end < start:
                end, start = start, end
            
            # Generate dates
            dates = []
            current = start
            while current <= end:
                dates.append(current)
                current += timedelta(days=1)
            
            # Generate prices
            base_prices = {
                "wheat": 2500,
                "rice": 3000,
                "corn": 2000,
                "soybeans": 4000,
                "cotton": 6000
            }
            
            base_price = base_prices.get(commodity.lower(), 3000)
            prices = []
            
            for date in dates:
                # Add some random variation to the price
                price = base_price + random.uniform(-100, 100)
                prices.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "price": round(price, 2),
                    "unit": "quintal",
                    "currency": "INR"
                })
            
            return {
                "commodity": commodity,
                "prices": prices,
                "start_date": start_date,
                "end_date": end_date,
                "data_source": "synthetic",
                "found": True
            }
            
        except ValueError as e:
            logger.error(f"Invalid date format in synthetic trends: {str(e)}")
            return {
                "commodity": commodity,
                "prices": [],
                "start_date": start_date,
                "end_date": end_date,
                "data_source": "synthetic",
                "found": False,
                "error": f"Invalid date format: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Error generating synthetic trends: {str(e)}")
            return {
                "commodity": commodity,
                "prices": [],
                "start_date": start_date,
                "end_date": end_date,
                "data_source": "synthetic",
                "found": False,
                "error": str(e)
            }
        
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self.cache_expiry:
            return False
        return datetime.now() < self.cache_expiry[cache_key]
        
    def _cache_price_data(self, cache_key: str, data: Dict[str, Any]):
        """Cache price data with expiry."""
        self.cache[cache_key] = data
        self.cache_expiry[cache_key] = datetime.now() + self.cache_duration
        
    def _identify_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify price patterns in the data."""
        patterns = []
        
        # Calculate moving averages
        df["MA7"] = df["price"].rolling(window=7).mean()
        df["MA30"] = df["price"].rolling(window=30).mean()
        
        # Identify crossovers
        for i in range(1, len(df)):
            if df["MA7"].iloc[i-1] < df["MA30"].iloc[i-1] and df["MA7"].iloc[i] > df["MA30"].iloc[i]:
                patterns.append({
                    "type": "bullish_crossover",
                    "date": df.index[i],
                    "price": float(df["price"].iloc[i])
                })
            elif df["MA7"].iloc[i-1] > df["MA30"].iloc[i-1] and df["MA7"].iloc[i] < df["MA30"].iloc[i]:
                patterns.append({
                    "type": "bearish_crossover",
                    "date": df.index[i],
                    "price": float(df["price"].iloc[i])
                })
                
        # Identify support/resistance levels
        price_levels = df["price"].unique()
        for level in price_levels:
            touches = len(df[df["price"].between(level * 0.99, level * 1.01)])
            if touches >= 3:
                patterns.append({
                    "type": "support_resistance",
                    "level": float(level),
                    "touches": touches
                })
                
        return patterns

    def get_available_commodities(self):
        """
        Get a list of available commodities.

        Returns:
            list: Available commodities
        """
        return self.commodities
    
    def format_price_response(self, price_data, include_trends=False):
        """
        Format price data into a natural language response.
        
        Args:
            price_data (dict): Price data from get_price
            include_trends (bool): Whether to include trend information
        
        Returns:
            str: Formatted response
        """
        if not price_data or not price_data.get("found", False):
            return f"I'm sorry, I couldn't find price information for {price_data.get('commodity', 'the requested commodity')}."
        
        commodity = price_data["commodity"].capitalize()
        price = price_data["price"]
        unit = price_data["unit"]
        currency = price_data["currency"]
        date = price_data.get("date", "")
        
        # Basic price info
        response = f"The price of {commodity} is {price} {currency} per {unit}"
        
        # Add market and state information if available (for live API data)
        if price_data.get("data_source") == "live":
            market = price_data.get("market")
            state = price_data.get("state")
            
            if market and state:
                response += f" in {market}, {state}"
            elif market:
                response += f" in {market}"
            elif state:
                response += f" in {state}"
        
        # Add date information
        if date:
            if price_data.get("is_latest", False):
                response += f" (as of {date}, the most recent data available)"
            elif price_data.get("is_exact_date", True):
                response += f" on {date}"
            else:
                response += f" (closest available date to your request: {date})"
        
        # Add min and max price if available
        if price_data.get("min_price") and price_data.get("max_price"):
            min_price = price_data["min_price"]
            max_price = price_data["max_price"]
            response += f". Price range: {min_price} - {max_price} {currency} per {unit}"
        
        # Add trend information if requested
        if include_trends:
            trend_data = self.get_price_trends(price_data["commodity"], price_data["start_date"], price_data["end_date"])
            
            if trend_data.get("found", False):
                stats = trend_data.get("statistics", {})
                percent_change = stats.get("percent_change", 0)
                
                if percent_change > 0:
                    trend_text = f" The price has increased by {abs(percent_change):.1f}% over the past {trend_data.get('days', 7)} days."
                elif percent_change < 0:
                    trend_text = f" The price has decreased by {abs(percent_change):.1f}% over the past {trend_data.get('days', 7)} days."
                else:
                    trend_text = f" The price has remained stable over the past {trend_data.get('days', 7)} days."
                
                response += f".{trend_text}"
        
        # Add data source info
        if price_data.get("data_source") == "live":
            response += " This information is based on real-time market data."
        else:
            response += " This information is based on synthetic data as real-time data is unavailable."
            
        return response


def main():
    """Test the commodity price model."""
    # Initialize the model
    use_live_data = os.getenv("USE_LIVE_PRICE_DATA", "True").lower() == "true"
    api_key = os.getenv("PRICE_API_KEY")
    api_url = os.getenv("PRICE_API_URL")
    
    model = CommodityPriceModel(
        use_live_data=use_live_data,
        api_key=api_key,
        api_url=api_url
    )
    
    print(f"Testing commodity price model (use_live_data={use_live_data})")
    print("Available commodities:")
    print(", ".join(model.get_available_commodities()))
    
    # Test with some commodities
    test_commodities = ['wheat', 'rice', 'potatoes', 'tomatoes', 'onions']
    
    for commodity in test_commodities:
        print(f"\nTesting price lookup for {commodity}:")
        price_data = model.get_price(commodity)
        print(f"Price data: {json.dumps(price_data, indent=2)}")
        
        # Format as natural language
        response = model.format_price_response(price_data, include_trends=True)
        print(f"Natural language response: {response}")
    
    # Test trend analysis
    print("\nTesting trend analysis for wheat:")
    trend_data = model.get_price_trends('wheat', '2023-01-01', '2023-01-31')
    print(f"Trend data: {json.dumps(trend_data, indent=2)}")


if __name__ == "__main__":
    main() 