#!/usr/bin/env python
"""
Script to run the Farm Chatbot API server.
This sets up the FastAPI server to interact with the chatbot.
"""

import os
import sys
import logging
from datetime import datetime
from dotenv import load_dotenv

# Add the project root to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

# Configure logging early
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logger.info("Starting Farm Chatbot API initialization")

# Import the API app
try:
    logger.info("Importing API modules...")
    from src.api.app import start
    logger.info("API modules imported successfully")
except Exception as e:
    logger.error(f"Error importing API modules: {str(e)}", exc_info=True)
    print(f"ERROR: Failed to import API modules: {str(e)}")
    sys.exit(1)

if __name__ == "__main__":
    print("=" * 80)
    print(f"Farm Chatbot - API Server")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Display information about API integration
    use_live_weather = os.getenv("USE_LIVE_WEATHER_DATA", "False").lower() == "true"
    use_live_price = os.getenv("USE_LIVE_PRICE_DATA", "False").lower() == "true"
    use_live_disease = os.getenv("USE_LIVE_DISEASE_DATA", "False").lower() == "true"
    
    print("\nAPI Integration Status:")
    print(f"- Weather API: {'Enabled' if use_live_weather else 'Disabled'}")
    print(f"- Price API: {'Enabled' if use_live_price else 'Disabled'}")
    print(f"- Disease API: {'Enabled' if use_live_disease else 'Disabled'}")
    
    # Show API connection information
    host = os.getenv("API_HOST", "localhost")
    port = int(os.getenv("API_PORT", 8000))
    print(f"\nAPI will be available at: http://{host}:{port}")
    print(f"Swagger documentation: http://{host}:{port}/docs")
    print("=" * 80)

    try:
        # Start the API server
        logger.info("Starting the API server...")
        start()
    except KeyboardInterrupt:
        logger.info("Server shutdown requested by user")
        print("\nShutting down server...")
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}", exc_info=True)
        print(f"\nError starting server: {str(e)}")

    print("\n" + "=" * 80) 