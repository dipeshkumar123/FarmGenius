import os
import logging
import time
from datetime import datetime

# Local imports
from src.data_processing.faq_processor import FAQProcessor
from src.data_processing.crop_processor import CropProcessor
from src.data_processing.price_processor import PriceProcessor
from src.data_processing.query_processor import QueryProcessor
from src.data_processing.weather_processor import WeatherProcessor
from src.utils.file_utils import get_project_root

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(get_project_root(), 'data_processing.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_processed_directory():
    """Create the processed data directory if it doesn't exist."""
    processed_dir = os.path.join(get_project_root(), 'dataset', 'processed')
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
        logger.info(f"Created processed data directory: {processed_dir}")
    return processed_dir

def process_all_datasets():
    """Process all datasets and save the results."""
    start_time = time.time()
    logger.info("Starting data processing pipeline")
    
    # Create the processed data directory
    processed_dir = setup_processed_directory()
    
    # Process FAQ dataset
    try:
        logger.info("Processing FAQ dataset")
        faq_processor = FAQProcessor()
        faq_data = faq_processor.process_faq_data()
        faq_output = faq_processor.save_processed_data(faq_data)
        logger.info(f"FAQ data processed and saved to {faq_output}")
    except Exception as e:
        logger.error(f"Error processing FAQ data: {str(e)}")
    
    # Process crop dataset
    try:
        logger.info("Processing crop dataset")
        crop_processor = CropProcessor()
        crop_data_dict = crop_processor.process_crop_data()
        crop_outputs = crop_processor.save_processed_data(crop_data_dict)
        logger.info("Crop data processed and saved to multiple files")
    except Exception as e:
        logger.error(f"Error processing crop data: {str(e)}")
    
    # Process price dataset
    try:
        logger.info("Processing commodity price dataset")
        price_processor = PriceProcessor()
        price_data_dict = price_processor.process_price_data()
        price_outputs = price_processor.save_processed_data(price_data_dict)
        logger.info("Price data processed and saved to multiple files")
    except Exception as e:
        logger.error(f"Error processing price data: {str(e)}")
    
    # Process farmer query dataset
    try:
        logger.info("Processing farmer query dataset")
        query_processor = QueryProcessor()
        query_data_dict = query_processor.process_query_data()
        query_outputs = query_processor.save_processed_data(query_data_dict)
        logger.info("Query data processed and saved to multiple files")
    except Exception as e:
        logger.error(f"Error processing query data: {str(e)}")
    
    # Process weather dataset
    try:
        logger.info("Processing weather dataset")
        weather_processor = WeatherProcessor()
        weather_data_dict = weather_processor.process_weather_data()
        weather_outputs = weather_processor.save_processed_data(weather_data_dict)
        logger.info("Weather data processed and saved to multiple files")
    except Exception as e:
        logger.error(f"Error processing weather data: {str(e)}")
    
    # Calculate and log processing time
    end_time = time.time()
    processing_time = end_time - start_time
    logger.info(f"Data processing pipeline completed in {processing_time:.2f} seconds")
    
    # Create a summary report
    summary = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'processing_time': f"{processing_time:.2f} seconds",
        'datasets_processed': []
    }
    
    if 'faq_output' in locals():
        summary['datasets_processed'].append('FAQ dataset')
    
    if 'crop_outputs' in locals():
        summary['datasets_processed'].append('Crop dataset')
    
    if 'price_outputs' in locals():
        summary['datasets_processed'].append('Price dataset')
    
    if 'query_outputs' in locals():
        summary['datasets_processed'].append('Query dataset')
    
    if 'weather_outputs' in locals():
        summary['datasets_processed'].append('Weather dataset')
    
    logger.info(f"Data processing summary: {summary}")
    
    return summary

if __name__ == "__main__":
    process_all_datasets() 