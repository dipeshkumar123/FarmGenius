import pandas as pd
import numpy as np
import logging
from datetime import datetime

from src.utils.file_utils import load_csv_data, save_processed_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WeatherProcessor:
    def __init__(self):
        pass
    
    def process_weather_data(self, data=None):
        """Process weather dataset."""
        if data is None:
            logger.info("Loading weather dataset")
            # Since weather data is large, we'll load with a limit or sample
            try:
                data = load_csv_data('weather_data.csv')
            except Exception as e:
                logger.error(f"Failed to load full weather dataset: {str(e)}")
                logger.info("Attempting to load with nrows parameter")
                data = load_csv_data('weather_data.csv', nrows=100000)  # Load first 100k rows
        
        logger.info(f"Processing weather dataset with {len(data)} records")
        
        # Check the structure of the data
        logger.info(f"Weather data columns: {data.columns.tolist()}")
        
        # Create a copy of the data
        processed_data = data.copy()
        
        # Basic cleaning and conversions will depend on the specific structure
        # For now, we'll implement a generic processor that can be refined later
        
        # Check for missing values
        missing_values = data.isnull().sum()
        logger.info(f"Missing values in weather data: {missing_values[missing_values > 0]}")
        
        # If there's a date column, convert to datetime
        date_columns = [col for col in data.columns if 'date' in col.lower()]
        for col in date_columns:
            processed_data[col] = pd.to_datetime(processed_data[col], errors='coerce')
        
        # If there are temperature columns, ensure they're numeric
        temp_columns = [col for col in data.columns if 'temp' in col.lower()]
        for col in temp_columns:
            processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')
        
        # If there are location columns, we might want to group by them
        location_columns = [
            col for col in data.columns 
            if any(loc in col.lower() for loc in ['city', 'state', 'district', 'location'])
        ]
        
        # If we have location and date columns, we can create aggregated datasets
        if location_columns and date_columns:
            logger.info(f"Creating location-based aggregations using columns: {location_columns}")
            
            # Try to create a location-based aggregation
            try:
                # Choose the first location column for simplicity
                location_col = location_columns[0]
                date_col = date_columns[0]
                
                # Extract year and month if we have a date column
                processed_data['year'] = processed_data[date_col].dt.year
                processed_data['month'] = processed_data[date_col].dt.month
                
                # Create monthly averages by location
                location_monthly = processed_data.groupby([location_col, 'year', 'month']).agg({
                    col: ['mean', 'min', 'max'] for col in temp_columns
                })
                
                # Flatten the multi-index columns
                location_monthly.columns = ['_'.join(col).strip() for col in location_monthly.columns.values]
                location_monthly = location_monthly.reset_index()
                
                logger.info(f"Created location monthly dataset with {len(location_monthly)} entries")
                
                return {
                    'processed_data': processed_data,
                    'location_monthly': location_monthly
                }
            except Exception as e:
                logger.error(f"Failed to create location aggregations: {str(e)}")
        
        # If we couldn't create aggregations, just return the processed data
        return {
            'processed_data': processed_data
        }
    
    def save_processed_data(self, processed_data_dict):
        """Save the processed weather data."""
        outputs = {}
        
        # Save the full processed dataset
        outputs['processed'] = save_processed_data(
            processed_data_dict['processed_data'], 
            'processed_weather.csv'
        )
        
        # Save any additional datasets
        for key, data in processed_data_dict.items():
            if key != 'processed_data':
                file_name = f'weather_{key}.csv'
                outputs[key] = save_processed_data(data, file_name)
        
        return outputs

def main():
    """Process the weather dataset and save the results."""
    processor = WeatherProcessor()
    processed_data_dict = processor.process_weather_data()
    output_paths = processor.save_processed_data(processed_data_dict)
    
    logger.info("Weather data processing completed. Outputs saved to:")
    for key, path in output_paths.items():
        logger.info(f"- {key}: {path}")

if __name__ == "__main__":
    main() 