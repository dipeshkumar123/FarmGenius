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

class PriceProcessor:
    def __init__(self):
        pass
    
    def process_price_data(self, data=None):
        """Process commodity price dataset."""
        if data is None:
            logger.info("Loading commodity price dataset")
            data = load_csv_data('commodity_prices.csv')
        
        logger.info(f"Processing commodity price dataset with {len(data)} records")
        
        # Check for missing values
        missing_values = data.isnull().sum()
        logger.info(f"Missing values in price data: {missing_values[missing_values > 0]}")
        
        # Create a copy of the data
        processed_data = data.copy()
        
        # Convert prices to numeric format
        price_columns = ['min_price', 'max_price', 'modal_price']
        for col in price_columns:
            processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')
        
        # Convert date to datetime
        processed_data['date'] = pd.to_datetime(processed_data['date'], errors='coerce')
        
        # Fill missing values
        processed_data['min_price'].fillna(processed_data['modal_price'], inplace=True)
        processed_data['max_price'].fillna(processed_data['modal_price'], inplace=True)
        
        # Drop rows with missing modal prices or dates
        processed_data = processed_data.dropna(subset=['modal_price', 'date'])
        
        # Create year, month, day columns for easier filtering
        processed_data['year'] = processed_data['date'].dt.year
        processed_data['month'] = processed_data['date'].dt.month
        processed_data['day'] = processed_data['date'].dt.day
        
        # Create a commodity summary dataframe
        commodity_summary = processed_data.groupby('commodity_name').agg({
            'modal_price': ['mean', 'min', 'max', 'std'],
            'date': ['min', 'max'],
            'state': 'nunique',
            'district': 'nunique',
            'market': 'nunique'
        })
        
        # Flatten the multi-index columns
        commodity_summary.columns = ['_'.join(col).strip() for col in commodity_summary.columns.values]
        commodity_summary = commodity_summary.reset_index()
        
        # Create a state-wise commodity dataframe
        state_commodity = processed_data.groupby(['state', 'commodity_name']).agg({
            'modal_price': ['mean', 'min', 'max'],
            'date': ['min', 'max'],
            'district': 'nunique',
            'market': 'nunique'
        })
        
        # Flatten the multi-index columns
        state_commodity.columns = ['_'.join(col).strip() for col in state_commodity.columns.values]
        state_commodity = state_commodity.reset_index()
        
        # Create a recent prices dataframe
        # Assuming we want the most recent data available
        max_date = processed_data['date'].max()
        recent_prices = processed_data[processed_data['date'] > (max_date - pd.Timedelta(days=30))]
        
        logger.info(f"Processed price dataset contains {len(processed_data)} records after cleaning")
        logger.info(f"Created commodity summary with {len(commodity_summary)} commodities")
        logger.info(f"Created state-commodity summary with {len(state_commodity)} entries")
        logger.info(f"Created recent prices dataset with {len(recent_prices)} entries")
        
        return {
            'processed_data': processed_data,
            'commodity_summary': commodity_summary,
            'state_commodity': state_commodity,
            'recent_prices': recent_prices
        }
    
    def save_processed_data(self, processed_data_dict):
        """Save the processed price data."""
        outputs = {}
        
        # Save the full processed dataset
        outputs['processed'] = save_processed_data(
            processed_data_dict['processed_data'], 
            'processed_prices.csv'
        )
        
        # Save the commodity summary
        outputs['commodity_summary'] = save_processed_data(
            processed_data_dict['commodity_summary'], 
            'commodity_summary.csv'
        )
        
        # Save the state-commodity summary
        outputs['state_commodity'] = save_processed_data(
            processed_data_dict['state_commodity'], 
            'state_commodity_prices.csv'
        )
        
        # Save the recent prices
        outputs['recent_prices'] = save_processed_data(
            processed_data_dict['recent_prices'], 
            'recent_prices.csv'
        )
        
        return outputs

def main():
    """Process the commodity price dataset and save the results."""
    processor = PriceProcessor()
    processed_data_dict = processor.process_price_data()
    output_paths = processor.save_processed_data(processed_data_dict)
    
    logger.info("Commodity price data processing completed. Outputs saved to:")
    for key, path in output_paths.items():
        logger.info(f"- {key}: {path}")

if __name__ == "__main__":
    main() 