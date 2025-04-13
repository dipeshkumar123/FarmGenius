import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from src.utils.file_utils import load_csv_data, save_processed_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CropProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
    
    def process_crop_data(self, data=None):
        """Process crop dataset for soil and crop recommendation models."""
        if data is None:
            logger.info("Loading crop dataset")
            data = load_csv_data('crop_data.csv')
        
        logger.info(f"Processing crop dataset with {len(data)} records")
        
        # Check for missing values
        missing_values = data.isnull().sum()
        logger.info(f"Missing values in crop data: {missing_values[missing_values > 0]}")
        
        # Fill missing values if any
        data = data.fillna(data.mean())
        
        # Create copies for different processing needs
        processed_data = data.copy()
        model_data = data.copy()
        
        # For the model data, scale features
        feature_columns = [col for col in data.columns if col != 'label']
        
        # Scale the features
        scaled_features = self.scaler.fit_transform(model_data[feature_columns])
        
        # Create a DataFrame with scaled features
        scaled_df = pd.DataFrame(scaled_features, columns=feature_columns)
        scaled_df['label'] = model_data['label']
        
        # Split data for training and testing
        features = scaled_df[feature_columns]
        labels = scaled_df['label']
        
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )
        
        # Combine into train and test DataFrames
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        
        logger.info(f"Processed crop dataset split into {len(train_df)} training and {len(test_df)} testing samples")
        
        return {
            'processed_data': processed_data,
            'train_data': train_df,
            'test_data': test_df,
            'scaler': self.scaler
        }
    
    def save_processed_data(self, processed_data_dict):
        """Save the processed crop data."""
        outputs = {}
        
        # Save the full processed dataset
        outputs['processed'] = save_processed_data(
            processed_data_dict['processed_data'], 
            'processed_crop_data.csv'
        )
        
        # Save the training dataset
        outputs['train'] = save_processed_data(
            processed_data_dict['train_data'], 
            'crop_train_data.csv'
        )
        
        # Save the testing dataset
        outputs['test'] = save_processed_data(
            processed_data_dict['test_data'], 
            'crop_test_data.csv'
        )
        
        return outputs

def main():
    """Process the crop dataset and save the results."""
    processor = CropProcessor()
    processed_data_dict = processor.process_crop_data()
    output_paths = processor.save_processed_data(processed_data_dict)
    
    logger.info("Crop data processing completed. Outputs saved to:")
    for key, path in output_paths.items():
        logger.info(f"- {key}: {path}")

if __name__ == "__main__":
    main() 