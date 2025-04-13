import os
import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_project_root():
    """Return the project root directory."""
    # Assuming this file is in src/utils/
    current_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_path, '../..'))

def get_dataset_path(file_name, directory=None):
    """
    Return the full path to a dataset file.
    
    Args:
        file_name (str): Name of the file
        directory (str, optional): Subdirectory within the dataset directory
    
    Returns:
        str: Full path to the file
    """
    if directory:
        return os.path.join(get_project_root(), 'dataset', directory, file_name)
    return os.path.join(get_project_root(), 'dataset', file_name)

def load_csv_data(file_name, directory=None, **kwargs):
    """
    Load a CSV file as a pandas DataFrame.
    
    Args:
        file_name (str): Name of the CSV file
        directory (str, optional): Subdirectory within the dataset directory
        **kwargs: Additional arguments to pass to pd.read_csv
    
    Returns:
        DataFrame: Loaded data
    """
    file_path = get_dataset_path(file_name, directory)
    logger.info(f"Loading data from {file_path}")
    
    try:
        return pd.read_csv(file_path, **kwargs)
    except Exception as e:
        logger.error(f"Error loading {file_path}: {str(e)}")
        raise

def save_processed_data(df, file_name, directory='processed'):
    """
    Save a DataFrame to the processed data directory.
    
    Args:
        df (DataFrame): Data to save
        file_name (str): Name of the output file
        directory (str): Subdirectory within the dataset directory
    
    Returns:
        str: Path to the saved file
    """
    processed_dir = os.path.join(get_project_root(), 'dataset', directory)
    
    # Create the directory if it doesn't exist
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
        
    output_path = os.path.join(processed_dir, file_name)
    logger.info(f"Saving processed data to {output_path}")
    
    try:
        df.to_csv(output_path, index=False)
        return output_path
    except Exception as e:
        logger.error(f"Error saving to {output_path}: {str(e)}")
        raise

def ensure_directory_exists(directory_path):
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory_path (str): Path to the directory
    
    Returns:
        str: Path to the directory
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logger.info(f"Created directory: {directory_path}")
    
    return directory_path 