import pandas as pd
import numpy as np
import re
import logging
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from src.utils.file_utils import load_csv_data, save_processed_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Download NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception as e:
    logger.warning(f"Error downloading NLTK resources: {str(e)}")

class FAQProcessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """Clean and preprocess text data."""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        
        return ' '.join(tokens)
    
    def process_faq_data(self, data=None):
        """Process FAQ dataset."""
        if data is None:
            logger.info("Loading FAQ dataset")
            data = load_csv_data('faq_dataset.csv')
        
        logger.info(f"Processing FAQ dataset with {len(data)} records")
        
        # Clean question and answer columns
        data['clean_question'] = data['question'].apply(self.clean_text)
        
        # Create a processed dataset with the original and cleaned text
        processed_data = data[['question', 'answers', 'clean_question']].copy()
        
        # Remove duplicates
        processed_data = processed_data.drop_duplicates(subset=['clean_question'])
        
        # Remove rows with empty questions or answers
        processed_data = processed_data[
            (processed_data['clean_question'].str.strip() != '') & 
            (processed_data['answers'].notna())
        ]
        
        logger.info(f"Processed FAQ dataset contains {len(processed_data)} records after cleaning")
        
        return processed_data
    
    def save_processed_data(self, processed_data):
        """Save the processed FAQ data."""
        return save_processed_data(processed_data, 'processed_faq.csv')

def main():
    """Process the FAQ dataset and save the results."""
    processor = FAQProcessor()
    processed_data = processor.process_faq_data()
    output_path = processor.save_processed_data(processed_data)
    logger.info(f"FAQ data processing completed. Output saved to {output_path}")

if __name__ == "__main__":
    main() 