import pandas as pd
import numpy as np
import re
import logging
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

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

class QueryProcessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(max_features=5000)
    
    def clean_text(self, text):
        """Clean and preprocess text data."""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        
        return ' '.join(tokens)
    
    def extract_keywords(self, text, top_n=5):
        """Extract the most important keywords from text."""
        if not text or text.strip() == "":
            return []
        
        # Tokenize the text
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords
        tokens = [token for token in tokens if token not in self.stop_words]
        
        # Count frequency of each token
        freq_dist = nltk.FreqDist(tokens)
        
        # Get the most common tokens
        keywords = [word for word, freq in freq_dist.most_common(top_n)]
        
        return keywords
    
    def categorize_query(self, query):
        """Simple rule-based categorization of queries."""
        query = query.lower()
        
        categories = {
            'pest_disease': ['pest', 'disease', 'infection', 'infest', 'bug', 'insect', 'fungus', 'bacteria'],
            'crop_management': ['cultivation', 'irrigation', 'fertilizer', 'manure', 'organic', 'spacing', 'planting'],
            'market_price': ['price', 'market', 'sell', 'cost', 'profit', 'rate', 'sale'],
            'weather': ['weather', 'rain', 'forecast', 'climate', 'temperature', 'humidity'],
            'soil': ['soil', 'ph', 'fertility', 'nutrient', 'compost'],
            'financial': ['loan', 'credit', 'scheme', 'subsidy', 'insurance', 'finance'],
            'equipment': ['machine', 'equipment', 'tool', 'tractor', 'implement'],
            'seeds': ['seed', 'variety', 'hybrid', 'germination'],
        }
        
        # Check each category for keyword matches
        scores = {category: 0 for category in categories}
        
        for category, keywords in categories.items():
            for keyword in keywords:
                if keyword in query:
                    scores[category] += 1
        
        # Get the highest scoring category
        max_score = max(scores.values())
        if max_score == 0:
            return 'general'
        
        top_categories = [category for category, score in scores.items() if score == max_score]
        return top_categories[0]  # Return the first highest scoring category
    
    def process_query_data(self, data=None):
        """Process farmer query dataset."""
        if data is None:
            logger.info("Loading farmer query dataset")
            data = load_csv_data('farmer_queries.csv')
        
        logger.info(f"Processing farmer query dataset with {len(data)} records")
        
        # Create a copy of the data
        processed_data = data.copy()
        
        # Clean the questions and answers
        processed_data['clean_question'] = processed_data['questions'].apply(self.clean_text)
        processed_data['clean_answer'] = processed_data['answers'].apply(self.clean_text)
        
        # Extract keywords from questions
        processed_data['keywords'] = processed_data['clean_question'].apply(
            lambda x: ','.join(self.extract_keywords(x))
        )
        
        # Categorize questions
        processed_data['category'] = processed_data['questions'].apply(self.categorize_query)
        
        # Remove rows with empty questions or answers
        processed_data = processed_data[
            (processed_data['clean_question'].str.strip() != '') & 
            (processed_data['clean_answer'].str.strip() != '')
        ]
        
        # Create category-specific datasets
        categories = processed_data['category'].unique()
        category_datasets = {}
        
        for category in categories:
            category_data = processed_data[processed_data['category'] == category]
            category_datasets[category] = category_data
            logger.info(f"Created dataset for category '{category}' with {len(category_data)} entries")
        
        logger.info(f"Processed query dataset contains {len(processed_data)} records after cleaning")
        
        return {
            'processed_data': processed_data,
            'category_datasets': category_datasets
        }
    
    def save_processed_data(self, processed_data_dict):
        """Save the processed query data."""
        outputs = {}
        
        # Save the full processed dataset
        outputs['processed'] = save_processed_data(
            processed_data_dict['processed_data'], 
            'processed_queries.csv'
        )
        
        # Save category-specific datasets
        for category, data in processed_data_dict['category_datasets'].items():
            file_name = f'queries_{category}.csv'
            outputs[category] = save_processed_data(data, file_name)
        
        return outputs

def main():
    """Process the farmer query dataset and save the results."""
    processor = QueryProcessor()
    processed_data_dict = processor.process_query_data()
    output_paths = processor.save_processed_data(processed_data_dict)
    
    logger.info("Farmer query data processing completed. Outputs saved to:")
    for key, path in output_paths.items():
        logger.info(f"- {key}: {path}")

if __name__ == "__main__":
    main() 