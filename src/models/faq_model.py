import os
import logging
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from src.models.text_embeddings import TextEmbedding, create_embeddings_from_faq
from src.utils.file_utils import load_csv_data, get_project_root

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FAQModel:
    """
    FAQ answering model using semantic similarity search.
    """
    
    def __init__(self, embedding_model=None, faq_data=None):
        """
        Initialize the FAQ model.
        
        Args:
            embedding_model: Pre-trained TextEmbedding model
            faq_data: DataFrame containing FAQ data
        """
        self.embedding_model = embedding_model
        self.faq_data = faq_data
        self.lemmatizer = WordNetLemmatizer()
        
        # Load data if not provided
        if self.faq_data is None:
            self._load_faq_data()
        
        # Load or create embedding model if not provided
        if self.embedding_model is None:
            self._load_embedding_model()
    
    def _load_faq_data(self):
        """Load the processed FAQ data."""
        try:
            self.faq_data = load_csv_data('processed_faq.csv', directory='processed')
            logger.info(f"Loaded FAQ data with {len(self.faq_data)} entries")
        except Exception as e:
            logger.error(f"Error loading FAQ data: {str(e)}")
            self.faq_data = pd.DataFrame(columns=['question', 'answers', 'clean_question'])
    
    def _load_embedding_model(self):
        """Load or create the embedding model."""
        model_dir = os.path.join(get_project_root(), 'models')
        model_path = os.path.join(model_dir, 'faq_embeddings.pkl')
        
        if os.path.exists(model_path):
            try:
                self.embedding_model = TextEmbedding.load(model_path)
                logger.info("Loaded existing embedding model")
            except Exception as e:
                logger.error(f"Error loading embedding model: {str(e)}")
                self._create_embedding_model(model_path)
        else:
            logger.info("No existing embedding model found, creating new one")
            self._create_embedding_model(model_path)
    
    def _create_embedding_model(self, model_path):
        """Create and save a new embedding model."""
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            self.embedding_model = create_embeddings_from_faq(
                faq_data=self.faq_data, 
                save_path=model_path
            )
            logger.info("Created new embedding model")
        except Exception as e:
            logger.error(f"Error creating embedding model: {str(e)}")
            # Create a basic empty model
            self.embedding_model = TextEmbedding()
    
    def _preprocess_query(self, query):
        """
        Preprocess the query text.
        
        Args:
            query (str): User query
        
        Returns:
            str: Preprocessed query
        """
        # Convert to lowercase
        query = query.lower()
        
        # Tokenize
        tokens = word_tokenize(query)
        
        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(tokens)
    
    def get_answer(self, query, threshold=0.3, max_results=3):
        """
        Get answer for a user query.
        
        Args:
            query (str): User question
            threshold (float): Minimum similarity score threshold
            max_results (int): Maximum number of answers to return
        
        Returns:
            dict: Response containing answers and metadata
        """
        # Preprocess the query
        processed_query = self._preprocess_query(query)
        
        # Search for similar questions
        search_results = self.embedding_model.search(processed_query, top_k=max_results)
        
        # Filter results by threshold and get corresponding answers
        filtered_results = []
        for idx, score, text in search_results:
            if score >= threshold:
                original_idx = self.faq_data[self.faq_data['clean_question'] == text].index[0]
                answer = self.faq_data.loc[original_idx, 'answers']
                original_question = self.faq_data.loc[original_idx, 'question']
                
                filtered_results.append({
                    'question': original_question,
                    'answer': answer,
                    'score': float(score)
                })
        
        # Construct response
        response = {
            'query': query,
            'results': filtered_results,
            'found_answer': len(filtered_results) > 0
        }
        
        return response
    
    def train(self, faq_data=None):
        """
        Train or retrain the FAQ model.
        
        Args:
            faq_data (DataFrame, optional): New FAQ data to train on
        
        Returns:
            self: The trained model
        """
        if faq_data is not None:
            self.faq_data = faq_data
        
        # Create a new embedding model
        model_dir = os.path.join(get_project_root(), 'models')
        model_path = os.path.join(model_dir, 'faq_embeddings.pkl')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        self.embedding_model = create_embeddings_from_faq(
            faq_data=self.faq_data, 
            save_path=model_path
        )
        
        return self


def main():
    """Test the FAQ model with sample queries."""
    # Create the FAQ model
    faq_model = FAQModel()
    
    # Test with some sample queries
    test_queries = [
        "What is crop rotation?",
        "How can I prevent soil erosion?",
        "What are the best irrigation methods?",
        "How to control pests in tomatoes?",
        "What fertilizer should I use for rice?",
    ]
    
    for query in test_queries:
        response = faq_model.get_answer(query)
        print(f"\nQuery: {query}")
        
        if response['found_answer']:
            for i, result in enumerate(response['results']):
                print(f"Answer {i+1} (score: {result['score']:.4f}):")
                print(f"  Q: {result['question']}")
                print(f"  A: {result['answer']}")
        else:
            print("No suitable answer found.")
    

if __name__ == "__main__":
    main() 