import numpy as np
import pandas as pd
import logging
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.file_utils import get_project_root

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TextEmbedding:
    """
    Text embedding model using TF-IDF for semantic similarity searches.
    This is a simpler alternative to transformer-based embeddings.
    """
    
    def __init__(self, max_features=5000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features, 
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.corpus_embeddings = None
        self.corpus_texts = None
    
    def fit(self, texts):
        """
        Fit the vectorizer and create embeddings for the corpus.
        
        Args:
            texts (list): List of text strings to embed
        """
        logger.info(f"Fitting TF-IDF vectorizer on {len(texts)} texts")
        self.corpus_texts = texts
        self.corpus_embeddings = self.vectorizer.fit_transform(texts)
        return self
    
    def encode(self, texts):
        """
        Encode new texts using the fitted vectorizer.
        
        Args:
            texts (list or str): Text(s) to encode
        
        Returns:
            Sparse matrix of TF-IDF features
        """
        if isinstance(texts, str):
            texts = [texts]
        
        return self.vectorizer.transform(texts)
    
    def search(self, query, top_k=5):
        """
        Search for most similar texts in the corpus.
        
        Args:
            query (str): Query text
            top_k (int): Number of results to return
        
        Returns:
            List of (index, score, text) tuples
        """
        if not self.corpus_embeddings is not None:
            raise ValueError("Model must be fit to a corpus before searching")
        
        # Encode the query
        query_embedding = self.encode([query])
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_embedding, self.corpus_embeddings).flatten()
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Create results
        results = [
            (idx, similarities[idx], self.corpus_texts[idx])
            for idx in top_indices
        ]
        
        return results
    
    def save(self, filepath):
        """Save the embedding model to a file."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'corpus_texts': self.corpus_texts,
                'corpus_embeddings': self.corpus_embeddings
            }, f)
        logger.info(f"Embedding model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load an embedding model from a file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        model = cls()
        model.vectorizer = data['vectorizer']
        model.corpus_texts = data['corpus_texts']
        model.corpus_embeddings = data['corpus_embeddings']
        
        logger.info(f"Embedding model loaded from {filepath}")
        return model


def create_embeddings_from_faq(faq_data=None, save_path=None):
    """
    Create and save text embeddings from FAQ data.
    
    Args:
        faq_data (DataFrame, optional): Processed FAQ data
        save_path (str, optional): Path to save the embedding model
    
    Returns:
        TextEmbedding model
    """
    from src.utils.file_utils import load_csv_data
    
    if faq_data is None:
        # Load the processed FAQ data
        try:
            faq_data = load_csv_data('processed_faq.csv', directory='processed')
        except Exception as e:
            logger.error(f"Error loading processed FAQ data: {str(e)}")
            return None
    
    logger.info(f"Creating embeddings from {len(faq_data)} FAQ entries")
    
    # Create embeddings
    embedding_model = TextEmbedding()
    embedding_model.fit(faq_data['clean_question'].tolist())
    
    # Save the model if a path is provided
    if save_path:
        embedding_model.save(save_path)
    else:
        # Use default path
        model_dir = os.path.join(get_project_root(), 'models')
        os.makedirs(model_dir, exist_ok=True)
        
        default_path = os.path.join(model_dir, 'faq_embeddings.pkl')
        embedding_model.save(default_path)
    
    return embedding_model


if __name__ == "__main__":
    # Create and save FAQ embeddings
    embedding_model = create_embeddings_from_faq()
    
    # Test the model with a sample query
    if embedding_model:
        test_query = "how to control pests"
        results = embedding_model.search(test_query, top_k=3)
        
        print(f"Test query: '{test_query}'")
        print("Top results:")
        for idx, score, text in results:
            print(f"  - {text} (score: {score:.4f})") 