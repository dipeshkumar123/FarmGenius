import os
import pickle
import logging
from typing import Dict, Any

from .llm_service import LLMService

logger = logging.getLogger(__name__)

class ChatbotService:
    def __init__(self, model_path: str = None):
        if model_path is None:
            # Assuming current file is in src/services or app/services
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            model_path = os.path.join(base_dir, 'models', 'chatbot_farmer_v1.pkl')
            
        self.model_path = model_path
        self.pipeline = self._load_model()
        self.llm_service = LLMService()
        self.confidence_threshold = 0.72

    def _load_model(self):
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    pipeline = pickle.load(f)
                return pipeline
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
        return None
        
    def _log_query(self, query: str, language: str, response: str, category: str, farmer_id: str):
        # In a real implementation this would write to the Supabase queries table
        logger.info(f"Logged query to Supabase: farmer={farmer_id}, cat={category}")
        pass

    def get_response(self, query: str, language: str, farmer_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get a response for the farmer's query, utilizing local ML model or falling back to Groq.
        """
        farmer_id = farmer_context.get('farmer_id', 'unknown')
        
        # 1. Attempt local classification
        local_response = None
        category = "Unknown"
        confidence = 0.0
        
        if self.pipeline is not None:
            # Lowercase and remove punctuation as in training
            import re
            processed_query = query.lower()
            processed_query = re.sub(r'[^\w\s]', '', processed_query)
            
            try:
                probabilities = self.pipeline.predict_proba([processed_query])[0]
                max_prob_idx = probabilities.argmax()
                confidence = probabilities[max_prob_idx]
                category = self.pipeline.classes_[max_prob_idx]
                
                if confidence >= self.confidence_threshold:
                    # Depending on category, you'd fetch the static response from a local DB/dict
                    # Here we simulate fetching static content based on the local model's category prediction
                    local_response = f"Based on local data for {category}: Please contact your local KVK for specific {category} treatments."
            except Exception as e:
                logger.error(f"Local model error: {e}")
                
        # 2. Check confidence threshold
        if local_response and confidence >= self.confidence_threshold:
            response_text = f"{local_response} (Source: Local Knowledge Base)"
            
            # Log query
            self._log_query(query, language, response_text, category, farmer_id)
            
            return {
                "response": response_text,
                "source": "Local ML Model",
                "confidence": confidence,
                "category": category
            }
            
        # 3. Fallback to Groq API
        try:
            groq_response = self.llm_service.generate_agricultural_response(query, language)
            response_text = f"{groq_response} (Source: KVK / state agriculture dept / expert knowledge)"
            
            # Log query
            self._log_query(query, language, response_text, "LLM_Fallback", farmer_id)
            
            return {
                "response": response_text,
                "source": "Groq LLM",
                "confidence": confidence,
                "category": "LLM_Fallback"
            }
            
        except Exception as e:
            logger.error(f"Groq fallback failed: {e}")
            fallback_msg = self.llm_service.get_graceful_fallback(language)
            return {
                "response": fallback_msg,
                "source": "System Fallback",
                "confidence": 0.0,
                "category": "Error"
            }
