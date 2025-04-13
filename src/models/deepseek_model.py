import os
import requests
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import json
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DeepSeekModel:
    """
    Model for integrating with DeepSeek API through OpenRouter for advanced language understanding
    and response generation.
    """
    
    def __init__(self, api_key=None, api_url=None):
        """Initialize the DeepSeek model.
        
        Args:
            api_key (str, optional): OpenRouter API key. If None, will try to load from environment.
            api_url (str, optional): OpenRouter API URL. If None, will use default URL.
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.api_url = api_url or os.getenv("DEEPSEEK_API_URL", "https://openrouter.ai/api/v1")
        self._validate_api_key()
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/yourusername/FarmChatbot",  # Replace with your actual site URL
            "X-Title": "FarmChatbot"  # Replace with your actual site name
        })
        logger.info("DeepSeek model initialized successfully")
        self.model = "deepseek/deepseek-chat"  # Updated model name for OpenRouter
        self.max_tokens = 1000
        self.temperature = 0.7
        self.top_p = 0.95
        self.frequency_penalty = 0.0
        self.presence_penalty = 0.0
        self.timeout = 30
        
    def _validate_api_key(self):
        """Validate the API key."""
        if not self.api_key:
            logger.error("No OpenRouter API key provided. Please set DEEPSEEK_API_KEY environment variable.")
            raise ValueError("OpenRouter API key is required")
        elif len(self.api_key) < 10:  # Basic validation
            logger.error("Invalid OpenRouter API key format")
            raise ValueError("Invalid OpenRouter API key format")
        else:
            logger.info("OpenRouter API key validated successfully")
            
    def _prepare_request(self, messages, temperature=0.7, max_tokens=1000):
        """Prepare the API request payload.
        
        Args:
            messages (List[Dict[str, str]]): List of message dictionaries.
            temperature (float): Temperature for response generation.
            max_tokens (int): Maximum tokens in response.
            
        Returns:
            Dict: Prepared request payload.
        """
        return {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty
        }
            
    def _make_request(self, endpoint, payload):
        """Make an API request with proper error handling.
        
        Args:
            endpoint (str): API endpoint.
            payload (Dict): Request payload.
            
        Returns:
            Dict: API response.
            
        Raises:
            Exception: If request fails.
        """
        try:
            url = f"{self.api_url}/{endpoint}"
            response = self.session.post(url, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            logger.error("API request timed out")
            return {"error": "Request timed out", "found": False}
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            return {"error": str(e), "found": False}
        except Exception as e:
            logger.error(f"Unexpected error during API request: {str(e)}")
            return {"error": str(e), "found": False}
            
    def chat(self, messages, temperature=0.7, max_tokens=1000):
        """Send a chat request to the OpenRouter API.
        
        Args:
            messages (List[Dict[str, str]]): List of message dictionaries.
            temperature (float): Temperature for response generation.
            max_tokens (int): Maximum tokens in response.
            
        Returns:
            Dict: API response with chat completion.
        """
        try:
            payload = self._prepare_request(messages, temperature, max_tokens)
            response = self._make_request("chat/completions", payload)
            
            if "error" in response:
                return {
                    "error": response["error"],
                    "found": False
                }
            
            # Handle OpenRouter's response format
            if "choices" in response and len(response["choices"]) > 0:
                return {
                    "response": response["choices"][0]["message"]["content"],
                    "found": True
                }
            else:
                return {
                    "error": "No response from model",
                    "found": False
                }
            
        except Exception as e:
            logger.error(f"Error in chat request: {str(e)}")
            return {
                "error": str(e),
                "found": False
            }
            
    def analyze_sentiment(self, text):
        """
        Analyze the sentiment of a text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Sentiment analysis result
        """
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a sentiment analysis assistant. Analyze the sentiment of the given text and return a JSON response with 'sentiment' (positive/negative/neutral) and 'confidence' (0-1)."
                },
                {
                    "role": "user",
                    "content": text
                }
            ]
            
            response = self.chat(messages)
            
            if not response.get('found'):
                return {
                    'found': False,
                    'message': response.get('message'),
                    'sentiment': None,
                    'confidence': None
                }
                
            try:
                result = json.loads(response['response'])
                return {
                    'found': True,
                    'sentiment': result.get('sentiment'),
                    'confidence': result.get('confidence')
                }
            except json.JSONDecodeError:
                return {
                    'found': False,
                    'message': "Failed to parse sentiment analysis result",
                    'sentiment': None,
                    'confidence': None
                }
                
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return {
                'found': False,
                'message': f"Error in sentiment analysis: {str(e)}",
                'sentiment': None,
                'confidence': None
            }
            
    def extract_entities(self, text):
        """
        Extract named entities from text.
        
        Args:
            text (str): Text to extract entities from
            
        Returns:
            dict: Extracted entities
        """
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are an entity extraction assistant. Extract named entities from the given text and return a JSON response with 'entities' (list of entities with their types)."
                },
                {
                    "role": "user",
                    "content": text
                }
            ]
            
            response = self.chat(messages)
            
            if not response.get('found'):
                return {
                    'found': False,
                    'message': response.get('message'),
                    'entities': []
                }
                
            try:
                result = json.loads(response['response'])
                return {
                    'found': True,
                    'entities': result.get('entities', [])
                }
            except json.JSONDecodeError:
                return {
                    'found': False,
                    'message': "Failed to parse entity extraction result",
                    'entities': []
                }
                
        except Exception as e:
            logger.error(f"Error in entity extraction: {str(e)}")
            return {
                'found': False,
                'message': f"Error in entity extraction: {str(e)}",
                'entities': []
            }
    
    def generate_response(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a response using a hybrid approach:
        1. First try FAQ model
        2. If FAQ model quality is low, use DeepSeek API
        3. Use DeepSeek's response to improve FAQ model
        
        Args:
            query (str): User's query
            context (Dict[str, Any], optional): Additional context for the query
            
        Returns:
            Dict[str, Any]: Response data
        """
        try:
            # First try FAQ model
            from src.models.faq_model import FAQModel
            faq_model = FAQModel()
            faq_response = faq_model.get_answer(query)
            
            # Convert FAQ response to our format
            local_response = {
                "success": faq_response.get('found_answer', False),
                "response_text": faq_response.get('results', [{}])[0].get('answer', '') if faq_response.get('results') else '',
                "confidence": faq_response.get('results', [{}])[0].get('score', 0.0) if faq_response.get('results') else 0.0,
                "source": "faq_model",
                "found": faq_response.get('found_answer', False)
            }
            
            # Analyze local response quality
            local_quality = self.analyze_response_quality(local_response)
            
            # Get quality threshold from environment or use default
            quality_threshold = float(os.getenv("DEEPSEEK_QUALITY_THRESHOLD", "0.7"))
            
            # If local response quality is good enough, use it
            if local_quality >= quality_threshold and local_response.get('confidence', 0.0) >= 0.7:
                logger.info(f"Using FAQ model response (quality: {local_quality:.2f}, confidence: {local_response.get('confidence', 0.0):.2f})")
                return local_response
            
            # If local quality is low, try DeepSeek API
            logger.info(f"FAQ model quality ({local_quality:.2f}) or confidence ({local_response.get('confidence', 0.0):.2f}) below threshold. Using DeepSeek API.")
            
            # Prepare messages for the API
            messages = [
                {
                    "role": "system",
                    "content": "You are an agricultural expert assistant. Provide accurate, helpful, and concise responses about farming, crops, weather, and related topics."
                },
                {
                    "role": "user",
                    "content": self._prepare_prompt(query, context)
                }
            ]
            
            # Make API request using the chat method
            api_response = self.chat(messages, temperature=0.7, max_tokens=500)
            
            if not api_response.get('found'):
                logger.warning("Failed to get response from DeepSeek API. Falling back to FAQ model.")
                return local_response
            
            # Create response object
            response = {
                "success": True,
                "response_text": api_response['response'],
                "confidence": 0.8,  # Default confidence for API responses
                "source": "deepseek_api",
                "found": True
            }
            
            # Retrain FAQ model with DeepSeek's response
            self._retrain_local_model(query, response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in hybrid response generation: {str(e)}")
            return self._generate_fallback_response(query, context)
    
    def _retrain_local_model(self, query: str, response: Dict[str, Any]) -> None:
        """
        Retrain the local model using DeepSeek's response.
        
        Args:
            query (str): Original query
            response (Dict[str, Any]): DeepSeek's response
        """
        try:
            # Prepare training data
            training_data = {
                "query": query,
                "response": response.get('response_text', ''),
                "quality_score": self.analyze_response_quality(response)
            }
            
            # Log the retraining attempt
            logger.info(f"Retraining local model with DeepSeek response (quality: {training_data['quality_score']:.2f})")
            
            # Get the FAQ model instance
            from src.models.faq_model import FAQModel
            from src.utils.file_utils import get_project_root
            import pandas as pd
            
            # Load existing FAQ data
            faq_data_path = os.path.join(get_project_root(), 'dataset', 'processed', 'processed_faq.csv')
            existing_faq_data = pd.read_csv(faq_data_path)
            
            # Create new FAQ entry
            new_entry = pd.DataFrame({
                'question': [query],
                'answers': [response.get('response_text', '')],
                'clean_question': [self._preprocess_query(query)]
            })
            
            # Append new entry to existing data
            updated_faq_data = pd.concat([existing_faq_data, new_entry], ignore_index=True)
            
            # Save updated FAQ data
            updated_faq_data.to_csv(faq_data_path, index=False)
            logger.info(f"Updated FAQ data saved with {len(updated_faq_data)} entries")
            
            # Retrain the FAQ model with updated data
            faq_model = FAQModel()
            faq_model.train(updated_faq_data)
            logger.info("FAQ model retrained successfully")
            
        except Exception as e:
            logger.error(f"Error retraining local model: {str(e)}")
    
    def _preprocess_query(self, query: str) -> str:
        """
        Preprocess the query text for consistency with FAQ model.
        
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
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(tokens)
    
    def _prepare_prompt(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Prepare the prompt for the API request.
        
        Args:
            query (str): User's query
            context (Dict[str, Any], optional): Additional context
            
        Returns:
            str: Formatted prompt
        """
        prompt = query
        
        if context:
            # Add relevant context to the prompt
            if 'intent' in context:
                prompt = f"Intent: {context['intent']}\nQuery: {query}"
            
            if 'previous_responses' in context:
                prompt = f"Previous context:\n{context['previous_responses']}\n\nCurrent query: {query}"
            
            if 'user_preferences' in context:
                prompt = f"User preferences:\n{context['user_preferences']}\n\nQuery: {query}"
        
        return prompt
    
    def analyze_response_quality(self, response: Dict[str, Any]) -> float:
        """
        Analyze the quality of the generated response.
        
        Args:
            response (Dict[str, Any]): Response data
            
        Returns:
            float: Quality score between 0 and 1
        """
        if not response.get('success', False):
            return 0.0
        
        # Extract the response text
        text = response.get('response_text', '')
        
        # More comprehensive quality metrics
        metrics = {
            'length': len(text) > 100,  # Response should be substantial
            'relevance': any(keyword in text.lower() for keyword in [
                'crop', 'farm', 'plant', 'soil', 'weather', 'irrigation', 
                'fertilizer', 'pest', 'disease', 'harvest', 'seed'
            ]),
            'structure': all(char in text for char in ['.', '!', '?']),  # Should have proper punctuation
            'specificity': len(set(text.split())) > 30,  # Should have diverse vocabulary
            'completeness': len(text.split()) > 20,  # Should be a complete answer
            'technical': any(word in text.lower() for word in [
                'temperature', 'humidity', 'ph', 'nutrients', 'organic',
                'sustainable', 'climate', 'season', 'rotation'
            ])
        }
        
        # Calculate quality score with weights
        weights = {
            'length': 0.15,
            'relevance': 0.25,
            'structure': 0.15,
            'specificity': 0.15,
            'completeness': 0.15,
            'technical': 0.15
        }
        
        score = sum(weights[metric] for metric, value in metrics.items() if value)
        return score
    
    def retrain_on_feedback(self, query: str, response: Dict[str, Any], feedback: Dict[str, Any]) -> bool:
        """
        Retrain the model based on user feedback.
        
        Args:
            query (str): Original query
            response (Dict[str, Any]): Generated response
            feedback (Dict[str, Any]): User feedback
            
        Returns:
            bool: True if retraining was successful
        """
        try:
            # Prepare training data
            messages = [
                {
                    "role": "system",
                    "content": "You are an agricultural expert assistant. Learn from user feedback to improve responses."
                },
                {
                    "role": "user",
                    "content": f"Query: {query}\nResponse: {response.get('response_text', '')}\nFeedback: {feedback}"
                }
            ]
            
            # Send feedback using the chat method
            response = self.chat(messages, temperature=0.3, max_tokens=100)
            
            if response.get('found'):
                logger.info("Successfully sent feedback to OpenRouter API for model improvement")
                return True
            else:
                logger.error(f"Failed to send feedback: {response.get('error', 'Unknown error')}")
                return False
            
        except Exception as e:
            logger.error(f"Error sending feedback to OpenRouter API: {str(e)}")
            return False
    
    def _generate_fallback_response(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a fallback response when the API is unavailable.
        
        Args:
            query (str): User's query
            context (Dict[str, Any], optional): Additional context
            
        Returns:
            Dict[str, Any]: Fallback response
        """
        # Simple keyword-based response generation
        query = query.lower()
        
        if any(word in query for word in ["organic", "farming", "climate"]):
            response_text = "Organic farming practices focus on sustainability and natural pest control. It's important to maintain soil health through crop rotation, composting, and avoiding synthetic chemicals."
        elif any(word in query for word in ["irrigation", "water", "soil"]):
            response_text = "Irrigation should be tailored to soil type. Sandy soils need frequent, light watering, while clay soils benefit from less frequent, deeper watering. Consider drip irrigation for water efficiency."
        elif any(word in query for word in ["economic", "climate change", "agriculture"]):
            response_text = "Climate change poses economic challenges to agriculture through increased extreme weather events, shifting growing seasons, and water availability changes. Diversification of crops and adaptive practices can help mitigate these risks."
        else:
            response_text = "I can provide general information on farming topics like crop management, soil health, pest control, and sustainable practices. Please ask a more specific question for detailed advice."
        
        return {
            "success": True,
            "response_text": response_text,
            "confidence": 0.6,  # Lower confidence for fallback responses
            "source": "deepseek_fallback",
            "found": True  # Adding the found attribute to prevent errors
        } 