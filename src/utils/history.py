import logging
import json
import os
from datetime import datetime
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def save_to_history(user_id: str, query: str, response: Dict[str, Any], intent: str = None) -> bool:
    """
    Save a query and response to the user's conversation history.
    
    Args:
        user_id (str): User identifier
        query (str): User's query
        response (Dict[str, Any]): System's response
        intent (str, optional): Detected intent of the query
        
    Returns:
        bool: True if saved successfully, False otherwise
    """
    try:
        # Create history directory if it doesn't exist
        history_dir = os.path.join('data', 'history')
        os.makedirs(history_dir, exist_ok=True)
        
        # Create user's history file path
        user_history_file = os.path.join(history_dir, f'{user_id}_history.json')
        
        # Load existing history or create new
        history = []
        if os.path.exists(user_history_file):
            with open(user_history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
        
        # Add new entry
        entry = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'response': response,
            'intent': intent
        }
        history.append(entry)
        
        # Save updated history
        with open(user_history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=4)
            
        logger.info(f"Saved conversation history for user {user_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving conversation history: {str(e)}")
        return False 