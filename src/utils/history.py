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
    Saves to both the database (primary) and JSON file (legacy fallback).
    
    Args:
        user_id (str): User identifier
        query (str): User's query
        response (Dict[str, Any]): System's response
        intent (str, optional): Detected intent of the query
        
    Returns:
        bool: True if saved successfully, False otherwise
    """
    try:
        # Try database first
        try:
            from src.db.connection import SessionLocal
            from src.db import crud as db_crud
            db = SessionLocal()
            try:
                response_text = response.get('message', '') if isinstance(response, dict) else str(response)
                db_crud.add_chat_entry(
                    db,
                    user_id=user_id,
                    query=query,
                    response=response_text,
                    intent=intent,
                )
            finally:
                db.close()
        except Exception as db_err:
            logger.warning(f"DB save failed, falling back to JSON: {db_err}")

        # Also save to JSON file (legacy / backup)
        history_dir = os.path.join('data', 'history')
        os.makedirs(history_dir, exist_ok=True)
        
        user_history_file = os.path.join(history_dir, f'{user_id}_history.json')
        
        history = []
        if os.path.exists(user_history_file):
            with open(user_history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
        
        entry = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'response': response,
            'intent': intent
        }
        history.append(entry)
        
        with open(user_history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=4)
            
        logger.info(f"Saved conversation history for user {user_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving conversation history: {str(e)}")
        return False 