import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import json
from pathlib import Path
import os
import sqlite3
from src.utils.file_utils import get_project_root, ensure_directory_exists

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class UserModel:
    """
    Model to handle user preferences, history, and personalization.
    """
    
    def __init__(self, db_path=None):
        """Initialize the user model with database connection."""
        try:
            # Set up database path
            if db_path is None:
                project_root = get_project_root()
                data_dir = os.path.join(project_root, 'data')
                ensure_directory_exists(data_dir)
                db_path = os.path.join(data_dir, 'users.db')
            
            self.db_path = db_path
            self._initialize_db()
            logger.info(f"User model initialized with database at {self.db_path}")
            
        except Exception as e:
            logger.error(f"Error initializing user model: {str(e)}")
            raise
            
    def _initialize_db(self):
        """Initialize the SQLite database and create tables if they don't exist."""
        try:
            ensure_directory_exists(os.path.dirname(self.db_path))
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    preferences TEXT,
                    language TEXT DEFAULT 'en'
                )
            ''')
            
            # Create chat history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    query TEXT,
                    response TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise
            
    def get_user(self, user_id):
        """Get user information."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT user_id, created_at, last_active, preferences, language
                FROM users
                WHERE user_id = ?
            ''', (user_id,))
            
            user = cursor.fetchone()
            conn.close()
            
            if user:
                return {
                    'user_id': user[0],
                    'created_at': user[1],
                    'last_active': user[2],
                    'preferences': json.loads(user[3]) if user[3] else {},
                    'language': user[4]
                }
            return None
            
        except Exception as e:
            logger.error(f"Error getting user: {str(e)}")
            return None
            
    def create_user(self, user_id, preferences=None, language='en'):
        """Create a new user."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO users (user_id, preferences, language)
                VALUES (?, ?, ?)
            ''', (user_id, json.dumps(preferences or {}), language))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Created new user: {user_id}")
            return True
            
        except sqlite3.IntegrityError:
            logger.warning(f"User already exists: {user_id}")
            return False
        except Exception as e:
            logger.error(f"Error creating user: {str(e)}")
            return False
            
    def update_preferences(self, user_id, preferences):
        """
        Update user preferences.
        
        Args:
            user_id (str): User identifier
            preferences (dict): Dictionary of preferences to update
            
        Returns:
            dict: Updated preferences or error response
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if user exists
            cursor.execute('SELECT preferences FROM users WHERE user_id = ?', (user_id,))
            result = cursor.fetchone()
            
            if not result:
                # Create user if doesn't exist
                cursor.execute('''
                    INSERT INTO users (user_id, preferences)
                    VALUES (?, ?)
                ''', (user_id, json.dumps(preferences)))
                new_prefs = preferences
            else:
                # Merge with existing preferences
                current_prefs = json.loads(result[0]) if result[0] else {}
                current_prefs.update(preferences)
                new_prefs = current_prefs
                
                # Update preferences
                cursor.execute('''
                    UPDATE users
                    SET preferences = ?, last_active = CURRENT_TIMESTAMP
                    WHERE user_id = ?
                ''', (json.dumps(new_prefs), user_id))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Updated preferences for user: {user_id}")
            return {
                'success': True,
                'user_id': user_id,
                'preferences': new_prefs,
                'message': 'Preferences updated successfully'
            }
            
        except Exception as e:
            logger.error(f"Error updating preferences: {str(e)}")
            return {
                'success': False,
                'user_id': user_id,
                'message': f'Error updating preferences: {str(e)}'
            }
            
    def get_preferences(self, user_id):
        """Get user preferences."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT preferences FROM users WHERE user_id = ?', (user_id,))
            result = cursor.fetchone()
            conn.close()
            
            if result and result[0]:
                return json.loads(result[0])
            return {}
            
        except Exception as e:
            logger.error(f"Error getting preferences: {str(e)}")
            return {}
            
    def update_language(self, user_id, language):
        """Update user's preferred language."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE users
                SET language = ?, last_active = CURRENT_TIMESTAMP
                WHERE user_id = ?
            ''', (language, user_id))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Updated language for user: {user_id} to {language}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating language: {str(e)}")
            return False
            
    def get_language(self, user_id):
        """Get user's preferred language."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT language FROM users WHERE user_id = ?', (user_id,))
            result = cursor.fetchone()
            conn.close()
            
            return result[0] if result else 'en'
            
        except Exception as e:
            logger.error(f"Error getting language: {str(e)}")
            return 'en'
            
    def add_chat_history(self, user_id, query, response):
        """Add a chat interaction to history."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO chat_history (user_id, query, response)
                VALUES (?, ?, ?)
            ''', (user_id, query, response))
            
            # Update last active timestamp
            cursor.execute('''
                UPDATE users
                SET last_active = CURRENT_TIMESTAMP
                WHERE user_id = ?
            ''', (user_id,))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Added chat history for user: {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding chat history: {str(e)}")
            return False
            
    def get_chat_history(self, user_id, limit=10):
        """Get recent chat history for a user."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT query, response, timestamp
                FROM chat_history
                WHERE user_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (user_id, limit))
            
            history = cursor.fetchall()
            conn.close()
            
            return [
                {
                    'query': entry[0],
                    'response': entry[1],
                    'timestamp': entry[2]
                }
                for entry in history
            ]
            
        except Exception as e:
            logger.error(f"Error getting chat history: {str(e)}")
            return []
        
    def add_query_history(self, user_id: str, query_type: str, query_data: Dict) -> bool:
        """
        Add a query to user history.
        
        Args:
            user_id (str): User identifier
            query_type (str): Type of query (queries, crop_recommendations, etc.)
            query_data (dict): Query data to store
            
        Returns:
            bool: True if successful
        """
        if user_id not in self.users:
            return False
            
        user = self.users[user_id]
        query_data['timestamp'] = datetime.now().isoformat()
        
        # Add to history
        user['history'][query_type].append(query_data)
        
        # Update statistics
        user['statistics'][f'total_{query_type}'] += 1
        
        # Keep only last 100 entries for each type
        if len(user['history'][query_type]) > 100:
            user['history'][query_type] = user['history'][query_type][-100:]
            
        user['last_active'] = datetime.now().isoformat()
        self._save_user(user_id)
        return True
        
    def get_user_statistics(self, user_id: str) -> Dict[str, Any]:
        """
        Get user statistics.
        
        Args:
            user_id (str): User identifier
            
        Returns:
            dict: User statistics including:
                - total_queries: Total number of queries
                - total_recommendations: Total number of crop recommendations
                - total_weather_queries: Total weather queries
                - total_price_queries: Total price queries
                - total_disease_queries: Total disease queries
                - last_active: Last activity timestamp
                - created_at: Account creation timestamp
                - success: Whether the operation was successful
                - message: Status message
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get user info
            cursor.execute('''
                SELECT created_at, last_active
                FROM users
                WHERE user_id = ?
            ''', (user_id,))
            
            user_info = cursor.fetchone()
            if not user_info:
                return {
                    'success': False,
                    'message': f'User {user_id} not found'
                }
            
            # Get query counts by type
            cursor.execute('''
                SELECT COUNT(*) as total_queries,
                       SUM(CASE WHEN query LIKE '%recommend crops%' THEN 1 ELSE 0 END) as crop_queries,
                       SUM(CASE WHEN query LIKE '%weather%' THEN 1 ELSE 0 END) as weather_queries,
                       SUM(CASE WHEN query LIKE '%price%' THEN 1 ELSE 0 END) as price_queries,
                       SUM(CASE WHEN query LIKE '%disease%' THEN 1 ELSE 0 END) as disease_queries
                FROM chat_history
                WHERE user_id = ?
            ''', (user_id,))
            
            stats = cursor.fetchone()
            conn.close()
            
            return {
                'success': True,
                'user_id': user_id,
                'total_queries': stats[0],
                'total_recommendations': stats[1],
                'total_weather_queries': stats[2],
                'total_price_queries': stats[3],
                'total_disease_queries': stats[4],
                'last_active': user_info[1],
                'created_at': user_info[0],
                'message': 'Statistics retrieved successfully'
            }
            
        except Exception as e:
            logger.error(f"Error getting user statistics: {str(e)}")
            return {
                'success': False,
                'user_id': user_id,
                'message': f'Error retrieving statistics: {str(e)}'
            }
        
    def get_recent_history(self, user_id: str, query_type: str, limit: int = 10) -> List[Dict]:
        """
        Get recent query history for a specific type.
        
        Args:
            user_id (str): User identifier
            query_type (str): Type of query to get history for
            limit (int): Maximum number of entries to return
            
        Returns:
            list: Recent query history
        """
        if user_id not in self.users:
            return []
            
        return self.users[user_id]['history'][query_type][-limit:]
        
    def add_price_alert(self, user_id: str, commodity: str, target_price: float, 
                       is_above: bool = True) -> bool:
        """
        Add a price alert for a commodity.
        
        Args:
            user_id (str): User identifier
            commodity (str): Commodity name
            target_price (float): Target price
            is_above (bool): True if alert when price goes above target
            
        Returns:
            bool: True if successful
        """
        if user_id not in self.users:
            return False
            
        alert = {
            'commodity': commodity,
            'target_price': target_price,
            'is_above': is_above,
            'created_at': datetime.now().isoformat(),
            'active': True
        }
        
        self.users[user_id]['preferences']['price_alerts'].append(alert)
        self._save_user(user_id)
        return True
        
    def add_weather_alert(self, user_id: str, location: str, conditions: List[str]) -> bool:
        """
        Add a weather alert for a location.
        
        Args:
            user_id (str): User identifier
            location (str): Location name
            conditions (list): List of weather conditions to alert for
            
        Returns:
            bool: True if successful
        """
        if user_id not in self.users:
            return False
            
        alert = {
            'location': location,
            'conditions': conditions,
            'created_at': datetime.now().isoformat(),
            'active': True
        }
        
        self.users[user_id]['preferences']['weather_alerts'].append(alert)
        self._save_user(user_id)
        return True 