import logging
from typing import Dict, List, Optional, Any
import json
from pathlib import Path
from googletrans import Translator
from langdetect import detect, detect_langs
from langdetect.lang_detect_exception import LangDetectException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LanguageModel:
    """
    Model to handle language detection and translation.
    """
    
    def __init__(self, data_dir: str = "data/languages"):
        """
        Initialize the language model.
        
        Args:
            data_dir (str): Directory to store language data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.translator = Translator()
        self.supported_languages = {
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'hi': 'Hindi',
            'bn': 'Bengali',
            'ar': 'Arabic'
        }
        self._load_translations()
        
    def _load_translations(self):
        """Load translations from disk."""
        self.translations = {}
        try:
            for lang_file in self.data_dir.glob("*.json"):
                with open(lang_file, 'r', encoding='utf-8') as f:
                    lang_code = lang_file.stem
                    self.translations[lang_code] = json.load(f)
            logger.info(f"Loaded translations for {len(self.translations)} languages")
        except Exception as e:
            logger.error(f"Error loading translations: {str(e)}")
            
    def _save_translations(self, lang_code: str):
        """Save translations to disk."""
        try:
            lang_file = self.data_dir / f"{lang_code}.json"
            with open(lang_file, 'w', encoding='utf-8') as f:
                json.dump(self.translations[lang_code], f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving translations for {lang_code}: {str(e)}")
            
    def detect_language(self, text: str) -> Dict[str, Any]:
        """
        Detect the language of the input text.
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Language detection result including:
                - detected_language: Detected language code
                - language_name: Full name of the detected language
                - confidence: Confidence score
                - supported: Whether the language is supported
                - message: Status message
        """
        try:
            if not text or len(text.strip()) < 10:
                return {
                    'detected_language': 'en',
                    'language_name': 'English',
                    'confidence': 0.0,
                    'supported': True,
                    'message': 'Text too short for reliable language detection'
                }
            
            # Get language probabilities
            lang_probabilities = detect_langs(text)
            
            # Get the most likely language
            best_lang = max(lang_probabilities, key=lambda x: x.prob)
            detected_lang = best_lang.lang
            confidence = best_lang.prob
            
            # Check if the detected language is supported
            is_supported = detected_lang in self.supported_languages
            if not is_supported:
                detected_lang = 'en'
                confidence = 1.0
            
            return {
                'detected_language': detected_lang,
                'language_name': self.supported_languages.get(detected_lang, 'Unknown'),
                'confidence': confidence,
                'supported': is_supported,
                'message': 'Language detected successfully'
            }
                
        except LangDetectException as e:
            logger.warning(f"Could not detect language: {str(e)}")
            return {
                'detected_language': 'en',
                'language_name': 'English',
                'confidence': 0.0,
                'supported': True,
                'message': 'Could not detect language, defaulting to English'
            }
        except Exception as e:
            logger.error(f"Error in language detection: {str(e)}")
            return {
                'detected_language': 'en',
                'language_name': 'English',
                'confidence': 0.0,
                'supported': True,
                'message': f'Error detecting language: {str(e)}'
            }
            
    def translate(self, text: str, target_lang: str, source_lang: Optional[str] = None) -> str:
        """
        Translate text to the target language.
        
        Args:
            text (str): Text to translate
            target_lang (str): Target language code
            source_lang (str, optional): Source language code
            
        Returns:
            str: Translated text
        """
        try:
            # Check if translation exists in cache
            if target_lang in self.translations:
                # TODO: Implement translation caching
                pass
                
            # Use Google Translate
            translation = self.translator.translate(
                text,
                dest=target_lang,
                src=source_lang
            )
            
            return translation.text
            
        except Exception as e:
            logger.error(f"Error translating text: {str(e)}")
            return text  # Return original text if translation fails
            
    def get_supported_languages(self) -> Dict[str, str]:
        """
        Get list of supported languages.
        
        Returns:
            dict: Dictionary of language codes and names
        """
        return self.supported_languages
        
    def is_language_supported(self, lang_code: str) -> bool:
        """
        Check if a language is supported.
        
        Args:
            lang_code (str): Language code to check
            
        Returns:
            bool: True if language is supported
        """
        return lang_code in self.supported_languages
        
    def translate_response(self, response: Dict[str, Any], target_lang: str) -> Dict[str, Any]:
        """
        Translate a response dictionary to the target language.
        
        Args:
            response (dict): Response dictionary
            target_lang (str): Target language code
            
        Returns:
            dict: Translated response
        """
        try:
            translated_response = response.copy()
            
            # Translate response text
            if 'response_text' in translated_response:
                translated_response['response_text'] = self.translate(
                    translated_response['response_text'],
                    target_lang
                )
                
            # Translate additional data if present
            if 'additional_data' in translated_response:
                additional_data = translated_response['additional_data']
                
                # Translate recommendation text
                if 'recommendation_text' in additional_data:
                    additional_data['recommendation_text'] = self.translate(
                        additional_data['recommendation_text'],
                        target_lang
                    )
                    
                # Translate crop names
                if 'top_recommendations' in additional_data:
                    for rec in additional_data['top_recommendations']:
                        if 'crop' in rec:
                            rec['crop'] = self.translate(
                                rec['crop'],
                                target_lang
                            )
                            
            return translated_response
            
        except Exception as e:
            logger.error(f"Error translating response: {str(e)}")
            return response  # Return original response if translation fails
            
    def get_language_name(self, lang_code: str) -> str:
        """
        Get the full name of a language from its code.
        
        Args:
            lang_code (str): Language code
            
        Returns:
            str: Language name
        """
        return self.supported_languages.get(lang_code, 'Unknown') 