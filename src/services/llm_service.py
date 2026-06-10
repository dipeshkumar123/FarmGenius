import os
import time
import logging
import requests
from requests.exceptions import RequestException

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model = "llama-3.1-70b-versatile"
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"

    def get_graceful_fallback(self, language: str) -> str:
        fallbacks = {
            "hi": "माफ करें, अभी सेवा उपलब्ध नहीं है। कृपया अपने KVK से संपर्क करें।",
            "kn": "ಕ್ಷಮಿಸಿ, ಸೇವೆ ಲಭ್ಯವಿಲ್ಲ. ನಿಮ್ಮ KVK ಗೆ ಸಂಪರ್ಕಿಸಿ.",
            "te": "క్షమించండి, సేవ అందుబాటులో లేదు. దయచేసి మీ KVKని సంప్రదించండి.",
            "ta": "மன்னிக்கவும், சேவை கிடைக்கவில்லை. உங்கள் KVK யை தொடர்பு கொள்ளவும்.",
            "mr": "माफ करा, सेवा उपलब्ध नाही. कृपया आपल्या KVK शी संपर्क साधा.",
            "en": "Sorry, service is unavailable. Please contact your local KVK."
        }
        # Fallback to English if language code is unknown
        return fallbacks.get(language.lower()[:2], fallbacks["en"])

    def generate_agricultural_response(self, query: str, language: str) -> str:
        """
        Generates an agricultural response using Groq. Native handling of translation and logic.
        """
        if not self.api_key:
            logger.error("GROQ_API_KEY is not set.")
            return self.get_graceful_fallback(language)

        system_prompt = (
            f"You are an agricultural advisor for Indian smallholder farmers. "
            f"The farmer is asking in {language}. Answer in {language}. "
            f"Keep answers under 3 sentences. Be specific — name exact quantities, "
            f"timings, and product names. If you are unsure, say so clearly and "
            f"recommend the farmer contact their local KVK. "
            f"Do not make up treatments or dosages."
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            "temperature": 0.2, # Low temperature for more factual responses
            "max_tokens": 150
        }

        max_retries = 1
        for attempt in range(max_retries + 1):
            try:
                response = requests.post(self.base_url, headers=headers, json=data, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    return result["choices"][0]["message"]["content"].strip()
                    
                elif response.status_code == 429:
                    if attempt < max_retries:
                        logger.warning("Rate limit hit (429). Waiting 60 seconds before retry...")
                        time.sleep(60)
                        continue
                    else:
                        logger.error("Rate limit hit (429) and max retries exceeded.")
                        return self.get_graceful_fallback(language)
                else:
                    logger.error(f"Groq API error: {response.status_code} - {response.text}")
                    return self.get_graceful_fallback(language)

            except RequestException as e:
                logger.error(f"Request to Groq API failed: {e}")
                if attempt == max_retries:
                    return self.get_graceful_fallback(language)
                
        return self.get_graceful_fallback(language)
