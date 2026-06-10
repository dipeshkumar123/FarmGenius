import groq
from app.core.config import settings

class LLMService:
    def __init__(self):
        self.client = groq.Groq(api_key=settings.GROQ_API_KEY)

    def get_response(self, query: str, language: str) -> dict:
        prompt = f"""You are an agricultural advisor for Indian smallholder farmers.
The farmer is asking in {language}. Answer in {language}.
Keep answers under 3 sentences. Be specific — name exact quantities, timings, and product names. If you are unsure, say so clearly and recommend the farmer contact their local KVK.
Do not make up treatments or dosages. Append source: KVK / state agriculture dept.
Farmer's query: {query}"""
        
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": prompt}
                ],
                model="llama-3.1-70b-versatile",
                temperature=0.3,
            )
            response_text = chat_completion.choices[0].message.content
            return {
                "response": response_text,
                "source": "LLM generated via Groq (Llama 3.1 70B)",
                "confidence": 0.9
            }
        except Exception as e:
            fallback_msgs = {
                "hi": "माफ करें, अभी सेवा उपलब्ध नहीं है। कृपया अपने KVK से संपर्क करें।",
                "kn": "ಕ್ಷಮಿಸಿ, ಸೇವೆ ಲಭ್ಯವಿಲ್ಲ. ನಿಮ್ಮ KVK ಗೆ ಸಂಪರ್ಕಿಸಿ.",
                "te": "క్షమించండి, సేవ అందుబాటులో లేదు. దయచేసి మీ KVK ని సంప్రదించండి.",
                "ta": "மன்னிக்கவும், சேவை கிடைக்கவில்லை. உங்கள் KVK ஐ தொடர்பு கொள்ளவும்.",
                "mr": "क्षमस्व, सेवा अनुपलब्ध आहे. कृपया तुमच्या KVK शी संपर्क साधा.",
                "en": "Sorry, service is unavailable. Please contact your local KVK."
            }
            msg = fallback_msgs.get(language, fallback_msgs["en"])
            return {
                "response": msg,
                "source": "Fallback System",
                "confidence": 0.0
            }

llm_service = LLMService()
