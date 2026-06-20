import groq
from app.core.config import settings

class TranslationService:
    def __init__(self):
        self.client = groq.Groq(api_key=settings.GROQ_API_KEY)

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        if source_lang == target_lang:
            return text
            
        prompt = f"Translate the following text from {source_lang} to {target_lang}. Output ONLY the translated text, with no explanations, markdown, or extra formatting. Original text: {text}"
        
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a direct translation engine. You output ONLY the translated text without quotes or explanations."},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.1,
            )
            return chat_completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"Translation error: {e}")
            return text

translation_service = TranslationService()
