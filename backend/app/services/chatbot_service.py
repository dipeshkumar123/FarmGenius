import os
import pickle
from app.services.llm_service import llm_service
from app.core.security import supabase

class ChatbotService:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.classes = []
        
        # Load local MultinomialNB model trained on FARMER_CORPUS if available
        # Note: Provide fallback behavior if file doesn't exist during initial build
        model_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "chatbot_farmer_v1.pkl")
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                pipeline = pickle.load(f)
                self.model = pipeline.named_steps['clf']
                self.vectorizer = pipeline.named_steps['tfidf']
                self.classes = self.model.classes_

    def get_response(self, query: str, language: str, farmer_id: str) -> dict:
        confidence = 0.0
        response_text = ""
        source = ""
        category = "General"

        # Attempt local model inference first
        if self.model and self.vectorizer:
            try:
                X = self.vectorizer.transform([query])
                probs = self.model.predict_proba(X)[0]
                max_prob = max(probs)
                max_idx = probs.argmax()
                
                if max_prob >= 0.72:
                    category = self.classes[max_idx]
                    response_text = f"Your query falls under the category: '{category}'. For precise guidance, please follow up with your local KVK."
                    confidence = max_prob
                    source = "Local ML Model"
            except Exception:
                pass
                
        # Fallback to Groq LLM API if confidence is below 0.72 or model missing
        if confidence < 0.72:
            llm_res = llm_service.get_response(query, language)
            response_text = llm_res["response"]
            confidence = llm_res["confidence"]
            source = llm_res["source"]

        # Background log to Supabase 'queries' table
        try:
            supabase.table("queries").insert({
                "farmer_id": farmer_id,
                "query_text": query,
                "language": language,
                "response": response_text,
                "category": category,
            }).execute()
        except Exception as e:
            # Silently catch insert errors to avoid disrupting user experience
            pass

        return {
            "response": response_text,
            "source": source,
            "confidence": confidence
        }

chatbot_service = ChatbotService()
