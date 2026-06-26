import os
import groq
import json
from app.core.config import settings

class CropService:
    def __init__(self):
        pass

    async def predict_crop(self, location: str, soil_type: str, water_availability: str, farm_size: str, season: str, n: str = None, p: str = None, k: str = None) -> list:
        try:
            client = groq.Groq(api_key=settings.GROQ_API_KEY)
            
            npk_context = ""
            if n and p and k:
                npk_context = f"The soil NPK values are approximately N:{n}, P:{p}, K:{k}."

            prompt = f"""You are an expert Indian Agronomist AI. A farmer wants crop recommendations based on these exact details:
Location: {location}
Soil Type: {soil_type}
Water Availability: {water_availability}
Farm Size: {farm_size}
Target Season: {season}
{npk_context}

Analyze these conditions and recommend the top 3 best crops for this specific farmer. 
You MUST respond ONLY with a valid JSON object in this exact schema, containing exactly 3 crops:
{{
  "recommendations": [
    {{
      "rank": 1,
      "name": "Paddy (Rice)",
      "emoji": "🌾",
      "suitability": 0.96,
      "expectedYield": "45 q/acre",
      "marketPrice": "₹2,060/q",
      "profitEstimate": "₹92,700",
      "season": "Kharif",
      "water": "High",
      "duration": "120-150 days"
    }}
  ]
}}

Ensure the estimates are realistic for the {location} region and {farm_size} size. Do not output any markdown formatting, only raw JSON.
"""

            completion = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(completion.choices[0].message.content)
            return result.get("recommendations", [])
            
        except Exception as e:
            print(f"Error during AI crop recommendation: {e}")
            # Fallback mock data in case of LLM failure
            return [
                { "rank": 1, "name": "Paddy (Fallback)", "emoji": "🌾", "suitability": 0.90, "expectedYield": "40 q/acre", "marketPrice": "₹2,000/q", "profitEstimate": "₹80,000", "season": season, "water": "Medium", "duration": "120 days" },
                { "rank": 2, "name": "Maize (Fallback)", "emoji": "🌽", "suitability": 0.85, "expectedYield": "30 q/acre", "marketPrice": "₹1,900/q", "profitEstimate": "₹57,000", "season": season, "water": "Medium", "duration": "100 days" },
                { "rank": 3, "name": "Soybean (Fallback)", "emoji": "🫘", "suitability": 0.70, "expectedYield": "15 q/acre", "marketPrice": "₹4,000/q", "profitEstimate": "₹60,000", "season": season, "water": "Low", "duration": "95 days" }
            ]

crop_service = CropService()
