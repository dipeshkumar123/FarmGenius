import groq
import json
from app.core.config import settings
from app.services.price_service import price_service
from app.services.weather_service import weather_service
from app.services.schemes_service import schemes_service
from app.services.crop_service import crop_service
import asyncio

class LLMService:
    def __init__(self):
        self.client = groq.AsyncGroq(api_key=settings.GROQ_API_KEY)

        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_mandi_prices",
                    "description": "Get current mandi prices for a specific crop/commodity in a specific district and state.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "commodity": {"type": "string", "description": "The name of the crop/commodity (e.g., Wheat, Tomato)"},
                            "district": {"type": "string", "description": "The district name"},
                            "state": {"type": "string", "description": "The state name"}
                        },
                        "required": ["commodity", "district", "state"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_weather_forecast",
                    "description": "Get the current weather forecast and farming advisories for a specific district.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "district": {"type": "string", "description": "The district name"},
                            "state": {"type": "string", "description": "The state name"}
                        },
                        "required": ["district", "state"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_realtime_schemes",
                    "description": "Fetch real-time government agricultural schemes and subsidies for a crop in a state.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "crop": {"type": "string", "description": "The name of the crop"},
                            "state": {"type": "string", "description": "The state name"}
                        },
                        "required": ["crop", "state"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_crop_recommendation",
                    "description": "Get a crop recommendation based on soil nutrient levels.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "n": {"type": "number", "description": "Nitrogen content"},
                            "p": {"type": "number", "description": "Phosphorous content"},
                            "k": {"type": "number", "description": "Potassium content"},
                            "ph": {"type": "number", "description": "Soil pH level"},
                            "ec": {"type": "number", "description": "Electrical Conductivity"},
                            "s": {"type": "number", "description": "Sulphur content"},
                            "cu": {"type": "number", "description": "Copper content"},
                            "fe": {"type": "number", "description": "Iron content"},
                            "mn": {"type": "number", "description": "Manganese content"},
                            "zn": {"type": "number", "description": "Zinc content"},
                            "b": {"type": "number", "description": "Boron content"}
                        },
                        "required": ["n", "p", "k", "ph", "ec", "s", "cu", "fe", "mn", "zn", "b"]
                    }
                }
            }
        ]

    async def get_response(self, query: str, language: str) -> dict:
        system_prompt = f"""You are FarmGenius, an expert agricultural advisor for Indian smallholder farmers.
The farmer will ask you questions. You MUST answer in {language}.
If the user asks about crop prices, weather, schemes, or crop recommendation based on soil, ALWAYS use the relevant tool to fetch real data before answering. Do not guess.
If the tool returns no data or fails, politely inform the user that the information is currently unavailable.
Keep answers under 3 sentences unless explaining a scheme. Be specific and helpful.
Do not make up treatments or prices.
"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

        try:
            # Step 1: Initial LLM call with tools
            response = await self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
                temperature=0.1
            )
            
            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls
            
            # Step 2: Check if the model wanted to call a tool
            if tool_calls:
                messages.append(response_message)
                
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    if function_name == "get_mandi_prices":
                        tool_res = await price_service.get_prices(
                            commodity=function_args.get("commodity"),
                            district=function_args.get("district"),
                            state=function_args.get("state")
                        )
                    elif function_name == "get_weather_forecast":
                        tool_res = await weather_service.get_weather(
                            district=function_args.get("district"),
                            state=function_args.get("state")
                        )
                    elif function_name == "get_realtime_schemes":
                        tool_res = schemes_service.get_realtime_schemes(
                            crop=function_args.get("crop"),
                            state=function_args.get("state")
                        )
                    elif function_name == "get_crop_recommendation":
                        tool_res = crop_service.predict_crop(
                            n=function_args.get("n", 0),
                            p=function_args.get("p", 0),
                            k=function_args.get("k", 0),
                            ph=function_args.get("ph", 7.0),
                            ec=function_args.get("ec", 0),
                            s=function_args.get("s", 0),
                            cu=function_args.get("cu", 0),
                            fe=function_args.get("fe", 0),
                            mn=function_args.get("mn", 0),
                            zn=function_args.get("zn", 0),
                            b=function_args.get("b", 0)
                        )
                    else:
                        tool_res = "Tool not found."
                        
                    messages.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": json.dumps(tool_res)
                        }
                    )
                
                # Step 3: Second LLM call to synthesize the response with tool data
                final_response = await self.client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages,
                    temperature=0.3
                )
                response_text = final_response.choices[0].message.content
                source = f"Live Data Aggregation ({function_name})"
            else:
                response_text = response_message.content
                source = "LLM Knowledge Base"

            return {
                "response": response_text,
                "source": source,
                "confidence": 0.95
            }
            
        except Exception as e:
            print(f"LLM Tool Error: {e}")
            fallback_msgs = {
                "hi": "माफ करें, अभी सेवा उपलब्ध नहीं है। कृपया अपने KVK से संपर्क करें।",
                "en": "Sorry, service is unavailable. Please contact your local KVK."
            }
            return {
                "response": fallback_msgs.get(language, fallback_msgs["en"]),
                "source": "Error Fallback",
                "confidence": 0.0
            }

llm_service = LLMService()
