import httpx
import json
import os
import time

class WeatherService:
    def __init__(self):
        self.district_coords = {}
        self._cache = {}
        self._cache_ttl = 3600  # 1 hour
        coords_path = os.path.join(os.path.dirname(__file__), "..", "districts_coords.json")
        try:
            with open(coords_path, "r") as f:
                self.district_coords = json.load(f)
        except Exception:
            self.district_coords = {"dharwad": {"lat": 15.45, "lon": 75.00}}

    async def get_weather(self, district: str, state: str) -> list:
        cache_key = f"{district.lower()}_{state.lower()}"
        current_time = time.time()
        
        # Check cache
        if cache_key in self._cache:
            cached_data, timestamp = self._cache[cache_key]
            if current_time - timestamp < self._cache_ttl:
                print(f"CACHE HIT: Weather for {district}")
                return cached_data

        print(f"CACHE MISS: Fetching weather for {district}")
        # Default to a central coordinate if district is unknown
        coord = self.district_coords.get(district.lower(), {"lat": 20.59, "lon": 78.96})
        
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": coord["lat"],
            "longitude": coord["lon"],
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max",
            "timezone": "Asia/Kolkata"
        }
        
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(url, params=params, timeout=10.0)
                data = resp.json()
                daily = data.get("daily", {})
                
                forecasts = []
                for i in range(len(daily.get("time", []))):
                    rainfall = daily.get("precipitation_sum")[i]
                    
                    # Simple rule-based advisory system
                    advisory = "Conditions look favorable for standard farming activities."
                    if rainfall > 20:
                        advisory = "Heavy rain expected — avoid spraying pesticides and monitor drainage."
                    elif rainfall == 0 and daily.get("temperature_2m_max")[i] > 38:
                        advisory = "High temperatures expected. Ensure adequate crop irrigation."
                        
                    forecasts.append({
                        "date": daily.get("time")[i],
                        "max_temp": daily.get("temperature_2m_max")[i],
                        "min_temp": daily.get("temperature_2m_min")[i],
                        "rainfall_mm": rainfall,
                        "wind_kmh": daily.get("wind_speed_10m_max")[i],
                        "farming_advisory": advisory
                    })
                
                # Store in cache
                if forecasts:
                    self._cache[cache_key] = (forecasts, current_time)
                    
                return forecasts
            except Exception:
                return []

weather_service = WeatherService()
