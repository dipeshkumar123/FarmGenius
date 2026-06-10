import time
from datetime import date
import httpx
from app.core.security import supabase
from app.core.config import settings

# In-memory dictionary cache: { "commodity_district": {"timestamp": float, "data": dict} }
price_cache = {}

class PriceService:
    async def get_prices(self, commodity: str, district: str, state: str) -> dict:
        key = f"{commodity}_{district}".lower()
        now = time.time()
        
        # 1. Check in-memory cache (valid for 6 hours / 21600 seconds)
        if key in price_cache:
            cache_entry = price_cache[key]
            if now - cache_entry['timestamp'] < 21600:
                return cache_entry['data']

        today_str = date.today().isoformat()
        
        # 2. Check Supabase prices_cache for today's data
        try:
            res = supabase.table("prices_cache").select("*").eq("commodity", commodity).eq("district", district).eq("date", today_str).execute()
            if res.data and len(res.data) > 0:
                result_data = res.data[0]
                price_cache[key] = {'timestamp': now, 'data': result_data}
                return result_data
        except Exception:
            pass
        
        # 3. Fetch from data.gov.in AGMARKNET API
        url = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
        params = {
            "api-key": settings.DATA_GOV_IN_API_KEY,
            "format": "json",
            "filters[commodity]": commodity,
            "filters[district]": district,
            "filters[state]": state
        }
        
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(url, params=params, timeout=10.0)
                data = resp.json()
                if data.get("records"):
                    record = data["records"][0]
                    result = {
                        "commodity": commodity,
                        "district": district,
                        "min_price": float(record.get("min_price", 0)),
                        "max_price": float(record.get("max_price", 0)),
                        "modal_price": float(record.get("modal_price", 0)),
                        "date": today_str,
                        "unit": "Quintal" # Default metric commonly used in mandis
                    }
                    
                    # Store in Supabase
                    try:
                        supabase.table("prices_cache").upsert(result).execute()
                    except Exception:
                        pass
                        
                    # Store in-memory cache
                    price_cache[key] = {'timestamp': now, 'data': result}
                    return result
            except Exception:
                pass
                
        # 4. Dummy/Fallback Data on total failure
        return {
            "commodity": commodity,
            "district": district,
            "min_price": 0.0,
            "max_price": 0.0,
            "modal_price": 0.0,
            "date": today_str,
            "unit": "Quintal"
        }

price_service = PriceService()
