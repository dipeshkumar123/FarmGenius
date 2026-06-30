from app.core.security import supabase

class SchemesService:
    def __init__(self):
        # Local fallback in case DB is down
        self.fallback_schemes = [
            {
                "scheme_name": "PM-KISAN (Pradhan Mantri Kisan Samman Nidhi)",
                "description": "Direct income support of ₹6,000 per year to eligible farmer families in 3 equal installments.",
                "eligibility": "Small & marginal farmers with less than 2 hectares of land.",
                "link": "https://pmkisan.gov.in",
                "category": "Direct Benefit",
                "amount": "₹6,000/year",
                "states": ["All"]
            }
        ]

    def get_realtime_schemes(self, crop: str, state: str) -> list:
        try:
            # Query schemes table from Supabase
            # state is passed from frontend (e.g. "Karnataka", "Maharashtra")
            # We want to match schemes where "All" is in states array, or the specific state is in states array
            # However, postgREST filter for "contains array" is a bit tricky, so we'll fetch all and filter in Python,
            # or just rely on a simple query if the table is small. 
            response = supabase.table("schemes").select("*").execute()
            
            # PostgREST query results
            db_schemes = response.data
            
            if not db_schemes:
                return self.fallback_schemes
                
            filtered = []
            for scheme in db_schemes:
                states = scheme.get("states", [])
                # Handle cases where states might be stored as a string or array
                if not states:
                    states = ["All"]
                elif isinstance(states, str):
                    states = [states]
                    
                if "All" in states or state in states:
                    filtered.append({
                        "scheme_name": scheme.get("scheme_name"),
                        "description": scheme.get("description"),
                        "eligibility": scheme.get("eligibility"),
                        "link": scheme.get("link"),
                        "category": scheme.get("category", "Government"),
                        "amount": scheme.get("amount", "Check details")
                    })
                    
            return filtered
            
        except Exception as e:
            print(f"Error fetching schemes from Supabase: {e}")
            return self.fallback_schemes

schemes_service = SchemesService()
