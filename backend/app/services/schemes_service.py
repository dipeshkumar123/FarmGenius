from googlesearch import search

class SchemesService:
    def __init__(self):
        pass

    def get_realtime_schemes(self, crop: str, state: str) -> list:
        query = f"government agriculture schemes for {crop} farmers in {state} site:gov.in"
        schemes = []
        try:
            results = list(search(query, num_results=3, advanced=True))
            
            for r in results:
                schemes.append({
                    "scheme_name": r.title if hasattr(r, 'title') else "Government Scheme",
                    "description": r.description if hasattr(r, 'description') else "Details available on website.",
                    "eligibility": "Refer to official website for details.",
                    "link": r.url if hasattr(r, 'url') else str(r)
                })
        except Exception as e:
            print(f"Error fetching real-time schemes: {e}")
            
        if not schemes:
            schemes.append({
                "scheme_name": "PM-KISAN",
                "description": "Direct benefit transfer of Rs. 6000 per year.",
                "eligibility": "All landholding farmers families.",
                "link": "https://pmkisan.gov.in"
            })
            
        return schemes

schemes_service = SchemesService()
