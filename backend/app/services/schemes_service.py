class SchemesService:
    def __init__(self):
        # A robust local database of real government schemes
        self.schemes_db = [
            {
                "scheme_name": "PM-KISAN (Pradhan Mantri Kisan Samman Nidhi)",
                "description": "Direct income support of ₹6,000 per year to eligible farmer families in 3 equal installments.",
                "eligibility": "Small & marginal farmers with less than 2 hectares of land.",
                "link": "https://pmkisan.gov.in",
                "category": "Direct Benefit",
                "amount": "₹6,000/year",
                "states": ["All"]
            },
            {
                "scheme_name": "PMFBY (Pradhan Mantri Fasal Bima Yojana)",
                "description": "Crop insurance scheme providing financial support if crops fail due to natural calamities, pests, or diseases.",
                "eligibility": "All farmers growing notified crops in notified areas.",
                "link": "https://pmfby.gov.in",
                "category": "Crop Insurance",
                "amount": "Up to full coverage",
                "states": ["All"]
            },
            {
                "scheme_name": "Kisan Credit Card (KCC)",
                "description": "Short-term credit needs for cultivation of crops, post-harvest expenses, and allied activities at subsidized interest rates.",
                "eligibility": "All farmers, tenant farmers, and sharecroppers.",
                "link": "https://www.nabard.org/kisan-credit-card",
                "category": "Credit/Loans",
                "amount": "Up to ₹3 lakh @ 7%",
                "states": ["All"]
            },
            {
                "scheme_name": "PMKSY (PM Krishi Sinchayee Yojana)",
                "description": "Micro-irrigation subsidy scheme for installing drip and sprinkler irrigation systems to improve water use efficiency.",
                "eligibility": "Individual farmers, farmer groups, cooperative societies.",
                "link": "https://pmksy.gov.in",
                "category": "Irrigation",
                "amount": "55% subsidy",
                "states": ["All"]
            },
            {
                "scheme_name": "eNAM (National Agriculture Market)",
                "description": "Online trading portal for farmers to sell produce directly to buyers across India, bypassing middlemen.",
                "eligibility": "Any registered farmer.",
                "link": "https://www.enam.gov.in",
                "category": "Market Access",
                "amount": "Free platform",
                "states": ["All"]
            },
            {
                "scheme_name": "Raita Vidya Nidhi",
                "description": "Scholarship program for the children of farmers to encourage higher education.",
                "eligibility": "Children of farmers in Karnataka.",
                "link": "https://ssp.postmatric.karnataka.gov.in/",
                "category": "Direct Benefit",
                "amount": "Varies by course",
                "states": ["Karnataka"]
            },
            {
                "scheme_name": "MahaDBT Farmer Schemes",
                "description": "A single portal for farmers in Maharashtra to apply for various agricultural subsidies and schemes.",
                "eligibility": "Farmers registered in Maharashtra.",
                "link": "https://mahadbtmahait.gov.in/",
                "category": "Government",
                "amount": "Varies by scheme",
                "states": ["Maharashtra"]
            },
            {
                "scheme_name": "Rythu Bandhu",
                "description": "Agriculture Investment Support Scheme providing ₹5,000 per acre per season to farmers.",
                "eligibility": "Land owning farmers in Telangana.",
                "link": "http://rythubandhu.telangana.gov.in/",
                "category": "Direct Benefit",
                "amount": "₹5,000/acre",
                "states": ["Telangana"]
            },
            {
                "scheme_name": "KALIA Scheme",
                "description": "Krushak Assistance for Livelihood and Income Augmentation for small/marginal farmers.",
                "eligibility": "Farmers in Odisha.",
                "link": "https://kalia.odisha.gov.in/",
                "category": "Direct Benefit",
                "amount": "₹4,000/year",
                "states": ["Odisha"]
            }
        ]

    def get_realtime_schemes(self, crop: str, state: str) -> list:
        # Filter schemes based on state (or return if 'All')
        filtered = []
        for scheme in self.schemes_db:
            if "All" in scheme["states"] or state in scheme["states"]:
                filtered.append({
                    "scheme_name": scheme["scheme_name"],
                    "description": scheme["description"],
                    "eligibility": scheme["eligibility"],
                    "link": scheme["link"],
                    "category": scheme.get("category", "Government"),
                    "amount": scheme.get("amount", "Check details")
                })
        return filtered

schemes_service = SchemesService()
