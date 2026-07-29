import os
import sys
import json
import asyncio
from bs4 import BeautifulSoup
from googlesearch import search
import httpx
from groq import AsyncGroq
from supabase import create_client, Client

# Ensure we can import app modules if run from backend/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core.config import settings

# Initialize clients
supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)
groq_client = AsyncGroq(api_key=settings.GROQ_API_KEY)

async def fetch_page_content(url: str) -> str:
    """Fetch text content from a URL."""
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            # Extract basic text (removing scripts and styles)
            for script in soup(["script", "style"]):
                script.extract()
            text = soup.get_text(separator=' ', strip=True)
            # Truncate to avoid context limit issues
            return text[:4000]
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return ""

async def aggregate_scheme_data():
    """Searches the web for latest Indian agriculture schemes and extracts data."""
    print("Searching for latest Indian agriculture schemes...")
    queries = [
        "latest agriculture schemes for farmers India 2026 site:gov.in",
        "PM Kisan Rythu Bandhu KALIA schemes updates 2026",
    ]
    
    urls = []
    for query in queries:
        try:
            for url in search(query, num_results=3, lang="en"):
                urls.append(url)
        except Exception as e:
            print(f"Search failed for '{query}': {e}")
            
    urls = list(set(urls))
    print(f"Found {len(urls)} URLs. Fetching content...")
    
    contents = []
    for url in urls[:5]:  # Limit to top 5
        text = await fetch_page_content(url)
        if text:
            contents.append(f"Source: {url}\nContent: {text}")
            
    combined_content = "\n\n".join(contents)
    
    # If no internet/search results (e.g. rate limits), provide a static context for the LLM
    if not combined_content:
        combined_content = """
        Source: https://agricoop.nic.in
        Content: Pradhan Mantri Kisan Samman Nidhi (PM-KISAN) provides ₹6000 per year. Pradhan Mantri Fasal Bima Yojana (PMFBY) provides crop insurance. Rythu Bandhu in Telangana gives ₹5000 per acre.
        """

    prompt = f"""
    You are an expert in Indian Agricultural Policy. Based on the following scraped text, extract a list of exactly 5 major government agriculture schemes for farmers in India.
    
    For each scheme, provide:
    1. scheme_name: The official name (e.g., "PM-KISAN")
    2. description: A clear 1-sentence explanation.
    3. eligibility: Who can apply (e.g., "Small & marginal farmers").
    4. link: The official application portal URL (must start with http:// or https://).
    5. category: One of ["Direct Benefit", "Crop Insurance", "Credit/Loans", "Irrigation", "Market Access", "Government"]
    6. amount: The financial benefit (e.g., "₹6,000/year").
    7. states: A list of applicable states. Use ["All"] if it's a central scheme, or ["Telangana"] if it's state-specific.
    
    Return ONLY a raw JSON array of objects. Do not include markdown formatting or backticks.
    
    Scraped Text:
    {combined_content[:15000]}
    """
    
    print("Calling Groq LLM to extract structured scheme data...")
    try:
        response = await groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.1
        )
        
        content = response.choices[0].message.content.strip()
        # Clean up any potential markdown backticks
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
            
        schemes = json.loads(content)
        return schemes
    except Exception as e:
        print(f"Failed to generate JSON with Groq: {e}")
        return []

async def main():
    print("Starting daily schemes update...")
    
    # 1. Fetch and Parse data
    new_schemes = await aggregate_scheme_data()
    
    if not new_schemes:
        print("No schemes extracted. Exiting.")
        return
        
    print(f"Extracted {len(new_schemes)} schemes. Updating Supabase...")
    
    # 2. Update Supabase
    # Wiping and re-inserting ensures stale schemes are removed.
    try:
        # Delete existing schemes. We delete where id is not null (everything)
        supabase.table("schemes").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
        
        # Insert new schemes
        response = supabase.table("schemes").insert(new_schemes).execute()
        print(f"Successfully inserted {len(response.data)} schemes into Supabase.")
    except Exception as e:
        print(f"Failed to update Supabase: {e}")

if __name__ == "__main__":
    asyncio.run(main())
