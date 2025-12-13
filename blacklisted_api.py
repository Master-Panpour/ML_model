import os
import requests
import requests_cache
from dotenv import load_dotenv

# Load environment variables from .env file (if present)
load_dotenv()

# Read the API key from environment variables
APIVOID_API_KEY = os.getenv("APIVOID_API_KEY")

# Initialize cache for blacklist API calls
requests_cache.install_cache('urls_cache', backend='sqlite', expire_after=3600)

def check_url_blacklist(url):
    """
    Check a URL against the blacklist API with caching.
    """
    if not APIVOID_API_KEY:
        return {
            "success": False,
            "error": "Missing APIVOID_API_KEY"
        }

    api_url = "https://api.apivoid.com/v2/url-reputation"
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": APIVOID_API_KEY
    }
    payload = {"url": url}

    try:
        response = requests.post(api_url, json=payload, headers=headers)
        if response.status_code == 200:
            data = response.json().get("data", {})
            return {
                "success": True,
                "risk_score": data.get("risk_score"),
                "blacklist_engines": data.get("blacklists", {})
            }
        else:
            return {"success": False, "error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"success": False, "error": str(e)}
