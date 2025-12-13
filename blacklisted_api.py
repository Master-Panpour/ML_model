import requests_cache
import requests

# Cache for blacklist API calls (SQLite backend, persistent) :contentReference[oaicite:1]{index=1}
requests_cache.install_cache('urls_cache', backend='sqlite', expire_after=3600)

APIVOID_API_KEY = "YOUR_APIVOID_API_KEY"

def check_url_blacklist(url):
    api_url = "https://api.apivoid.com/v2/url-reputation"
    headers = {"Content-Type":"application/json", "X-API-Key":APIVOID_API_KEY}
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
            return {"success": False, "error":f"HTTP {response.status_code}"}
    except Exception as e:
        return {"success": False, "error":str(e)}
