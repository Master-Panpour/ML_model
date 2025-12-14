from urllib.parse import urlparse

TRUSTED_DOMAINS = {
    "google.com",
    "microsoft.com",
    "github.com",
    "openai.com",
    "amazon.com",
    "apple.com",
    "facebook.com",
    "linkedin.com",
    "cloudflare.com"
}

def is_trusted_url(url: str) -> bool:
    try:
        domain = urlparse(url).netloc.lower()
        return any(domain == d or domain.endswith("." + d) for d in TRUSTED_DOMAINS)
    except:
        return False
