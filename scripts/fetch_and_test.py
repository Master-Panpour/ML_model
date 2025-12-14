import sys
from pathlib import Path
import requests
import joblib

# Ensure project root is importable when running this script directly
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.main import extract_features

MODEL = "models/malicious_url_model.pkl"
SCALER = "models/malicious_url_scaler.pkl"

SEED_URLS = [
    "https://example.com",
    "https://github.com",
    "https://www.google.com",
    "https://login.microsoftonline.com",
    "https://www.microsoft.com"
]


def follow_redirect(url, timeout=5):
    try:
        r = requests.head(url, allow_redirects=True, timeout=timeout)
        final = r.url
        if not final:
            r = requests.get(url, allow_redirects=True, timeout=timeout)
            final = r.url
        return final
    except Exception:
        return url


def load_model():
    model = joblib.load(MODEL)
    scaler = joblib.load(SCALER)
    return model, scaler


def test_urls(urls):
    model, scaler = load_model()
    results = []
    for u in urls:
        final = follow_redirect(u)
        feats = extract_features([final], check_ssl=False)
        X = scaler.transform(feats)
        pred = model.predict(X)[0]
        try:
            score = float(model.decision_function(X)[0])
        except Exception:
            score = None
        results.append((u, final, int(pred), score))
    return results


if __name__ == "__main__":
    res = test_urls(SEED_URLS)
    for orig, final, pred, score in res:
        print(f"Orig: {orig}")
        print(f"Final: {final}")
        print(f"Predicted: {pred}")
        print(f"Score: {score}\n")
