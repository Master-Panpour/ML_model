import numpy as np
import re
from urllib.parse import urlparse
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from datasets import load_dataset
import joblib
import os
import ssl
import socket
from datetime import datetime

MODEL_FILE = "malicious_url_model.pkl"
SCALER_FILE = "malicious_url_scaler.pkl"

# Extract numeric features from raw URLs
def has_ip_in_url(url):
    netloc = urlparse(url).netloc
    return int(bool(re.match(r"\d+\.\d+\.\d+\.\d+", netloc)))

def count_subdomains(url):
    parts = urlparse(url).netloc.split('.')
    return max(0, len(parts) - 2)

def extract_ssl_info(host):
    try:
        context = ssl.create_default_context()
        with socket.create_connection((host, 443), timeout=3) as sock:
            with context.wrap_socket(sock, server_hostname=host) as ssock:
                cert = ssock.getpeercert()
                exp_date = datetime.strptime(cert['notAfter'], "%b %d %H:%M:%S %Y %Z")
                days_left = max(0, (exp_date - datetime.utcnow()).days)
                return 1, days_left
    except Exception:
        return 0, 0

def extract_features(urls):
    feats = []
    for url in urls:
        u = str(url)
        parsed = urlparse(u)
        length = len(u)
        dots = u.count(".")
        digits = len(re.findall(r"[0-9]", u))
        specials = len(re.findall(r"[^a-zA-Z0-9:/._-]", u))
        ip_flag = has_ip_in_url(u)
        sub_count = count_subdomains(u)
        host = parsed.netloc.split(":")[0]
        cert_valid, days_left = extract_ssl_info(host)
        feats.append([length, dots, digits, specials, ip_flag, sub_count, cert_valid, days_left])
    return np.array(feats)

def load_or_train_model():
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        return model, scaler

    print("Training model from Hugging Face dataset…")
    dataset = load_dataset("Anvilogic/URL-Guardian-Dataset")
    df = dataset["train"].to_pandas()
    urls = df["value"].astype(str)
    labels = df["label"].astype(int)

    X = extract_features(urls)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = SGDClassifier(loss="log")
    model.partial_fit(X_scaled, labels, classes=np.unique(labels))

    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    print("Model trained and saved.")
    return model, scaler

def predict(model, scaler, url_list):
    feats = extract_features(url_list)
    scaled = scaler.transform(feats)
    preds = model.predict(scaled)
    return preds

def update_model(model, scaler, url_list, correct_labels):
    feats = extract_features(url_list)
    scaled = scaler.transform(feats)
    model.partial_fit(scaled, correct_labels)
    joblib.dump(model, MODEL_FILE)
    print("Model updated with feedback!")
