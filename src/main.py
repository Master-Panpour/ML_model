import os
import sys
import json
import numpy as np
import pandas as pd
import requests
import joblib
from pathlib import Path
from datetime import datetime

# Deep Learning Imports
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, Input
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Custom Imports
try:
    from src.trust import is_trusted_url
except ImportError:
    # Fallback if src.trust is missing
    def is_trusted_url(url): return False

# ---------------- Configuration ---------------- #

ROOT = Path(__file__).resolve().parent
DATA_DIR = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")
MALIGN_CSV = os.path.join(DATA_DIR, "malign_Train.csv")
BENIGN_CSV = os.path.join(DATA_DIR, "benign_Train.csv")

MODEL_FILE = os.path.join(MODELS_DIR, "urlnet_cnn.h5")
TOKENIZER_FILE = os.path.join(MODELS_DIR, "tokenizer.json")

APIVOID_KEY = os.getenv("APIVOID_KEY")

# Hyperparameters
MAX_LEN = 150       # First 150 chars of URL
EMBEDDING_DIM = 32
VOCAB_SIZE = 0      # Will be set dynamically

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


# ---------------- 1. Data & Tokenizer ---------------- #

def load_data(sample_size=200000):
    """Loads and balances the dataset for training."""
    print("Loading datasets...")
    if not os.path.exists(MALIGN_CSV) or not os.path.exists(BENIGN_CSV):
        raise FileNotFoundError("Training CSVs not found in data/ folder.")

    mal = pd.read_csv(MALIGN_CSV, usecols=['url'])
    mal['label'] = 1
    ben = pd.read_csv(BENIGN_CSV, usecols=['url'])
    ben['label'] = 0

    # Balance & Shuffle
    df = pd.concat([
        mal.sample(n=min(len(mal), sample_size), random_state=42),
        ben.sample(n=min(len(ben), sample_size), random_state=42)
    ], ignore_index=True).sample(frac=1, random_state=42)
    
    return df['url'].astype(str).values, df['label'].values

def get_tokenizer(urls=None, train=False):
    """
    Loads existing tokenizer or trains a new one.
    """
    if os.path.exists(TOKENIZER_FILE) and not train:
        with open(TOKENIZER_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return tokenizer_from_json(json.dumps(data))
    
    if urls is None:
        raise ValueError("No URLs provided to train tokenizer!")

    print("Training new Character Tokenizer...")
    # char_level=True is the key to fixing 'slash' and 'obfuscation' bugs
    tokenizer = Tokenizer(char_level=True, lower=True)
    tokenizer.fit_on_texts(urls)
    
    # Save
    with open(TOKENIZER_FILE, 'w', encoding='utf-8') as f:
        f.write(tokenizer.to_json())
        
    return tokenizer

def preprocess_urls(tokenizer, urls):
    """Converts URL strings to padded integer sequences."""
    seqs = tokenizer.texts_to_sequences(urls)
    return pad_sequences(seqs, maxlen=MAX_LEN, padding='post', truncating='post')


# ---------------- 2. Model Logic (CNN) ---------------- #

def build_model(vocab_size):
    """Defines the Character-Level CNN."""
    model = Sequential([
        Input(shape=(MAX_LEN,)),
        Embedding(input_dim=vocab_size + 1, output_dim=EMBEDDING_DIM),
        Conv1D(filters=128, kernel_size=4, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def load_or_train_model():
    """
    Loads the Deep Learning model if it exists, otherwise trains it.
    """
    if os.path.exists(MODEL_FILE) and os.path.exists(TOKENIZER_FILE):
        print("Loading existing Deep Learning model...")
        return load_model(MODEL_FILE), get_tokenizer(train=False)

    print("No model found. Starting training pipeline...")
    
    # 1. Load Data
    urls, y = load_data()
    
    # 2. Train Tokenizer
    tokenizer = get_tokenizer(urls, train=True)
    vocab_size = len(tokenizer.word_index)
    
    # 3. Vectorize
    X = preprocess_urls(tokenizer, urls)
    
    # 4. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    # 5. Build & Train
    model = build_model(vocab_size)
    print("Training CNN (This may take a few minutes)...")
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=1024, verbose=1)
    
    # 6. Save
    model.save(MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")
    
    return model, tokenizer


# ---------------- 3. Prediction Pipeline ---------------- #

def check_apivoid(url):
    """Checks URL against APIVoid (Hybrid Security)."""
    if not APIVOID_KEY: return None
    try:
        api_url = f"https://endpoint.apivoid.com/urlrep/v1/pay-as-you-go/?key={APIVOID_KEY}&url={url}"
        res = requests.get(api_url, timeout=3)
        if res.status_code == 200:
            stats = res.json().get("data", {}).get("report", {}).get("blacklists", {}).get("detections", 0)
            return 1 if stats > 0 else 0
    except:
        pass
    return None

def predict_url(model, tokenizer, url):
    """
    The Master Prediction Function.
    Flow: Whitelist -> API -> Deep Learning
    """
    u = str(url).strip().lower()
    
    # 1. Whitelist Check (Instant Safe)
    if is_trusted_url(u):
        return 0.0, "Trusted (Whitelist)"

    # 2. API Check (Instant Malicious)
    api_result = check_apivoid(u)
    if api_result == 1:
        return 1.0, "Malicious (APIVoid)"

    # 3. Deep Learning Scan
    # Preprocess
    seq = preprocess_urls(tokenizer, [u])
    
    # Predict
    score = model.predict(seq, verbose=0)[0][0]
    return score, "AI Prediction"


# ---------------- 4. Feedback Loop ---------------- #

def update_model_feedback(model, tokenizer, url, correct_label):
    """
    Online Learning: Updates the CNN with a single new example.
    correct_label: 1 for Malicious, 0 for Safe
    """
    print(f"Retraining model on correction: {url} -> {correct_label}")
    
    X_new = preprocess_urls(tokenizer, [url])
    y_new = np.array([correct_label])
    
    # Perform a single gradient update
    model.fit(X_new, y_new, epochs=1, verbose=0)
    model.save(MODEL_FILE)
    print("Model updated and saved.")


# ---------------- Main Execution ---------------- #

def main():
    # Load Engine
    model, tokenizer = load_or_train_model()
    
    print("\n" + "="*40)
    print("   AI URL DEFENSE SYSTEM (Char-CNN)   ")
    print("="*40)
    print("Type 'q' to quit.")
    
    while True:
        url = input("\nURL > ").strip()
        if url.lower() in ['q', 'quit']: break
        if not url: continue
        
        # Predict
        score, source = predict_url(model, tokenizer, url)
        
        # Display
        is_malicious = score > 0.65
        label = "\033[91mMALICIOUS 🚨\033[0m" if is_malicious else "\033[92mSAFE ✅\033[0m"
        
        print(f"Result: {label}")
        print(f"Confidence: {score:.4f}")
        print(f"Source: {source}")
        
        # Feedback Option
        # If user disagrees, they can type 'wrong' to correct the model
        if (is_malicious and input("Is this result correct? (y/n) > ").lower() == 'n'):
            correct_label = 0 if is_malicious else 1
            update_model_feedback(model, tokenizer, url, correct_label)

if __name__ == '__main__':
    main()