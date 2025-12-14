import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from pathlib import Path

# ------------------ Configuration ------------------ #
ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = os.path.join(ROOT, "models")
MODEL_FILE = os.path.join(MODELS_DIR, "urlnet_cnn.h5")
TOKENIZER_FILE = os.path.join(MODELS_DIR, "tokenizer.json")

# Must match training
MAX_LEN = 150 

# Whitelist (Always keep this as a safety net)
WHITELIST = {'google.com', 'youtube.com', 'facebook.com', 'amazon.com', 'chatgpt.com', 'openai.com'}

def load_artifacts():
    if not os.path.exists(MODEL_FILE) or not os.path.exists(TOKENIZER_FILE):
        print("Error: Models not found. Run train_cnn.py first.")
        return None, None
    
    # Load Model
    print("Loading Neural Network...")
    model = tf.keras.models.load_model(MODEL_FILE)
    
    # Load Tokenizer
    with open(TOKENIZER_FILE, 'r', encoding='utf-8') as f:
        tokenizer_data = json.load(f)
        tokenizer = tokenizer_from_json(json.dumps(tokenizer_data))
        
    return model, tokenizer

def predict_url(model, tokenizer, url):
    # 1. Basic Normalization (Just lowercase)
    # Note: CNN handles missing slashes/http better, but lowercase helps consistency
    u = str(url).lower().strip()
    
    # 2. Whitelist Check
    domain_part = u.replace("https://", "").replace("http://", "").split('/')[0]
    if domain_part in WHITELIST or "www." + domain_part in WHITELIST:
        return 0.00 # 100% Safe
    
    # 3. Preprocess for AI
    sequence = tokenizer.texts_to_sequences([u])
    padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')
    
    # 4. Predict
    score = model.predict(padded, verbose=0)[0][0]
    return score

def main():
    model, tokenizer = load_artifacts()
    if not model: return
    
    print("\n" + "="*40)
    print("   DEEP LEARNING URL SCANNER (CNN)   ")
    print("="*40)
    
    while True:
        url = input("\nURL > ").strip()
        if url.lower() in ['q', 'quit']: break
        if not url: continue
        
        score = predict_url(model, tokenizer, url)
        
        # Thresholding
        if score > 0.75:
            label = "\033[91mMALICIOUS 🚨\033[0m" # Red
        elif score > 0.40:
            label = "\033[93mSUSPICIOUS ⚠️\033[0m" # Orange
        else:
            label = "\033[92mSAFE ✅\033[0m"       # Green
            
        print(f"Result: {label}")
        print(f"Malware Probability: {score:.4f}")

if __name__ == '__main__':
    main()