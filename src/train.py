import os
import sys
import numpy as np
import pandas as pd
import json
import joblib
from pathlib import Path

# Deep Learning Imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, Input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# ------------------ Configuration ------------------ #
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")
MALIGN_CSV = os.path.join(DATA_DIR, "malign_Train.csv")
BENIGN_CSV = os.path.join(DATA_DIR, "benign_Train.csv")

# Artifact paths
MODEL_FILE = os.path.join(MODELS_DIR, "urlnet_cnn.h5") # The Neural Network
TOKENIZER_FILE = os.path.join(MODELS_DIR, "tokenizer.json") # The "Dictionary"

# Hyperparameters (Tuned for URLNet)
MAX_LEN = 150       # Look at first 150 chars of URL
EMBEDDING_DIM = 32  # Vector size per character
BATCH_SIZE = 1024   # Process 1024 URLs at a time (Faster)
EPOCHS = 5          # Number of training passes

os.makedirs(MODELS_DIR, exist_ok=True)

# ------------------ Data Loading ------------------ #
def load_and_preprocess():
    print("Loading datasets...")
    # Load only necessary columns
    mal = pd.read_csv(MALIGN_CSV, usecols=['url'])
    mal['label'] = 1
    
    ben = pd.read_csv(BENIGN_CSV, usecols=['url'])
    ben['label'] = 0
    
    # Balance Dataset (500k each for speed, or remove .sample to use all)
    # Using 4M rows might require 32GB+ RAM. We start with a strong subset.
    SAMPLE_SIZE = 500000 
    print(f"Balancing to {SAMPLE_SIZE} samples per class...")
    
    df = pd.concat([
        mal.sample(n=min(len(mal), SAMPLE_SIZE), random_state=42),
        ben.sample(n=min(len(ben), SAMPLE_SIZE), random_state=42)
    ], ignore_index=True)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Force string type
    urls = df['url'].astype(str).values
    y = df['label'].values
    
    return urls, y

# ------------------ Tokenization ------------------ #
def train_tokenizer(urls):
    print("Training Character Tokenizer...")
    # char_level=True means we split by letter 'a', 'b', '.', '/'
    # This is CRITICAL for catching obfuscation like 'p.a.y.p.a.l'
    tokenizer = Tokenizer(char_level=True, lower=True)
    tokenizer.fit_on_texts(urls)
    
    # Save tokenizer for prediction script
    tokenizer_json = tokenizer.to_json()
    with open(TOKENIZER_FILE, 'w', encoding='utf-8') as f:
        f.write(tokenizer_json)
    print(f"Tokenizer saved to {TOKENIZER_FILE}")
    
    return tokenizer

# ------------------ Model Architecture ------------------ #
def build_urlnet(vocab_size, max_len):
    """
    Builds a Conv1D Neural Network (Simpler version of URLNet)
    """
    model = Sequential([
        Input(shape=(max_len,)),
        # 1. Embedding: Converts integers to vectors
        Embedding(input_dim=vocab_size + 1, output_dim=EMBEDDING_DIM),
        
        # 2. Convolution: Scans for patterns (like "login", ".exe", "wp-admin")
        Conv1D(filters=128, kernel_size=4, activation='relu'),
        
        # 3. Pooling: Grabs the strongest signal from the scan
        GlobalMaxPooling1D(),
        
        # 4. Dense Layers: Decision making
        Dense(64, activation='relu'),
        Dropout(0.5), # Prevents overfitting
        Dense(1, activation='sigmoid') # Output: 0-1 probability
    ])
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# ------------------ Main Pipeline ------------------ #
def train_deep_model():
    # 1. Load Data
    urls, y = load_and_preprocess()
    
    # 2. Tokenize (Convert text to numbers)
    tokenizer = train_tokenizer(urls)
    sequences = tokenizer.texts_to_sequences(urls)
    X = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    
    vocab_size = len(tokenizer.word_index)
    print(f"Vocabulary Size: {vocab_size} unique characters")
    
    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    # 4. Build Model
    print("Building Deep Learning Model...")
    model = build_urlnet(vocab_size, MAX_LEN)
    model.summary()
    
    # 5. Train
    print("Starting Training (This uses CPU/GPU)...")
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )
    
    # 6. Save
    model.save(MODEL_FILE)
    print(f"SUCCESS: Model saved to {MODEL_FILE}")

if __name__ == '__main__':
    train_deep_model()