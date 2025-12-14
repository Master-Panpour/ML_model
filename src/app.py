from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

# Import from the new Deep Learning main.py
from src.main import load_or_train_model, predict_url, update_model_feedback

# Assuming this exists from your previous setup
# If not, remove the import and the blacklist logic block
try:
    from blacklisted_api import check_url_blacklist
except ImportError:
    # Dummy fallback if file is missing
    def check_url_blacklist(url): return {"success": False}

app = FastAPI(
    title="AI Malicious URL Detection API",
    description="Hybrid detection using APIVoid Blacklists + Deep Learning (CNN)",
    version="2.0.0"
)

# Global variables for the AI engine
model = None
tokenizer = None

@app.on_event("startup")
def startup_event():
    """Load the Deep Learning Model and Tokenizer on startup"""
    global model, tokenizer
    # This calls the logic from main.py to load 'urlnet_cnn.h5' and 'tokenizer.json'
    model, tokenizer = load_or_train_model()
    print("API Startup: Deep Learning Model Loaded.")

# --- Pydantic Models ---

class URLRequest(BaseModel):
    urls: List[str]

class FeedbackRequest(BaseModel):
    urls: List[str]
    correct_labels: List[int] # 1 = Malicious, 0 = Safe

class PredictionResult(BaseModel):
    url: str
    prediction: str          # "malicious" or "benign"
    confidence: float        # 0.0 to 1.0
    source: str              # "blacklist", "whitelist", or "ai_model"
    blacklist_info: Optional[dict] = None

# --- Endpoints ---

@app.get("/")
def health():
    return {
        "status": "online", 
        "engine": "Deep Learning (Char-CNN)",
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=List[PredictionResult])
def predict_endpoint(req: URLRequest):
    results = []
    
    for url in req.urls:
        # 1. Check External Blacklist API (APIVoid, etc.)
        bl = check_url_blacklist(url)
        
        # If Blacklist confirms risk, return immediately (Fast Path)
        if bl.get("success") and bl.get("risk_score", 0) >= 50:
            results.append({
                "url": url,
                "prediction": "malicious",
                "confidence": 1.0,
                "source": "blacklist_api",
                "blacklist_info": bl
            })
            continue

        # 2. AI Prediction (Deep Learning)
        # predict_url returns tuple: (score, source_string)
        score, source = predict_url(model, tokenizer, url)
        
        # Threshold Logic
        # > 0.65 is a safe threshold for the CNN to reduce false positives
        is_malicious = score > 0.65
        
        results.append({
            "url": url,
            "prediction": "malicious" if is_malicious else "benign",
            "confidence": float(score), # Convert numpy float to python float
            "source": source,
            "blacklist_info": None
        })

    return results

@app.post("/feedback")
def feedback_endpoint(req: FeedbackRequest):
    """
    Online Learning Endpoint.
    Feeds user corrections back into the Neural Network immediately.
    """
    if len(req.urls) != len(req.correct_labels):
        raise HTTPException(status_code=400, detail="Mismatched URLs and Labels length.")
        
    updated_count = 0
    
    for url, label in zip(req.urls, req.correct_labels):
        # Validate label
        if label not in [0, 1]:
            continue
            
        # Call the update logic from main.py
        update_model_feedback(model, tokenizer, url, label)
        updated_count += 1
        
    return {
        "status": "success", 
        "message": f"Model retrained on {updated_count} URLs."
    }