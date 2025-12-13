from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from main import load_or_train_model, predict, update_model
from blacklisted_api import check_url_blacklist

app = FastAPI(
    title="Malicious URL Detection API",
    description="Check URLs with blacklist API + ML model",
    version="1.0.0"
)

model, scaler = load_or_train_model()

class URLRequest(BaseModel):
    urls: List[str]

class FeedbackRequest(BaseModel):
    urls: List[str]
    correct_labels: List[int]

@app.get("/")
def health():
    return {"message":"Service running"}

@app.post("/predict")
def predict_urls(req: URLRequest):
    results = []
    for url in req.urls:
        rec = {"url":url}

        bl = check_url_blacklist(url)
        rec["blacklist"]=bl

        if bl.get("success") and bl.get("risk_score") is not None:
            if bl["risk_score"]>=50:
                rec["prediction"]="malicious (blacklist)"
                results.append(rec)
                continue

        p = predict(model, scaler, [url])[0]
        rec["prediction"] = "malicious" if p==1 else "benign"
        results.append(rec)
    return results

@app.post("/feedback")
def feedback_fb(req: FeedbackRequest):
    if len(req.urls)!=len(req.correct_labels):
        raise HTTPException(status_code=400, detail="Mismatch length")
    update_model(model, scaler, req.urls, req.correct_labels)
    return {"status":"model updated"}
