# 🔐 Malicious URL Detection API

![GitHub Repo Size](https://img.shields.io/github/repo-size/Master-Panpour/ML_model)
![GitHub Stars](https://img.shields.io/github/stars/Master-Panpour/ML_model?style=social)
![GitHub License](https://img.shields.io/github/license/Master-Panpour/ML_model)

A **FastAPI‑based service** to detect whether a URL is **malicious or benign** using:

- 🌐 Blacklist API checks (with caching)
- 🤖 Machine Learning model trained on a Hugging Face dataset
- 📊 URL structural & SSL features
- 🔄 Incremental learning via user feedback

---

## 📌 Table of Contents

1. [Overview](#overview)  
2. [Features](#features)  
3. [How It Works](#how-it-works)  
4. [Quick Start](#quick-start)  
5. [API Endpoints](#api-endpoints)  
6. [Online Learning](#online-learning)  
7. [Tech Stack](#tech-stack)  
8. [Security](#security)  
9. [Contributing](#contributing)  
10. [License](#license)

---

## 📌 Overview

This project provides an API to classify URLs as malicious or benign through a blend of external threat intelligence and machine learning. The model learns over time using user feedback.

---

## 🚀 Features

- ✅ Real‑time blacklist reputation checks with caching  
- ✅ ML classification using lexical/structural and SSL certificate features  
- ⚡ FastAPI server with auto‑generated documentation  
- 🧠 Incremental learning using user‑provided feedback  
- 📦 Easy integration in other projects

---

## 🧠 How It Works

### 1. Blacklist API Check

Calls a threat intelligence API to get risk scores and flags for known malicious URLs, with caching to reduce redundant API requests.

### 2. Feature Extraction

Extracts:
- URL length, digit count, punctuation  
- Subdomain count
- SSL certificate validity and expiry

### 3. ML Prediction

A classifier trained on a public dataset predicts whether a URL is malicious.

### 4. Online Learning

Users can correct predictions and update the model over time.

---

## ⚙️ Quick Start

### 1. Clone the Repo

```bash
git clone https://github.com/Master-Panpour/ML_model.git
cd ML_model
```
### 2. Create a .env File
ini
```bash
APIVOID_API_KEY=your_apiovoid_api_key
```
- 🔒 Add .env to .gitignore so your API key isn’t committed.

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Start the API
```bash
uvicorn app:app --reload
```
- Visit the docs:
```bash
🔗 http://127.0.0.1:8000/docs
```
- 📡 API Endpoints
🟢 GET /
Returns API health:

json
Copy code
{ "message": "Service running" }
🟡 POST /predict
Predict malicious/benign for a list of URLs.

```bash
Request -json
{
  "urls": [
    "http://example.com/login",
    "https://suspicious-site.ru"
  ]
}
```
Response

json
Copy code
[
  {
    "url": "http://example.com/login",
    "blacklist": {
      "success": true,
      "risk_score": 5,
      "blacklist_engines": {}
    },
    "prediction": "benign"
  },
  {
    "url": "https://suspicious-site.ru",
    "blacklist": {
      "success": true,
      "risk_score": 90,
      "blacklist_engines": {}
    },
    "prediction": "malicious (blacklist)"
  }
]
🔵 POST /feedback
Send corrected labels to update the model.

Request

json
Copy code
{
  "urls": ["http://example.com/login"],
  "correct_labels": [0]
}
Response

json
Copy code
{ "status": "model updated" }
🔄 Online Learning
After initial training, the model can continue learning using user‑verified corrections, improving future predictions.

---

## 🧰 Tech Stack
 - Component Technology
 - API Framework	FastAPI
 - ML Library	scikit‑learn
 - Caching	requests‑cache
 - ASGI Server	Uvicorn
 - Dataset	Hugging Face

## 🛡️ Security & Best Practices
 - 🗝️ Store API keys via environment variables
 - 🚫 Never commit sensitive credentials
 - 📉 Use cache to reduce external API usage
 - 🚀 Keep dependencies up to date

---

## 👥 Contributing
### To contribute:

 - Fork the repository
 - Create a branch (git checkout -b feature/xyz)
 - Commit (git commit -m "Add feature")
 - Push (git push origin feature/xyz)
 - Open a pull request

---

## 📜 License
 - This project is licensed under the GPL-2.0 License.

---

# ⭐ If you find this project useful, please consider giving it a star!