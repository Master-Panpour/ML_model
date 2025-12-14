# 🛡️ AI Malicious URL Hunter (Char-CNN)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![FastAPI](https://img.shields.io/badge/FastAPI-Production-009688?style=for-the-badge&logo=fastapi)
![License](https://img.shields.io/badge/License-GPLv2-blue?style=for-the-badge)

> **Next-Generation Phishing Defense.** A hybrid security engine combining **Deep Learning (Character-Level CNN)**, **Threat Intelligence APIs**, and **Real-Time Online Learning** to detect malicious URLs that bypass traditional lexical filters.

---

## ⚡ Overview

Traditional URL filters rely on counting dots, slashes, or checking static blacklists. Attackers bypass these easily with URL shortening, obfuscation (`p.a.y.p.a.l`), or by launching on fresh domains.

**This project breaks the "Lexical Ceiling" by treating URLs as biological sequences.** Using a **Convolutional Neural Network (CNN)**, the model reads the raw character structure of a URL to detect hidden patterns of malicious intent (DGA, typosquatting, abnormal entropy) without relying on manual feature engineering.

### 🚀 Key Features

* **🧠 Deep Learning Core:** Uses a **Char-CNN** (1D Convolutions + Max Pooling) to learn features directly from raw text.
* **🛡️ Hybrid Defense Layer:**
    1.  **Whitelist:** Instant pass for trusted giants (Google, Microsoft, etc.).
    2.  **Threat Intel:** APIVoid integration for checking global blacklists.
    3.  **AI Inference:** Deep scan for unknown/zero-day threats.
* **🔄 Online Feedback Loop:** The API accepts feedback (`/feedback`) to retrain the neural network on the fly, adapting to new attack vectors instantly.
* **⚡ High-Performance API:** Built on **FastAPI** for asynchronous, low-latency inference.

---

## 🏗️ System Architecture

```mermaid
graph LR
    A["User / Client"] -->|POST /predict| B("FastAPI Gateway")
    B --> C{"Whitelist?"}
    C -- Yes --> D["SAFE ✅"]
    C -- No --> E{"Blacklist API?"}
    E -- Detected --> F["MALICIOUS 🚨"]
    E -- Clean --> G["🧠 Deep Learning Model"]
    G --> H{"Score > 0.65?"}
    H -- Yes --> F
    H -- No --> D
  ```

  ---
  
## 📂 Project Structure
```Bash

.
├── data/
│   ├── benign_Train.csv       # Training Data (Safe URLs)
│   └── malign_Train.csv       # Training Data (Phishing/Malware URLs)
├── models/
│   ├── urlnet_cnn.h5          # Trained Neural Network Artifact
│   └── tokenizer.json         # Character Tokenizer Dictionary
├── src/
│   ├── main.py                # Core ML Engine (Train/Predict/Update)
│   └── trust.py               # Whitelist Logic
├── app.py                     # FastAPI REST Interface
├── blacklisted_api.py         # APIVoid Integration
├── requirements.txt           # Python Dependencies
└── README.md                  # Documentation
```

---

## 🛠️ Installation
### 1. Clone the Repository
```Bash
git clone [https://github.com/yourusername/ai-url-hunter.git](https://github.com/yourusername/ai-url-hunter.git)
cd ai-url-hunter
```

### 2. Set up Virtual Environment
```Bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies
```Bash
pip install -r requirements.txt
(Ensure tensorflow, fastapi, uvicorn, pandas, requests are in your requirements.txt)
```
### 4. Configuration
- Create a .env file or export your APIVoid key (optional, for hybrid detection):

```Bash
export APIVOID_KEY="your_api_key_here"
```
#### 🚦 Usage
- 1. Train the Neural Network
Before running the API, you must generate the model artifact. The system will auto-balance your dataset and train the Char-CNN.

```Bash
python src/main.py
Input: Parses data/*.csv.

Output: Saves models/urlnet_cnn.h5 and models/tokenizer.json.
```
Note: First run may take a few minutes depending on dataset size.

- 2. Start the API Server
Launch the production-ready FastAPI server.

Bash

uvicorn app:app --reload --host 0.0.0.0 --port 8000
📡 API Documentation
POST /predict
Scan a batch of URLs for threats.

Request:

JSON

{
  "urls": [
    "[http://google.com](http://google.com)",
    "[http://secure-login-paypal.xyz/update](http://secure-login-paypal.xyz/update)"
  ]
}
###### Response:

```JSON```
```
[
  {
    "url": "[http://google.com](http://google.com)",
    "prediction": "benign",
    "confidence": 0.0,
    "source": "Trusted (Whitelist)"
  },
  {
    "url": "[http://secure-login-paypal.xyz/update](http://secure-login-paypal.xyz/update)",
    "prediction": "malicious",
    "confidence": 0.985,
    "source": "AI Prediction"
  }
]
```

POST /feedback
Retrain the model instantly on a misclassified URL (Online Learning).

Request:

```JSON```
```
{
  "urls": ["[http://false-positive-site.com](http://false-positive-site.com)"],
  "correct_labels": [0]
}
```
Labels: 0 = Safe, 1 = Malicious

## 🧠 Model Performance
- Architecture: 1D-CNN with Embedding Layer (32-dim) + Global Max Pooling.

- Accuracy: ~98.5% on validation set (4M rows).

- False Positive Rate: <0.5% (mitigated by whitelist).

- Inference Time: ~15ms per URL (CPU).

## 📜 License
- This project is licensed under the GPL-2.0 License - see the LICENSE file for details.

<div align="center"> <sub>Built with 💀 and ☕ by Master_Panpour</sub> </div>