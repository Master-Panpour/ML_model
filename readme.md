project:
  name: "Malicious URL Detection API"
  description: "A FastAPI‑based service to detect whether a URL is malicious or benign using blacklist API checks with caching, a machine learning model, and online learning."
features:
  - "Blacklist API checks with caching"
  - "Machine learning classifier trained on a Hugging Face dataset"
  - "SSL certificate validity & features used in ML model"
  - "Online learning (incremental updates) using user feedback"
  - "FastAPI with interactive API docs"
  - "REST API integration"

structure:
  - app.py: "FastAPI application"
  - main.py: "Model training, feature extraction, prediction"
  - blacklisted_api.py: "Blacklist API integration with caching"
  - urls_cache.db: "SQLite cache for blacklist responses"
  - malicious_url_model.pkl: "Persisted machine learning model"
  - malicious_url_scaler.pkl: "Persisted feature scaler"
  - requirements.txt: "Python dependencies"
  - README.md: "Project documentation"

requirements:
  python_version: "3.8+"
  dependencies:
    - fastapi
    - uvicorn
    - scikit-learn
    - pandas
    - joblib
    - requests
    - requests-cache
    - datasets
    - tldextract

configuration:
  blacklist_api:
    file: "blacklisted_api.py"
    variable: "APIVOID_API_KEY"
    replace_with: "Your API key for the blacklist service"

run:
  install_dependencies: "pip install -r requirements.txt"
  start_server: "uvicorn app:app --reload"
  base_url: "http://127.0.0.1:8000"
  docs: "http://127.0.0.1:8000/docs"

api_endpoints:
  health_check:
    method: "GET"
    path: "/"
    description: "Health check endpoint"
    example_response:
      message: "Service running"

  predict:
    method: "POST"
    path: "/predict"
    request_body:
      urls:
        type: "array"
        items: "string"
    response_example:
      - url: "http://example.com/login"
        blacklist:
          success: true
          risk_score: 10
          blacklist_engines: {}
        prediction: "benign"
      - url: "https://malicious.ru/test"
        blacklist:
          success: true
          risk_score: 80
          blacklist_engines: {}
        prediction: "malicious (blacklist)"

  feedback:
    method: "POST"
    path: "/feedback"
    request_body:
      urls:
        type: "array"
        items: "string"
      correct_labels:
        type: "array"
        items: "integer"
    response:
      status: "model updated"

cache:
  file: "urls_cache.db"
  description: "Persistent SQLite cache for blacklist results"

workflow:
  blacklist_flow:
    - step: "Check cache for URL"
    - step: "If cached and valid, return cached result"
    - step: "If not cached, call external blacklist API"
    - step: "Save response to cache"

  ml_flow:
    - step: "Extract features from URL"
    - step: "Run model prediction"
    - step: "Return prediction"

  feedback_flow:
    - step: "Receive user verified labels"
    - step: "Update model via online learning"
    - step: "Persist updated model"

dataset:
  name: "Anvilogic/URL‑Guardian‑Dataset"
  source: "Hugging Face"

future_improvements:
  - "Add WHOIS and domain age features"
  - "Add additional threat intelligence integrations"
  - "Containerize with Docker"
  - "Add authentication and rate‑limiting"

contributing:
  steps:
    - "Fork the repository"
    - "Create a feature branch"
    - "Commit changes"
    - "Push and create a pull request"

license: "MIT License"

contact:
  author: "Master_Panpour"
  platforms:
    - "LinkedIn"
    - "GitHub"
    - "Email"
