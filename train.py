import pandas as pd
import joblib
from main import extract_features
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

# Load your local CSV dataset
df = pd.read_csv("train.csv")  # replace with your CSV file

# Extract URLs and labels
urls = df["url"].values
labels = df["label"].values

# Convert URLs to features
X = extract_features(urls)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = SGDClassifier(loss="log")
model.fit(X_scaled, labels)

# Save model + scaler
joblib.dump(model, "malicious_url_model.pkl")
joblib.dump(scaler, "malicious_url_scaler.pkl")

print("Model training and saving completed!")
