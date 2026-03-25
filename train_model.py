import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import os

# Create models folder if not exists
os.makedirs("models", exist_ok=True)

# Load preprocessed dataset
df = pd.read_csv("datasets/text_dataset.csv")

# Features and labels
X = df["comment_text"]
y = df["label"]

# Convert text to numerical vectors
vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model and vectorizer inside models/
joblib.dump(model, "models/text_model.pkl")
joblib.dump(vectorizer, "models/text_vectorizer.pkl")

print("✅ Model trained and saved as 'models/text_model.pkl' and 'models/text_vectorizer.pkl'")
