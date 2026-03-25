# utils/text_utils.py

import joblib
import string
import re
import os

# Load model and vectorizer (ensure paths are correct)
model_path = os.path.join("models", "text_model.pkl")
vectorizer_path = os.path.join("models", "text_vectorizer.pkl")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

def clean_text(text):
    """
    Preprocesses the input text:
    - Lowercases
    - Removes URLs, mentions, hashtags
    - Removes punctuation and extra spaces
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'@\w+|#', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def predict_text(text):
    """
    Predicts if the input text is harmful or safe.
    Returns a user-friendly label.
    """
    clean = clean_text(text)
    vec = vectorizer.transform([clean])
    prediction = model.predict(vec)[0]
    return "⚠️ Harmful" if prediction == 1 else "✅ Safe"
