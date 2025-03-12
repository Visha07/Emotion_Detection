import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
from sklearn.metrics import accuracy_score
# Load original dataset
df = pd.read_csv("data.csv")

# Load corrected data (if exists)
try:
    corrected_df = pd.read_csv("corrected_data.csv", names=["text", "emotion"])
    df = pd.concat([df, corrected_df], ignore_index=True)  # Merge datasets
except FileNotFoundError:
    print("No corrected data found. Training with original dataset.")

# Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["emotion"]

# Train model
model = MultinomialNB()
model.fit(X, y)

# Save model & vectorizer
joblib.dump(model, "model/emotion_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("Model retrained successfully!")
