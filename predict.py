import joblib
import neattext.functions as nfx
import pandas as pd

# Load model and vectorizer
loaded_model = joblib.load("model/emotion_model.pkl")
loaded_vectorizer = joblib.load("model/vectorizer.pkl")

def predict_emotion(text):
    text_clean = nfx.remove_stopwords(nfx.remove_puncts(text))
    text_tfidf = loaded_vectorizer.transform([text_clean])
    emotion_label = loaded_model.predict(text_tfidf)[0]  # Predicted emotion
    
    print("Raw Prediction Label:", emotion_label)
    
    # Ask user if prediction is correct
    is_correct = input(f"Is this correct? (y/n): ").strip().lower()
    
    if is_correct == "n":
        correct_emotion = input("Enter the correct emotion (happy, sad, angry, etc.): ").strip().lower()
        save_correction(text, correct_emotion)  # Save for retraining
        return correct_emotion
    else:
        return emotion_label

def save_correction(text, correct_label):
    """Save incorrect predictions with corrected labels for retraining."""
    df = pd.DataFrame([[text, correct_label]], columns=["text", "emotion"])
    df.to_csv("corrected_data.csv", mode="a", header=False, index=False)  # Append to file

# Test input
if __name__ == "__main__":
    text = input("Enter a sentence: ")
    predicted_emotion = predict_emotion(text)
    print("Final Predicted Emotion:", predicted_emotion)
