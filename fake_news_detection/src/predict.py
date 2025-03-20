import joblib
import os

# Get the absolute path of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Use the correct relative path to the model directory
model_path = os.path.join(current_dir, "../model/model.pkl")
vectorizer_path = os.path.join(current_dir, "../model/vectorizer.pkl")

# Load trained model and vectorizer
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

def predict_news(news_text):
    '''Predict whether a given news article is Fake or Real.'''
    text_vectorised = vectorizer.transform([news_text])
    prediction = model.predict(text_vectorised)[0]
    return "Fake News" if prediction == 0 else "Real News"

if __name__ == "__main__":
    news_text = input("Jesus is Lord")
    print("\nPrediction:", predict_news(news_text))

