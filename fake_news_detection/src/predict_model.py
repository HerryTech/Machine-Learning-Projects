import sys
import pandas as pd
from src.utils import load_model
from src.logger import get_logger
from src.exception import CustomException

logger = get_logger(__name__)

def predict_news(text, model_path, vectorizer_path):
    """Predict whether a given news text is Fake or Real."""
    try:
        # Load trained model and vectorizer
        model = load_model(model_path)
        vectorizer = load_model(vectorizer_path)

        # Transform input text
        text_tfidf = vectorizer.transform([text])

        # Make prediction
        prediction = model.predict(text_tfidf)[0]

        # Return result
        result = "Real News" if prediction == 1 else "Fake News"
        logger.info(f"Prediction: {result}")
        return result

    except Exception as e:
        raise CustomException(f"Error in predicting news: {e}", sys)

if __name__ == "__main__":
    sample_text = input("Enter news text to predict: ")
    result = predict_news(sample_text, "model/model.pkl", "model/vectorizer.pkl")
    print(f"\nPredicted News Category: {result}")







