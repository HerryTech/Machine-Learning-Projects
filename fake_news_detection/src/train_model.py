import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from src.utils import save_model
from src.logger import get_logger
from src.exception import CustomException
import sys

logger = get_logger(__name__)

def train_model(data_path, model_path, vectorizer_path):
    """Train a logistic regression model for fake news detection."""
    try:
        df = pd.read_csv(data_path)
        X = df["text"]
        y = df["label"]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logger.info(f"Training data size: {len(X_train)}, Testing data size: {len(X_test)}")

        # Convert text to TF-IDF features
        vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        logger.info(f"TF-IDF feature shape: {X_train_tfidf.shape}")

        # Train model
        model = LogisticRegression(class_weight='balanced', C=0.5, random_state=42)
        model.fit(X_train_tfidf, y_train)

        # Save model and vectorizer
        save_model(model, model_path)
        save_model(vectorizer, vectorizer_path)

        logger.info("Model training completed and saved successfully.")

    except Exception as e:
        raise CustomException(f"Error in training model: {e}", sys)

if __name__ == "__main__":
    train_model("data/processed/cleaned_news.csv", "model/model.pkl", "model/vectorizer.pkl")
