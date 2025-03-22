import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.utils import load_model
from src.logger import get_logger
from src.exception import CustomException
import sys

logger = get_logger(__name__)

def evaluate_model(data_path, model_path, vectorizer_path):
    """Evaluate the trained model and display performance metrics."""
    try:
        df = pd.read_csv(data_path)
        X = df["text"]
        y = df["label"]

        # Load trained model and vectorizer
        model = load_model(model_path)
        vectorizer = load_model(vectorizer_path)

        # Transform text data
        X_tfidf = vectorizer.transform(X)

        # Make predictions
        y_pred = model.predict(X_tfidf)

        # Calculate accuracy
        accuracy = accuracy_score(y, y_pred)
        logger.info(f"Model Accuracy: {accuracy:.2f}")

        # Display classification report
        print("Classification Report:\n", classification_report(y, y_pred))

        # Plot confusion matrix
        cm = confusion_matrix(y, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()

    except Exception as e:
        raise CustomException(f"Error in evaluating model: {e}", sys)

if __name__ == "__main__":
    evaluate_model("../data/processed/cleaned_news.csv", "../model/model.pkl", "../model/vectorizer.pkl")
