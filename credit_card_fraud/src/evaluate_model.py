import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.logger import get_logger
from src.exception import CustomException
from src.train_model import split_dataset
from src.utils import load_model

logger = get_logger(__name__)

def evaluate_model(data_path, model_path):
    """Evaluate the trained model using some metrics"""
    try:
        _, X_test, _, y_test = split_dataset(data_path)

        model = load_model(model_path)

        #make prediction
        y_pred = model.predict(X_test)

        #calculate accuracy score
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Model Accuracy: {accuracy:.2f}")

        # Display classification report
        print("Classification Report:\n", classification_report(y_test, y_pred))

        #plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot = True, xticklabels = ["Non-Fraud", "Fraud"], yticklabels = ["Non-Fraud", "Fraud"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")

    except Exception as e:
        raise CustomException (f"Error in evaluating model: {e}", sys)
    
if __name__ == "__main__":
    evaluate_model("data/processed/cleanedcredit.csv", "model/model.pkl")
