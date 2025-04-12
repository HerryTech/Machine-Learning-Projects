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

def evaluate_model(data_path, model, model_path):
    """Evaluate the trained model using some metrics"""
    try:
        X_test, y_test = split_dataset(data_path)

        load_model(model, model_path)

        #make prediction
        y_pred = model

        #accuracy score
        accuracy = accuracy_score()

    except Exception as e:
        raise CustomException (f"Error in evaluating model: {e}", sys)
    
if __name__ == "__main__":
    
