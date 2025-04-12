import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from src.logger import get_logger
from src.exception import CustomException
from src.utils import load_data, save_model

logger = get_logger(__name__)

def split_dataset(data_path):
    """Split data into training and testing set"""
    try:
        df = load_data(data_path)
        X = df.drop(columns = ["Class"])
        y = df["Class"]
        logger.info("Data splitting starting...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
        logger.info(f"Data splitting successful! - Total training set: {len(X_train)}")
        return X_train, X_test, y_train, y_test

    except Exception as e:
        raise CustomException(f"Error in splitting dataset: {e}", sys)
    
def handle_imbalance(X_train, y_train):
    """Handle dataset imbalance using SMOTE"""
    try:
        logger.info("Handling imbalance begins...")
        smote = SMOTE(random_state = 42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        logger.info(f"Training data after SMOTE: {len(X_train_smote)}")
        return X_train_smote, y_train_smote
    
    except Exception as e:
        raise CustomException(f"Error handling imbalance: {e}", sys)
    
def train_model(X_train_smote, y_train_smote):
    """Train Logistic Regression model for credit card fraud detection"""
    try:
        logger.info("Logistic Regression starting...")
        model = LogisticRegression()
        model.fit(X_train_smote, y_train_smote)
        logger.info("Modeling completed!")
        
        #save model


    except Exception as e:
        raise CustomException(f"Error training model: {e}", sys)
    
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = split_dataset("data/processed/cleanedcredit.csv")
    X_train_smote, y_train_smote = handle_imbalance(X_train, y_train)
    train_model(X_train_smote, y_train_smote)