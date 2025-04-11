import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from src.logger import get_logger
from src.exception import CustomException
from src.utils import load_data

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

    except Exception as e:
        raise CustomException(f"Error in splitting dataset: {e}", sys)