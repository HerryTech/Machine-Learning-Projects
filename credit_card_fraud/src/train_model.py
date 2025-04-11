import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from src.logger import get_logger
from src.exception import CustomException
from src.utils import load_data

logger = get_logger(__name__)

def split_dataset(data_path):
    """Split data into training and testing set"""
    df = load_data(data_path)
    X = 
    logger.info("Data splitting starting...")
    X_train, y