import pandas as pd
import sys
from src.logger import get_logger
from src.exception import CustomException

logger = get_logger(__name__)

def load_flight_data(file_path):
    """Function to load data"""
    try:
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Shape of data: {df.shape}")
        return df
    except Exception as e:
        raise CustomException(e, sys)
    