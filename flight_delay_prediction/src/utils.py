import pandas as pd
from src.logger import logger
from src.exception import CustomException
import sys

def save_processed_data(df, file_path):
    try:
        df.to_csv(file_path, index=False)
        logger.info(f"Data saved to {file_path}")
    except Exception as e:
        raise CustomException(e, sys)
