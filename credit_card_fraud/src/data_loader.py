import pandas as pd
import numpy as np
import sys
from src.logger import get_logger
from src.exception import CustomException

logger = get_logger(__name__)

def load_clean_data(data_path):
    """load and clean the dataset"""
    try:
        logger.info("Loading dataset...")
        df = pd.read_csv(data_path)

        #remove missing values
        df.dropna(inplace = True)

        #remove duplicates
        df.drop_duplicates(inplace = True)
        logger.info(f"Dataset loaded and cleaned successfully! Total Dataset: {len(df)}")

        #save cleaned data
        save_data()

        return df

    except Exception as e:
        raise CustomException (f"Error loading dataset: {e}", sys)

