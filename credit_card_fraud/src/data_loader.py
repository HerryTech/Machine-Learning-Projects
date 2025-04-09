import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler
from src.logger import get_logger
from src.exception import CustomException
from src.utils import save_data

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

        return df

    except Exception as e:
        raise CustomException (f"Error loading dataset: {e}", sys)
    
def preprocessing_and_save(df, output_path):
    """Scale Amount and Time Columns and saves the data"""
    try:
        logger.info("Starting feature scaling...")
        feature_to_scale = ["Amount", "Time"]
        scale = StandardScaler()
        df[feature_to_scale] = scale.fit_transform(df[feature_to_scale])

        #save cleaned data
        save_data(df, output_path)

    except Exception as e:
        raise CustomException(f"Error scaling and saving the data: {e}", sys)
    
if __name__ == "__main__":
    df = load_clean_data("data/raw/creditcard.csv")
    preprocessing_and_save(df, "data/processed/cleanedcredit.csv")



