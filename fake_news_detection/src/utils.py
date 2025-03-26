import pandas as pd
import joblib
import os
import sys
from src.logger import get_logger 
from src.exception import CustomException  

# Initialize logger
logger = get_logger(__name__)

def save_data(df: pd.DataFrame, output_path: str):
    """Saves a DataFrame to a CSV file, ensuring the directory exists."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Data successfully saved to {output_path}")
    except Exception as e:
        raise CustomException(f"Error saving data to {output_path}: {e}", sys)

def save_model(model, filename):
    """Save the trained model to a file."""
    try:
        joblib.dump(model, filename)
        logger.info(f"Model successfully saved to {filename}")
    except Exception as e:
        raise CustomException(f"Error saving model to {filename}: {e}", sys)

def load_model(filename):
    """Load a saved model from a file."""
    try:
        return joblib.load(filename)
    except Exception as e:
        raise CustomException(f"Error loading model from {filename}: {e}", sys)
