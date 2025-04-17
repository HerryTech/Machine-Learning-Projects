import pandas as pd
from src.logger import get_logger

logger = get_logger(__name__)

def load_data(file_path):
    """Function to load data"""
    return pd.read_csv(file_path)
    