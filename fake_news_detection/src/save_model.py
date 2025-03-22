import joblib
import os
from src.logger import get_logger
from src.exception import CustomException
import sys

logger = get_logger(__name__)

def save_model(model, filename):
    """Save the trained model to a file."""
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        joblib.dump(model, filename)
        logger.info(f"Model saved at {filename}")
    except Exception as e:
        raise CustomException(f"Error in saving model: {e}", sys)

def load_model(filename):
    """Load a saved model from a file."""
    try:
        model = joblib.load(filename)
        logger.info(f"Model loaded from {filename}")
        return model
    except Exception as e:
        raise CustomException(f"Error in loading model: {e}", sys)
