import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.logger import get_logger
from src.exception import CustomException
from src.utils import load_model

logger = get_logger(__name__)

def predict_fraud(df, model_path):
    """Predict whether a transaction is fraudulent or not"""
    try:
        model = load_model(model_path)
        feature_to_scale = ["Amount", "Time"]
        scale = StandardScaler()
        df[feature_to_scale] = scale.fit_transform(df[feature_to_scale])


