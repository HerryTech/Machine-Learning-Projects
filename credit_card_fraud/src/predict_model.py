import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.logger import get_logger
from src.exception import CustomException
from src.utils import load_model

logger = get_logger(__name__)

def predict_fraud(input_dict, model_path):
    """Predict whether a transaction is fraudulent or not"""
    try:
        #convert input to DataFrame
        df = pd.DataFrame([input_dict])

        #load trained model
        model = load_model(model_path)

        #scale data to predict
        feature_to_scale = ["Amount", "Time"]
        scale = StandardScaler()
        df[feature_to_scale] = scale.transform(df[feature_to_scale])

        #make prediction
        prediction = model.predict(df)

        #return result
        result = "Non-fraudulent transaction" if prediction == 0 else "Fraudulent transaction"
        logger.info(f"Prediction: {result}")
        return result



