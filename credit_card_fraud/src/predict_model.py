import pandas as pd
import sys
import joblib
from sklearn.preprocessing import StandardScaler
from src.logger import get_logger
from src.exception import CustomException
from src.utils import load_model

logger = get_logger(__name__)

def predict_fraud(model_path):
    """Predict whether a transaction is fraudulent or not"""
    try:
        sample = {
    'Time': 12095,
    'V1': -4.727712656,
    'V2': 3.044469102,
    'V3': -5.598354267,
    'V4': 5.928190802,
    'V5': -2.190769729,
    'V6': -1.529322966,
    'V7': -4.48742196,
    'V8': 0.916391814,
    'V9': -1.307010423,
    'V10': -4.138891214,
    'V11': 5.149408783,
    'V12': -11.12401861,
    'V13': 0.543067766,
    'V14': -7.840942205,
    'V15': 0.743633945,
    'V16': -6.77706924,
    'V17': -9.931765154,
    'V18': -4.093021122,
    'V19': 1.504924859,
    'V20': -0.207759445,
    'V21': 0.650988236,
    'V22': 0.254983289,
    'V23': 0.628843469,
    'V24': -0.238128454,
    'V25': -0.671332332,
    'V26': -0.033590063,
    'V27': -1.331777322,
    'V28': 0.70569759,
    'Amount': 30.39
}

        #convert input to DataFrame
        df = pd.DataFrame([sample])

        #load trained model
        model = load_model(model_path)

        #scale data to predict
        feature_to_scale = ["Amount", "Time"]
        scale = StandardScaler()
        df[feature_to_scale] = scale.fit_transform(df[feature_to_scale])

        #make prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        #return result
        result = "Non-fraudulent transaction" if prediction == 0 else "Fraudulent transaction"
        logger.info(f"Prediction: {result}")
        print(f"The transaction is a {result}")
        return result, probability
    
    except Exception as e:
        raise CustomException (f"Error in prediction: {e}", sys)
    
if __name__ == "__main__":
    predict_fraud("model/model.pkl")


