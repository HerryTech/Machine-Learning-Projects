import joblib
import streamlit as st
from src.utils import load_model
from src.predict_model import predict_fraud

st.set_page_config(page_title = "Credit Card Fraud Detector", page_icon="💳")
st.title("💳Credit Card Fraud Detection")
st.markdown("Fill in the transaction details below to check for potential fraud.")

#dictionary to hold user's input
sample = {}

#define column names
columns = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

#create input field
for col in columns:
    sample[col] = st.number_input(label = col, value = 0.0, format = "%.4f")

if st.button("🔍 Predit Fraud"):
    try:
        #define path and load model
        model_path = "model/model.pkl"
        model = load_model(model_path)
        result, probability = predict_fraud(sample, model_path)
        st.success(f"This is a {result}")
        st.info(f"Confidence level: {probability:.2%}")
        
    except Exception as e:
        st.error(f"⚠️ Something went wrong: {e}")


    




