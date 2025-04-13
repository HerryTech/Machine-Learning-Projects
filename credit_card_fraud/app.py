import joblib
import streamlit as st
from src.utils import load_model

#define path and load model
model_path = "model/model.pkl"
model = load_model(model_path)

st.set_page_config(page_title = "Credit Card Fraud Detector", page_icon="💳")
st.title("💳Credit Card Fraud Detection")