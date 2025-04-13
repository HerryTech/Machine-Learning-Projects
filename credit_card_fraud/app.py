import joblib
import streamlit as st
from src.utils import load_model

#define path and load model
model_path = "model/model.pkl"
model = load_model(model_path)

def main():
    st.title("Credit Card Fraud Detection")