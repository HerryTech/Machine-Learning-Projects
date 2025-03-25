import streamlit as st
from src.utils import load_model
import os

# Load model & vectorizer
model = load_model("../model/model.pkl")
vectorizer = load_model("../model/vectorizer.pkl")

# Function to clear text input
def clear_text():
    st.session_state.news_text = ""

# Streamlit UI
st.title("📰 Fake News Detection App")
st.write("Enter a news article and check if it's **FAKE** or **REAL**.")

# Ensure session state has 'news_text'
if "news_text" not in st.session_state:
    st.session_state.news_text = ""

# Input text area
news_text = st.text_area("✍️ Enter news text:", key="news_text")

# 🔹 Create two columns for buttons
col1, col2 = st.columns([5, 1])

# "Check News" button
with col1:
    if st.button("Check News"):
        if not model or not vectorizer:
            st.error("❌ Model is not loaded. Please check your deployment.")
        elif news_text:
            text_tfidf = vectorizer.transform([news_text])
            prediction = model.predict(text_tfidf)[0]
            probability = model.predict_proba(text_tfidf)[0]

            fake_prob, real_prob = probability[0], probability[1]

            if prediction == 0:
                st.error(f"🚨 This news article is **FAKE**! (Confidence: {fake_prob:.2%})")
            else:
                st.success(f"✅ This news article is **REAL**! (Confidence: {real_prob:.2%})")
        else:
            st.warning("⚠️ Please enter some text.")

# "Clear Text" button
with col2:
    st.button("Clear Text", on_click=clear_text)
