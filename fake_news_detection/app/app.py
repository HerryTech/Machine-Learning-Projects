import os
import sys
import streamlit as st
import joblib

# Define the paths
model_path = "model/model.pkl"
vectorizer_path = "model/vectorizer.pkl"

# Load the saved model and vectorizer
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

print("Model and vectorizer loaded successfully!")

# Function to clear text
def clear_text():
    st.session_state.news_text = ""

# Streamlit UI
st.title("📰 Fake News Detection App")
st.write("Enter a news article and check if it's FAKE or REAL.")

# Ensure session state has 'news_text'
if "news_text" not in st.session_state:
    st.session_state.news_text = ""

# Input text area
news_text = st.text_area("✍️ Enter news text:", key="news_text")

# 🔹 Create two columns for side-by-side buttons
col1, col2 = st.columns([5, 1])

# "Check News" button in the first column
with col1:
    if st.button("Check News"):
        if news_text:
            text_tfidf = vectorizer.transform([news_text])
            prediction = model.predict(text_tfidf)[0]
            probability = model.predict_proba(text_tfidf)[0]  # Get probability

            fake_prob = probability[0]
            real_prob = probability[1]

            if prediction == 0:
                st.error(f"🚨 This news article is FAKE! (Confidence: {fake_prob:.2%})")
            else:
                st.success(f"✅ This news article is REAL! (Confidence: {real_prob:.2%})")

        else:
            st.warning("⚠️ Please enter some text.")

# "Clear Text" button in the second column
with col2:
    st.button("Clear Text", on_click=clear_text)  