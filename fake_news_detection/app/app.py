import streamlit as st
import os
import pickle

# Function to load the model safely
def load_model(filename):
    """Load a saved model from a file."""
    if not os.path.exists(filename):
        st.error(f"❌ File not found: {filename}")
        return None
    with open(filename, "rb") as file:
        return pickle.load(file)

# Get the absolute path for deployment
MODEL_DIR = os.path.join(os.getcwd(), "model")  # ✅ Use `os.getcwd()` for Streamlit Cloud
model_path = os.path.join(MODEL_DIR, "model.pkl")
vectorizer_path = os.path.join(MODEL_DIR, "vectorizer.pkl")

# ✅ Check if model files exist before loading
if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    st.error("⚠️ Model files are missing! Make sure `model.pkl` and `vectorizer.pkl` are in the `model/` folder.")

# Load model & vectorizer
model = load_model(model_path)
vectorizer = load_model(vectorizer_path)

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
    if st.button("🔍 Check News"):
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

            st.write(f"📊 **Prediction Confidence:** FAKE: {fake_prob:.2%}, REAL: {real_prob:.2%}")
        else:
            st.warning("⚠️ Please enter some text.")

# "Clear Text" button
with col2:
    st.button("🧹 Clear Text", on_click=clear_text)
