import pickle
import streamlit as st
import os

os.environ["STREAMLIT_CONFIG"] = os.path.join(os.getcwd(), ".streamlit", "config.toml")

st.title("Testing Custom Theme")
st.write("If this works, your theme should be applied!")

# Define the paths
model_path = "fake_news_detection/model/model.pkl"
vectorizer_path = "fake_news_detection/model/vectorizer.pkl"

# Load the saved model and vectorizer
with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

with open(vectorizer_path, "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)
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