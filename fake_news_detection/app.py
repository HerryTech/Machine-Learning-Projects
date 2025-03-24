import streamlit as st
from src.utils import load_model

model = load_model("model/model.pkl")
vectorizer = load_model("model/vectorizer.pkl")

#streamlit UI
st.title("Fake New Dectection App")
st.write("Enter a news article and check if it's FAKE or REAL.")

#input text area
news_text = st.text_area("Enter news text:", "")

if st.button("Check News"):
    if news_text:
        text_tfidf = vectorizer.transform([news_text])
        prediction = model.predict(text_tfidf)[0]
        if prediction == 0:
            st.error("🚨 This news article is FAKE!")
        else:
            st.success("✅ This news article is REAL!")
    else:
        st.warning("⚠️ Please enter some text.")
