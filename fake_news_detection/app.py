import pickle
import streamlit as st

# Load the saved model and vectorizer
with open("fake_news_detection/model/model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("fake_news_detection/model/vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Initialize Streamlit app
def main():
    st.title("📰 Fake News Detection App")
    st.write("Enter a news article and check if it's FAKE or REAL.")

    # Initialize session state for inputs
    if "news_text" not in st.session_state:
        st.session_state.news_text = ""

    # Text input for the news article
    st.session_state.news_text = st.text_area("✍️ Enter news text:", value=st.session_state.news_text)

    # Buttons for prediction and clearing text
    col1, col2 = st.columns([5, 1])

    with col1:
        if st.button("Check News"):
            if st.session_state.news_text:
                text_tfidf = vectorizer.transform([st.session_state.news_text])
                prediction = model.predict(text_tfidf)[0]
                probability = model.predict_proba(text_tfidf)[0]

                fake_prob = probability[0]
                real_prob = probability[1]

                if prediction == 0:
                    st.error(f"🚨 This news article is FAKE! (Confidence: {fake_prob:.2%})")
                else:
                    st.success(f"✅ This news article is REAL! (Confidence: {real_prob:.2%})")
            else:
                st.warning("⚠️ Please enter some text.")

    with col2:
        if st.button("Clear Text"):
            st.session_state.news_text = ""

if __name__ == "__main__":
    main()
