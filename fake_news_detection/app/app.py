from flask import Flask, request, render_template
import pickle
import os
from src.utils import load_model

app = Flask(__name__)

MODEL_PATH = "model/model.pkl"
VECTORIZER = "model/vectorizer.pkl"

model = load_model(MODEL_PATH)
vectorizer = load_model(VECTORIZER)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", method = ["POST"])
def predict():
    if request.method == "POST":
        news_text = request.form("news_text")
        text_tfidf = vectorizer.transform([news_text])
        prediction = model.predict(text_tfidf)[0]
        result = "Real News" if prediction == 1 else "Fake News"

        return render_template("result.html", news_text = news_text, prediction = result )