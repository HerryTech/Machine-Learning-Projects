import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize stopwords & lemmatizer
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def load_data(true_path, fake_path):
    """Loads and merges true & fake news datasets"""
    true_news = pd.read_csv(true_path)
    fake_news = pd.read_csv(fake_path)
    
    # Add labels: 1 = True News, 0 = Fake News
    true_news["label"] = 1
    fake_news["label"] = 0
    
    # Combine datasets
    df = pd.concat([true_news, fake_news], axis=0)
    
    # Shuffle dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Select relevant columns
    df = df[["title", "text", "label"]]

    #Remove missing values
    df = df.dropna()
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    return df

def clean_text(text):
    """Cleans text by removing punctuation, stopwords, and applying lemmatization"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

def preprocess_and_save(df, output_path):
    """Applies text cleaning and saves the cleaned dataset"""
    df["title"] = df["title"].astype(str).map(clean_text)
    df["text"] = df["text"].astype(str).map(clean_text)

    # Replace empty strings with NaN (if cleaning removes all words)
    df["title"] = df["title"].replace("", np.nan)
    df["text"] = df["text"].replace("", np.nan)

    # Remove missing values after cleaning
    print("Missing values after cleaning:\n", df.isnull().sum())
    df = df.dropna()
    print("After removing missing values:\n", df.isnull().sum())

    # Save cleaned dataset again
    df.to_csv(output_path, index=False)
    print(f"✅ Final cleaned data saved to {output_path}")

    return df

if __name__ == "__main__":
    # Load, clean, and save data
    df = load_data("data/raw/True.csv", "data/raw/Fake.csv")
    preprocess_and_save(df, "data/processed/cleaned_news.csv")
