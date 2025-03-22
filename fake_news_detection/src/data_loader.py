import pandas as pd
import numpy as np
import re
import nltk
import sys
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from src.logger import get_logger
from src.exception import CustomException
from src.utils import save_data

# Download necessary NLTK data
nltk.download("stopwords")
nltk.download("wordnet")

# Initialize logger
logger = get_logger(__name__)

# Initialize stopwords & lemmatizer
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def load_data(true_path, fake_path):
    """Loads and merges true & fake news datasets"""
    try:
        logger.info("Loading datasets...")
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

        # Remove missing values
        df.dropna(inplace=True)

        # Remove duplicates
        df.drop_duplicates(inplace=True)

        logger.info(f"Dataset loaded successfully! Total samples: {len(df)}")
        return df

    except Exception as e:
        raise CustomException(f"Error loading dataset: {e}", sys)

def clean_text(text):
    """Cleans text by removing punctuation, stopwords, and applying lemmatization"""
    try:
        text = text.lower()
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        words = text.split()
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        return " ".join(words)
    except Exception as e:
        raise CustomException(f"Error in text cleaning: {e}", sys)

def preprocess_and_save(df, output_path):
    """Applies text cleaning and saves the cleaned dataset"""
    try:
        logger.info("Starting text cleaning process...")

        df["title"] = df["title"].astype(str).map(clean_text)
        df["text"] = df["text"].astype(str).map(clean_text)

        # Replace empty strings with NaN (if cleaning removes all words)
        df.replace({"": np.nan}, inplace=True)

        # Remove missing values after cleaning
        df.dropna(inplace=True)

        # Save cleaned dataset using utils function
        save_data(df, output_path)

        logger.info(f"Final cleaned data saved to {output_path}")
        return df

    except Exception as e:
        raise CustomException(f"Error in preprocessing and saving data: {e}", sys)

if __name__ == "__main__":
    try:
        df = load_data("data/raw/True.csv", "data/raw/Fake.csv")
        preprocess_and_save(df, "data/processed/cleaned_news.csv")
    except Exception as e:
        logger.error(f"Script failed: {e}")
