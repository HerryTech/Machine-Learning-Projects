import unittest
import pickle
from src.utils import load_model

class TestFakeNewsModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_model("model/model.pkl")
        cls.vectorizer = load_model("model.vectorizer.pkl")

    def test_prediction(self):
        sample_text = ["Breaking news: AI is taking over the world!"]
        text_tfidf = self.vectorizer.transform(sample_text)
        prediction = self.model.predict(text_tfidf)[0]
        self.assertIn(prediction, [0,1])

if __name__ == "__main__":
    unittest.main()