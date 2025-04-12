import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.logger import get_logger

logger = get_logger(__name__)

def evaluate_model():
    """Evaluate the trained model using some metrics"""
    try:
        