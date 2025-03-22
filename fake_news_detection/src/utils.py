import pandas as pd
import pickle
import os

def save_data(df: pd.DataFrame, output_path: str):
    """Saves a DataFrame to a CSV file, ensuring the directory exists."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Data successfully saved to {output_path}")

def save_model(model, filename):
    """Save the trained model to a file."""
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

def load_model(filename):
    """Load a saved model from a file."""
    with open(filename, 'rb') as file:
        return pickle.load(file)
