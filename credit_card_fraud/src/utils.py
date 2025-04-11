import pandas as pd
import numpy as np

def load_data(data_path):
    """Load dataset"""
    df = pd.read_csv(data_path)
    return df

def save_data(df, output_path):
    """Save dataframe to CSV"""
    df.to_csv(output_path, index = False)
    print(f"Data successfully saved to {output_path}")


