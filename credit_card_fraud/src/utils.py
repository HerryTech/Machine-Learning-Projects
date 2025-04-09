import pandas as pd
import numpy as np

def save_data(df, output_path):
    """Save dataframe to CSV"""
    df.to_csv(output_path, index = False)
    print(f"Data successfully saved to {output_path}")


