import pandas as pd
import numpy as np
from datasets import load_dataset
from config import DATASET_NAME  

def load_employee_reviews():
    dataset = load_dataset(DATASET_NAME)
    return pd.DataFrame(dataset['train'])

def convert_to_numeric(rating):
    try:
        return float(rating)
    except (ValueError, TypeError):
        return np.nan

def is_numeric_column(series):
    numeric_vals = series.apply(convert_to_numeric)
    return numeric_vals.notna().mean() > 0.8