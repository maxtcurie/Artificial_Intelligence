import joblib
import pandas as pd
import numpy as np 

# Define a function to count NaN values in a NumPy array or DataFrame
def count_nans(data):
    if isinstance(data, np.ndarray):
        return np.isnan(data).sum()
    elif isinstance(data, pd.DataFrame):
        return data.isna().sum().sum()
    else:
        raise ValueError("Unsupported data type")

# Define a function to load and inspect the scaler
def load_and_inspect_scaler(scaler_file):
    # Load the scaler object
    scaler = joblib.load(scaler_file)
    
    # Inspect the scaler attributes
    print("Scaler Attributes:")
    print(f"n_quantiles: {scaler.n_quantiles_}")
    print(f"quantiles shape: {scaler.quantiles_.shape}")
    print(f"num of nan: {count_nans(scaler.quantiles_)}")
    print(f"quantiles: {scaler.quantiles_}")
    print(f"output_distribution: {scaler.output_distribution}")
    print(f"n_features: {scaler.n_features_in_}")
    print(f"random_state: {scaler.random_state}")
    
    return scaler


for yr in range(2000,2021):

    print('*****************')
    print(yr)
    # Define the scaler file path
    scaler_file = f'stock_data/stock_data_{yr}_scaler.joblib'

    # Load and inspect the scaler
    scaler = load_and_inspect_scaler(scaler_file)

