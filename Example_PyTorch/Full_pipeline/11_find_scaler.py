import pandas as pd
from sklearn.preprocessing import QuantileTransformer
import os
import joblib

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the function to calculate daily returns
def calculate_daily_returns(df):
    returns = df.pct_change()
    returns=returns[1:] #first is one is nan
    return returns

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
    print(f"output_distribution: {scaler.output_distribution}")
    print(f"n_features: {scaler.n_features_in_}")
    print(f"random_state: {scaler.random_state}")
    
    return scaler

# Process and save each file individually
def process_and_scaler(file_path,n_quantiles_=100):
    # Load the data
    df = pd.read_csv(file_path, index_col=0)
    
    # Calculate daily returns
    daily_returns = calculate_daily_returns(df)
    
    # Print rows with NaN values
    nan_rows = df[df.isna().any(axis=1)]
    if not nan_rows.empty:
        print(f"Rows with NaN values in {file_path}:")
        print(nan_rows)

    
    # Handle NaN values by filling with column mean
    daily_returns = daily_returns.bfill()

    # Normalize the data using QuantileTransformer
    scaler = QuantileTransformer(output_distribution='normal',n_quantiles=n_quantiles_)

    normalized_data = scaler.fit_transform(daily_returns)

    return scaler

# Define the file pathss
file_paths = [f'stock_data/stock_data_{year}.csv' for year in range(2000, 2021)]

scaler_list = []

# Process each file
for file_path in file_paths:
    scaler=process_and_scaler(file_path,n_quantiles_=50)

    scaler_list.append(scaler.quantiles_)

    # Convert scaler list to a NumPy array
    scaler_list_tmp = np.array(scaler_list)

    # Compute the mean of the quantiles ignoring NaNs
    quantiles_mean = np.nanmean(scaler_list_tmp, axis=0)

    num_tmp=count_nans(quantiles_mean)
    if num_tmp==0:
        scaler.quantiles_ = quantiles_mean
        joblib.dump(scaler,f'stock_data/stock_data_total_scaler.joblib')
        print(file_path)
        print(quantiles_mean)
        break