import pandas as pd
from sklearn.preprocessing import QuantileTransformer
import os
import joblib

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


scaler=joblib.load('stock_data/stock_data_total_scaler.joblib')

# Define the function to calculate daily returns
def calculate_daily_returns(df):
    returns = df.pct_change()
    returns=returns[1:] #first is one is nan
    return returns

# Process and save each file individually
def process_and_save_file(file_path,scaler,n_quantiles_=100):
    # Load the data
    df = pd.read_csv(file_path, index_col=0)
    
    # Calculate daily returns
    daily_returns = calculate_daily_returns(df)

    
    # Handle NaN values by filling with column mean
    daily_returns = daily_returns.bfill().fillna(0)

    normalized_data = scaler.fit_transform(daily_returns)

    # Save the processed data to a file
    processed_data_file = file_path.replace('.csv', '_processed.joblib')

    joblib.dump(normalized_data, processed_data_file)
    print(f"Processed data saved to {processed_data_file}")

    return 0 

# Define the file pathss
file_paths = [f'stock_data/stock_data_{year}.csv' for year in range(2000, 2021)]

scaler_list = []

# Process each file
for file_path in file_paths:
    process_and_save_file(file_path,scaler,n_quantiles_=50)