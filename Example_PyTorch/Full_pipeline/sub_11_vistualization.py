import pandas as pd
from sklearn.preprocessing import QuantileTransformer
import os
import joblib

# Define the function to calculate daily returns
def calculate_daily_returns(df):
    returns = df.pct_change().dropna()
    return returns

# Process and save each file individually
def process_and_save_file(file_path):
    # Load the data
    df = pd.read_csv(file_path, index_col=0)
    
    # Calculate daily returns
    daily_returns = calculate_daily_returns(df)
    
    # Handle NaN values by filling with column mean
    daily_returns = daily_returns.fillna(daily_returns.mean())
    
    # Normalize the data using QuantileTransformer
    scaler = QuantileTransformer(output_distribution='normal')
    normalized_data = scaler.fit_transform(daily_returns)
    
    # Save the processed data to a file
    processed_data_file = file_path.replace('.csv', '_processed.joblib')
    joblib.dump(normalized_data, processed_data_file)
    
    # Save the scaler object to a file
    scaler_file = file_path.replace('.csv', '_scaler.joblib')
    joblib.dump(scaler, scaler_file)

    print(f"Processed data saved to {processed_data_file}")
    print(f"Scaler object saved to {scaler_file}")

# Define the file paths
file_paths = [f'stock_data/stock_data_{year}.csv' for year in range(2000, 2021)]

# Process each file
for file_path in file_paths:
    process_and_save_file(file_path)
