import pandas as pd
from sklearn.preprocessing import QuantileTransformer
import os
import matplotlib.pyplot as plt

import joblib

# Define the function to calculate daily returns
def calculate_daily_returns(df):
    returns = df.pct_change()
    returns=returns[1:] #first is one is nan
    return returns



# Process and save each file individually
def process_and_plot(file_path,n_quantiles_=100,plot=False):
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
    if plot:
        plt.clf()
        plt.hist(daily_returns,label=daily_returns.columns)
        plt.title('daily_returns')
        plt.legend()
        plt.show()

        plt.clf()
        plt.hist(normalized_data,label=daily_returns.columns)
        plt.title('norm_data')
        plt.legend()

        plt.show()


# Define the file pathss
file_paths = [f'stock_data/stock_data_{year}.csv' for year in range(2000, 2021)]

# Process each file
for file_path in file_paths:
    process_and_plot(file_path,n_quantiles_=50,plot=True)
