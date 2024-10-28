import joblib
import pandas as pd

# Define a function to load and use the scaler
def load_and_use_scaler(scaler_file, data_file):
    # Load the scaler object
    scaler = joblib.load(scaler_file)
    
    # Load the processed data
    data = joblib.load(data_file)
    
    # To apply the scaler to new data (e.g., new daily returns)
    # Example: new_data = some_new_data_to_transform
    # transformed_data = scaler.transform(new_data)
    
    # Inverse transform the processed data to original scale
    original_data = scaler.inverse_transform(data)