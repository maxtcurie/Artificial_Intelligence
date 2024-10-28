import yfinance as yf
import os

# Define the stock tickers and the date ranges
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']  # Facebook's new ticker is META

# Create a directory to store the data
os.makedirs('stock_data', exist_ok=True)

# Download the historical stock data for each year from 2000 to 2020
for year in range(2000, 2021):
    start_date = f'{year}-01-01'
    end_date = f'{year}-12-31'
    
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    
    data.to_csv(f'./stock_data/stock_data_{year}.csv')
