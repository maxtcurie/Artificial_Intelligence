import pandas as pd
import matplotlib.pyplot as plt

# Download the historical stock data for each year from 2000 to 2020
plt.clf()
for year in range(2000, 2021):

    df = pd.read_csv(f'./stock_data/stock_data_{year}.csv', index_col=0)
    print(df.head())

    plt.plot(df)
plt.show()
