
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
# For reading stock data from yahoo
from pandas_datareader.data import DataReader
import yfinance as yf
from pandas_datareader import data as pdr

yf.pdr_override()

# For time stamps
from datetime import datetime

# Get the stock quote
df = pdr.get_data_yahoo('AAPL', start='2012-01-01', end=datetime.now())

des = df.describe()
fix, ax = plt.subplots()
ax.axis('off')
table = pd.plotting.table(ax, des, loc='center',
                          cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
plt.show()

sp500_stocks = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
selected_stocks = sp500_stocks.Symbol.to_list()  # ['AAPL', 'MSFT', 'GOOGL', 'AMZN']  # List of stocks to analyze

stocks = pd.DataFrame()
for stock in selected_stocks:
    data = pdr.get_data_yahoo(stock, start='2023-01-01', end=datetime.now())
    data['Ticker'] = stock
    stocks = pd.concat([stocks, data])

stocks_selected = stocks.loc[stocks['Ticker'].isin(selected_stocks)]
stocks_selected['Date'] = pd.to_datetime(stocks_selected.index)

pivot_stocks = stocks_selected.pivot(index='Date', columns='Ticker', values='Close')

# Calculating correlation matrix
correlation_matrix = pivot_stocks.corr()

# Set threshold for low correlation (e.g., abs(correlation) < 0.3)
threshold = 0.75

# Specific stock for filtering correlations
specific_stock = 'AAPL'

# Filter correlations for the specific stock
specific_stock_corr = correlation_matrix[specific_stock]
filtered_corr = specific_stock_corr[specific_stock_corr > threshold]

# Displaying the filtered correlations
for stock, corr in correlation_matrix.items():
    if stock not in filtered_corr.index:
        correlation_matrix.drop(stock, axis=0, inplace=True)
        correlation_matrix.drop(stock, axis=1, inplace=True)


# Visualizing the filtered correlations
plt.figure(figsize=(10, 1))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5, linecolor='black')
plt.title(f'Filtered High Correlation with {specific_stock}')
plt.show()

sorted_series = filtered_corr.sort_values(ascending=False)
print(sorted_series)


# Assuming 'data' is a DataFrame containing daily stock data with 'Date' and 'Close' columns
df['Date'] = pd.to_datetime(df.index)
df.set_index('Date', inplace=True)

# Resample data to weekly or monthly frequency
weekly_data = df['Close'].resample('W').mean()  # Weekly average closing prices
monthly_data = df['Close'].resample('M').mean()  # Monthly average closing prices

plt.figure(figsize=(14, 7))
plt.style.use('seaborn-v0_8')

# Plot weekly average closing prices
plt.subplot(2, 1, 1)
plt.plot(weekly_data.index, weekly_data.values, marker='o', linestyle='-')
plt.title('Weekly Average Closing Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)

# Plot monthly average closing prices
plt.subplot(2, 1, 2)
plt.plot(monthly_data.index, monthly_data.values, marker='o', linestyle='-')
plt.title('Monthly Average Closing Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)

plt.tight_layout()
plt.show()

# Show the data
plt.figure(figsize=(16, 6))
plt.style.use('seaborn-v0_8')

plt.title('Close Price History')
plt.plot(df['Close'], label='Close Price')
plt.plot(df['Close'].rolling(window=20).mean(), label='20 Day Moving Average')
plt.plot(df['Close'].rolling(window=50).mean(), label='50 Day Moving Average')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.legend(loc='upper left')
plt.show()

# Create a new dataframe with only the 'Close column
data = df.filter(['Close'])
# Convert the dataframe to a numpy array
dataset = data.values
# Get the number of rows to train the model on
training_data_len = int(np.ceil( len(dataset) * .95 ))

# Calculate 20-day moving average and standard deviation
df['Date'] = pd.to_datetime(df.index)
window = 20
rolling_mean = data.rolling(window).mean()
rolling_std = data.rolling(window).std()

# Calculate upper and lower Bollinger Bands
upper_band = rolling_mean + (2 * rolling_std)
lower_band = rolling_mean - (2 * rolling_std)

# Plotting Bollinger Bands
plt.figure(figsize=(14, 7))
plt.style.use('seaborn-v0_8')
plt.plot(df['Date'][2000:], data[2000:], label='Closing Price', color='blue')
plt.plot(df['Date'][2000:], rolling_mean[2000:], label='20-Day SMA', color='green')
plt.plot(df['Date'][2000:], upper_band[2000:], label='Upper Bollinger Band', color='red', linestyle='--')
plt.plot(df['Date'][2000:], lower_band[2000:], label='Lower Bollinger Band', color='orange', linestyle='--')
plt.fill_between(df['Date'][2000:], lower_band['Close'][2000:], upper_band['Close'][2000:], alpha=0.2, color='gray')  # Fill between the bands
plt.title('Bollinger Bands for Stock Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()
