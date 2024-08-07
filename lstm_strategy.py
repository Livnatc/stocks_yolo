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
import tensorflow as tf
import tensorflow_addons as tfa
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input, LayerNormalization


# Define Transformer Encoder Layer
class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, head_size, num_heads, ff_dim, dropout=0):
        super(TransformerEncoderLayer, self).__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(head_size)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

    def call(self, inputs, training):
        attn_output = self.mha(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


# # Get the stock quote
df = pdr.get_data_yahoo('AAPL', start='2012-01-01', end=datetime.now())
#
# des = df.describe()
# fix, ax = plt.subplots()
# ax.axis('off')
# table = pd.plotting.table(ax, des, loc='center',
#                           cellLoc='center')
# table.auto_set_font_size(False)
# table.set_fontsize(12)
# plt.show()
#
# sp500_stocks = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
# selected_stocks = sp500_stocks.Symbol.to_list()  # ['AAPL', 'MSFT', 'GOOGL', 'AMZN']  # List of stocks to analyze
#
# stocks = pd.DataFrame()
# for stock in selected_stocks:
#     data = pdr.get_data_yahoo(stock, start='2023-01-01', end=datetime.now())
#     data['Ticker'] = stock
#     stocks = pd.concat([stocks, data])
#
# stocks_selected = stocks.loc[stocks['Ticker'].isin(selected_stocks)]
# stocks_selected['Date'] = pd.to_datetime(stocks_selected.index)
#
# pivot_stocks = stocks_selected.pivot(index='Date', columns='Ticker', values='Close')
#
# # Calculating correlation matrix
# correlation_matrix = pivot_stocks.corr()
#
# # Set threshold for low correlation (e.g., abs(correlation) < 0.3)
# threshold = 0.75
#
# # Specific stock for filtering correlations
# specific_stock = 'AAPL'
#
# # Filter correlations for the specific stock
# specific_stock_corr = correlation_matrix[specific_stock]
# filtered_corr = specific_stock_corr[specific_stock_corr > threshold]
#
# # Displaying the filtered correlations
# for stock, corr in correlation_matrix.items():
#     if stock not in filtered_corr.index:
#         correlation_matrix.drop(stock, axis=0, inplace=True)
#         correlation_matrix.drop(stock, axis=1, inplace=True)
#
#
# # Visualizing the filtered correlations
# plt.figure(figsize=(10, 1))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5, linecolor='black')
# plt.title(f'Filtered High Correlation with {specific_stock}')
# plt.show()
#
# sorted_series = filtered_corr.sort_values(ascending=False)
# print(sorted_series)
#
#
# # Assuming 'data' is a DataFrame containing daily stock data with 'Date' and 'Close' columns
# df['Date'] = pd.to_datetime(df.index)
# df.set_index('Date', inplace=True)
#
# # Resample data to weekly or monthly frequency
# weekly_data = df['Close'].resample('W').mean()  # Weekly average closing prices
# monthly_data = df['Close'].resample('M').mean()  # Monthly average closing prices
#
# plt.figure(figsize=(14, 7))
# plt.style.use('seaborn-v0_8')
#
# # Plot weekly average closing prices
# plt.subplot(2, 1, 1)
# plt.plot(weekly_data.index, weekly_data.values, marker='o', linestyle='-')
# plt.title('Weekly Average Closing Prices')
# plt.xlabel('Date')
# plt.ylabel('Price')
# plt.grid(True)
#
# # Plot monthly average closing prices
# plt.subplot(2, 1, 2)
# plt.plot(monthly_data.index, monthly_data.values, marker='o', linestyle='-')
# plt.title('Monthly Average Closing Prices')
# plt.xlabel('Date')
# plt.ylabel('Price')
# plt.grid(True)
#
# plt.tight_layout()
# plt.show()
#
# # Show the data
# plt.figure(figsize=(16, 6))
# plt.style.use('seaborn-v0_8')
#
# plt.title('Close Price History')
# plt.plot(df['Close'], label='Close Price')
# plt.plot(df['Close'].rolling(window=20).mean(), label='20 Day Moving Average')
# plt.plot(df['Close'].rolling(window=50).mean(), label='50 Day Moving Average')
# plt.xlabel('Date', fontsize=18)
# plt.ylabel('Close Price USD ($)', fontsize=18)
# plt.legend(loc='upper left')
# plt.show()
#
# Create a new dataframe with only the 'Close column
data = df.filter(['Close'])
# Convert the dataframe to a numpy array
dataset = data.values
# Get the number of rows to train the model on
training_data_len = int(np.ceil( len(dataset) * .95 ))
#
# # Calculate 20-day moving average and standard deviation
# df['Date'] = pd.to_datetime(df.index)
# window = 20
# rolling_mean = data.rolling(window).mean()
# rolling_std = data.rolling(window).std()
#
# # Calculate upper and lower Bollinger Bands
# upper_band = rolling_mean + (2 * rolling_std)
# lower_band = rolling_mean - (2 * rolling_std)
#
# # Plotting Bollinger Bands
# plt.figure(figsize=(14, 7))
# plt.style.use('seaborn-v0_8')
# plt.plot(df['Date'][2000:], data[2000:], label='Closing Price', color='blue')
# plt.plot(df['Date'][2000:], rolling_mean[2000:], label='20-Day SMA', color='green')
# plt.plot(df['Date'][2000:], upper_band[2000:], label='Upper Bollinger Band', color='red', linestyle='--')
# plt.plot(df['Date'][2000:], lower_band[2000:], label='Lower Bollinger Band', color='orange', linestyle='--')
# plt.fill_between(df['Date'][2000:], lower_band['Close'][2000:], upper_band['Close'][2000:], alpha=0.2, color='gray')  # Fill between the bands
# plt.title('Bollinger Bands for Stock Prices')
# plt.xlabel('Date')
# plt.ylabel('Price')
# plt.legend()
# plt.grid(True)
# plt.show()


# Scale the data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

# Create the training data set
# Create the scaled training data set
train_data = scaled_data[0:int(training_data_len), :]
# Split the data into x_train and y_train data sets
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i - 60:i, 0])
    y_train.append(train_data[i, 0])
    if i <= 61:
        print(x_train)
        print(y_train)
        print()

# Convert the x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


# Build the LSTM model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
# model = Sequential([
#     Input(shape=(x_train.shape[1], x_train.shape[2])),
#     TransformerEncoderLayer(head_size=64, num_heads=4, ff_dim=32, dropout=0.1),
#     TransformerEncoderLayer(head_size=64, num_heads=4, ff_dim=32, dropout=0.1),
#     tf.keras.layers.GlobalAveragePooling1D(),
#     Dense(25, activation='relu'),
#     Dense(1)
# ])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1, verbose=1)

# Create the testing data set
# Create a new array containing scaled values from index 1543 to 2002
test_data = scaled_data[training_data_len - 60:, :]
# Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i - 60:i, 0])

# Convert the data to a numpy array
x_test = np.array(x_test)

# Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Get the models predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))


# Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
# Visualize the data
plt.figure(figsize=(16,6))
plt.style.use('seaborn-v0_8')

plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

print('done')

# test new data:
test_data = scaled_data[- 60:, :]
# Create the data sets x_test and y_test
x_test = []
y_test = dataset[- 60:, :]
pred = []
for i in range(60, 70):
    x_test = []
    x_test.append(test_data[i - 60:i, 0])

    # Convert the data to a numpy array
    x_test = np.array(x_test)

    # Reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Get the models predicted price values
    predictions = model.predict(x_test)
    test_data = np.append(test_data, np.array(predictions[-1])[0])
    test_data = np.reshape(test_data, (test_data.shape[0], 1))

    predictions = scaler.inverse_transform(predictions)

    pred.append(predictions)

# Get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))


# Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
# Visualize the data
plt.figure(figsize=(16,6))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

