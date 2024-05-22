import yfinance as yf
import mplfinance as mpf
import pandas as pd


if __name__ == '__main__':

    ticker = "AAPL"
    data = yf.download(ticker, period="5d", interval='30m')

    # Create a candlestick chart
    # Format the index without spaces
    data.index = data.index.strftime('%Y-%m-%dT%H:%M:%S')

    # Convert the index back to datetime for mplfinance
    data.index = pd.to_datetime(data.index)

    # Plot the candlestick chart
    mpf.plot(data, type='candle', style='charles', title=f'Candlestick chart for {ticker}', ylabel='Price',
             datetime_format='%Y-%m-%dT%H:%M:%S')

    print(data)