import yfinance as yf
import pandas as pd
import datetime
import plotly.graph_objs as go

'''
Finds the stocks that apply for these conditions:
#   Moving Average Crossover Strategy:
    if the 10-day moving average is above the 20-day moving average, then the stock is considered to be in an uptrend.

'''


def moving_average_crossover(ticker):

    df = yf.download(ticker, start=start_date, end=end_date)
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    # Plot the closing prices and SMAs
    # Create the plot
    fig = go.Figure()

    # Add closing price trace
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price', line=dict(color='blue')))

    # Add 10-day SMA trace
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_5'], mode='lines', name='5-Day SMA', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_10'], mode='lines', name='10-Day SMA', line=dict(color='orange')))

    # Add 20-day SMA trace
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], mode='lines', name='50-Day SMA', line=dict(color='green')))

    # Customize the layout
    fig.update_layout(
        title=f'{ticker} Closing Prices and SMAs',
        xaxis_title='Date',
        yaxis_title='Price',
        legend=dict(x=0, y=1.0),
        hovermode='x unified'
    )

    # Show the plot
    # Save the plot as an HTML file
    fig.write_html(f"technical_figs\\sma_plot_{stock}.html")

    # Save the plot as a static image (PNG)
    fig.write_image(f"technical_figs\\sma_plot_{stock}.png", engine="kaleido")

    signal = df['SMA_5'].iloc[-1] > df['SMA_50'].iloc[-1] and df['SMA_10'].iloc[-1] > df['SMA_50'].iloc[-1]# Simple crossover signal
    return signal


if __name__ == '__main__':

    with open('stocks.txt') as f:
        stocks_list = f.readlines()
        stocks_list = [x.strip() for x in stocks_list]

    # check if the stock is above MA:
    # Define the date range
    end_date = datetime.datetime.now() - datetime.timedelta(days=5)
    start_date = end_date - datetime.timedelta(days=100)  # 30 days of historical data
    potential_risers = []

    for stock in stocks_list:
        if moving_average_crossover(stock):
                print(f'{stock} is above MA')
                potential_risers.append(stock)

    print(potential_risers)
    with open('potential_risers.txt', 'w') as f:
        for item in potential_risers:
            f.write("%s\n" % item)
