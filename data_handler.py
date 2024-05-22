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
    # mpf.plot(data, type='candle', style='charles', title=f'Candlestick chart for {ticker}', ylabel='Price',
    #          datetime_format='%Y-%m-%dT%H:%M:%S')

    fig, ax = mpf.plot(data, type='candle', style='charles', returnfig=True)

    # Remove ticks and labels
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_xlabel('')
    ax[0].set_ylabel('')
    ax[0].title.set_text('')

    # Adjust the layout to remove whitespace
    fig.tight_layout(pad=0)

    # Show the plot
    fig.savefig('test.png', bbox_inches='tight', pad_inches=0)
    mpf.show()


    print(data)