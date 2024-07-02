import yfinance as yf
from yahoofinancials import YahooFinancials
import mplfinance as mpf
import pandas as pd
import os


if __name__ == '__main__':

    export_path = 'stocks-23-05-24'

    if not os.path.exists(export_path):
        os.makedirs(export_path)

    # Get the holdings of the SPDR S&P 500 ETF Trust (SPY)

    sp500_stocks = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    sp500_stocks = sp500_stocks.Symbol.to_list()

    for stock in sp500_stocks:
        print(stock)
        # Download stock data
        data = yf.download(stock, period="1y", interval='1wk')

        # Create a candlestick chart
        # Format the index without spaces
        if len(data) > 0:
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
            fig.savefig(f'{export_path}/{stock}.png', bbox_inches='tight', pad_inches=0)
            # mpf.show()

    print('Done')