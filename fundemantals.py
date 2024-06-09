import yfinance as yf
import pandas as pd
import datetime
import plotly.graph_objs as go

'''
Finds the stocks that apply for these conditions:
#	Sales increase in 10% than prev. year
#	Earning increase in 10% than prev. year
#	Debt lower than prev. year
#	ROE 10%+
#   PEG ratio < 1
#  	P/S ratio < 1
#  	P/B ratio < 1
#  	EV/EBITDA < 10
#  	P/FCF  < 15


## Moving Average Crossover Strategy:
if the 10-day moving average is above the 20-day moving average, then the stock is considered to be in an uptrend.

'''

def get_growth_stocks():
    # Get the holdings of the SPDR S&P 500 ETF Trust (SPY)
    sp500_stocks = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    sp500_stocks = sp500_stocks.Symbol.to_list()
    stocks_list = []

    for stock in sp500_stocks:
        print(stock)
        try:
            # Download stock data
            data = yf.Ticker(stock)
            s = data.splits
            f = data.financials
            f_q = data.quarterly_financials
            c = data.cashflow
            c_q = data.quarterly_cashflow
            b = data.balance_sheet

            # Check if Sales increase in 10% than prev. year
            sales_inc = f.loc['Total Revenue'].iloc[0] / f.loc['Total Revenue'].iloc[1]

            # Check if Earning increase in 10% than prev. year
            erning_inc = f.loc['Net Income'].iloc[0] / f.loc['Net Income'].iloc[1]

            # Check if Debt lower than prev. year
            debt = b.loc['Long Term Debt'].iloc[0] < b.loc['Long Term Debt'].iloc[1]

            # Check if ROE 10%+
            # Calculate Shareholder's Equity (Shareholder's Equity is typically the difference between Total Assets and Total Liabilities)
            total_assets = b.loc['Total Assets'].iloc[0]
            total_liabilities = b.loc['Stockholders Equity'].iloc[0]
            shareholders_equity = total_assets - total_liabilities
            roe = f.loc['Net Income'].iloc[0] / shareholders_equity

            # Check if PEG ratio < 1
            peg = data.info['pegRatio'] < 1

            # Check if P/S ratio < 1
            ps = data.info['priceToSalesTrailing12Months'] < 1

            # Check if P/B ratio < 1
            pb = data.info['priceToBook'] < 1

            # Check if EV/EBITDA < 10
            ev = data.info['enterpriseToEbitda'] < 10

            # Check if P/FCF  < 15  price share/ (levered free cash flow/ shares outstanding).
            pfcf = data.info['open'] / (c.loc['Free Cash Flow'].iloc[0] / data.info.get('sharesOutstanding')) < 15

            if sales_inc > 1.1 and erning_inc > 1.1 and debt:  # and roe > 1.1 and peg and ps and pb and ev and pfcf:
                print(f'{stock} applies for the conditions')
                stocks_list.append(stock)
        except:
            print(f'{stock} cannot be calculated for these conditions')
            pass

    with open('stocks.txt', 'w') as f:
        for item in stocks_list:
            f.write("%s\n" % item)

    print(stocks_list)
    return stocks_list


if __name__ == '__main__':

    search_stocks_flag = False
    if search_stocks_flag:
        stocks_list = get_growth_stocks()
    else:
        try:
            with open('stocks.txt') as f:
                stocks_list = f.readlines()
                stocks_list = [x.strip() for x in stocks_list]
        except:
            print('No stocks found')
            stocks_list = get_growth_stocks()

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