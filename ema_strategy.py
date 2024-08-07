import shutil

import yfinance as yf
import pandas as pd
import datetime
import plotly.graph_objs as go
import numpy as np
import os
from fundemantals import Fundamentals


'''
Finds the stocks that apply for these conditions:
#   Moving Average Crossover Strategy:
    if the X-day exponential moving average is above the Y-day exponential moving average, then the stock is considered
    to be in an uptrend.
    X between 7 and 15
    Y between 25 and 45

'''


def get_uptrend_stocks(num_days, sig_x, sig_y):
    min_diff = 10
    min_idx = -1
    trend = 'none'
    x = sig_x[-num_days:]
    y = sig_y[-num_days:]
    for k, (i, j) in enumerate(zip(x, y)):
        diff = np.abs(i - j)
        if diff < min_diff:
            min_diff = diff
            min_idx = k
            if k < (min_diff - 1) and x.iloc[k+1] > y.iloc[k+1]:
                trend = 'Uptrend'

    return min_diff, min_idx, trend


def moving_average_crossover(ticker, start_date, end_date, X, Y):

        df = yf.download(ticker, start=start_date, end=end_date)
        df['EMA_X'] = df['Close'].ewm(span=X, adjust=False).mean()
        df['EMA_Y'] = df['Close'].ewm(span=Y, adjust=False).mean()
        # Plot the closing prices and EMAs

        if not os.path.exists(f"potential_risers_figs/{month - 1}"):
            os.makedirs(f"potential_risers_figs/{month - 1}")
        # Create the plot
        fig = go.Figure()

        # Add closing price trace
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price', line=dict(color='blue')))
        # Add X-day EMA trace
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_X'], mode='lines', name=f'{X}-Day EMA', line=dict(color='orange')))
        # Add Y-day EMA trace
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_Y'], mode='lines', name=f'{Y}-Day EMA', line=dict(color='green')))
        # Customize the layout
        fig.update_layout(
            title=f'{ticker} Closing Prices and EMAs',
            xaxis_title='Date',
            yaxis_title='Price',
            legend=dict(x=0, y=1.0),
            hovermode='x unified'
        )
        # fig.show()
        fig.write_html(f"potential_risers_figs/{month - 1}/{stock}.html")

        return df['EMA_X'], df['EMA_Y'], df['Close']


if __name__ == '__main__':

    # Get the holdings of the SPDR S&P 500 ETF Trust (SPY)
    sp500_stocks = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    sp500_stocks = sp500_stocks.Symbol.to_list()
    stocks_list = []
    x = list(range(7, 17, 2))
    y = list(range(25, 35, 2))
    res = []

    for month in range(1):
        res = []

        for stock in sp500_stocks:
            signals = []
            print(stock)
            # for i in x:
            #     for j in y:
            for (i, j) in (zip(x, y)):
                signal_x, signal_y, cl_price = moving_average_crossover(stock, datetime.datetime(2021, 1, 1), datetime.datetime(2024, 7, 26), i, j)
                diff, idx, direction = get_uptrend_stocks(10, signal_x, signal_y)
                res.append([stock, i, j, diff, idx, direction])
                signals.append([signal_x, signal_y])
                print(f'{stock} {i} {j}')

            fig = go.Figure()
            for sig in signals:

                # Add EMAs price trace
                fig.add_trace(go.Scatter(x=sig[0].index, y=sig[0], mode='lines', name='EMA short', line=dict(color='red')))
                fig.add_trace(go.Scatter(x=sig[1].index, y=sig[1], mode='lines', name='EMA long', line=dict(color='green')))


            # Add title and labels
            fig.update_layout(
                title=f'Multiple Scatter Plots for {stock} EMAs',
                xaxis_title='X Axis',
                yaxis_title='Y Axis'
            )

            # Save the plot
            if not os.path.exists(f"emas_figs/{month - 1}"):
                os.makedirs(f"emas_figs/{month - 1}")
            fig.write_html(f"emas_figs/{month - 1}/{stock}_EMAs.html")

        # Save the results to a CSV file
        res_pd_ = pd.DataFrame(res, columns=['Stock', 'X', 'Y', 'Difference', 'Index', 'Trend'])
        res_pd_.to_csv(f"res_{month}_today.csv")
        # print(res)
        # res = pd.read_csv('res.csv')
        res_pd = pd.DataFrame(res, columns=['Stock', 'X', 'Y', 'Difference', 'Index', 'Trend'])
        res_pd_stocks = res_pd.groupby('Stock')
        # Create a dictionary of DataFrames
        dfs = {category: data for category, data in res_pd_stocks}
        potential_risers = []

        # Check the DataFrames
        for category, data in dfs.items():
            print(f"DataFrame for Category '{category}':")
            if (data['Trend'] == 'Uptrend').all():
                if len(data['Index'].unique()) == 1 and data['Difference'].mean() < 1.5:
                    potential_risers.append(data)
                    shutil.copy(f"emas_figs/{month - 1}/{data.iloc[0]['Stock']}_EMAs.html", f"potential_risers_figs/{month - 1}/{data.iloc[0]['Stock']}_EMAs.html")

        print("****** potential_risers evaluation ******")

        print('*** potential_risers - fundamental check ***')
        f = Fundamentals()
        potential_risers_stocks = [x.iloc[0]['Stock'] for x in potential_risers]
        for stock in potential_risers_stocks:
            info = f.get_stock_data(stock)
            peg = f.get_peg()
            ps = f.get_ps()
            pb = f.get_pb()
            ev = f.get_ev()
            pfcf = f.get_pfcf()

            if peg < 4 and ps < 4 and pb < 5 and ev < 12 and pfcf < 20:
                print(f'{stock} is a potential riser')
                shutil.copy(
                            f"potential_risers_figs/{month - 1}/{data.iloc[0]['Stock']}_EMAs.html"
                , f"potential_risers_fund_figs/{month - 1}/{stock}_fund.html")

        succses = 0
        potential_risers_eval = []
        for s in potential_risers:
            stock = s.iloc[0]['Stock']
            mean_diff = s['Difference'].mean()
            df = yf.download(stock, start=datetime.datetime(2024, 7, 21), end=datetime.datetime(2024, 7, 28))
            close_max = df['Close'][3:].max()
            close_start = df['Close'][0]

            revenue = (close_max / close_start)
            if revenue > 1:
                succses += 1

            potential_risers_eval.append([stock, revenue, mean_diff, s['Difference']])

            fig = go.Figure()

            # Add closing price trace
            # if not os.path.exists(f"potential_risers_figs/{month-1}"):
            #     os.makedirs(f"potential_risers_figs/{month-1}")
            #
            # fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price', line=dict(color='blue')))
            # # Customize the layout
            # fig.update_layout(
            #     title=f'{stock} Closing Prices',
            #     xaxis_title='Date',
            #     yaxis_title='Price',
            #     legend=dict(x=0, y=1.0),
            #     hovermode='x unified'
            # )
            # fig.write_html(f"potential_risers_figs/{month-1}/{stock}_eval.html")

        print(f"**** eval results month = {month}****")
        print(f'Total sucsess {succses} out of {len(potential_risers)}')
        potential_risers_eval_pd = pd.DataFrame(potential_risers_eval)
        potential_risers_eval_pd.to_csv(f"potential_risers_eval_pd_{month}.csv")
