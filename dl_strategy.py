import os
import shutil

import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import datetime
import scipy.stats as stats
import numpy as np
import plotly.express as px

'''
Finds the next price for each stock in the list:
#   try to predict the next price for each stock in the list:
    using stochastic process theory

'''

def check_normality(prices):
    print('Checking normality')
    prices = list(prices)
    # shapiro_test = stats.shapiro(data['Close'])
    # print('Shapiro-Wilk Test:', shapiro_test)
    try:
        ks_test = stats.kstest(data['Close'], 'norm', args=(np.mean(data['Close']), np.std(data['Close'])))
        print('Kolmogorov-Smirnov Test:', ks_test)

        ad_test = stats.anderson(data['Close'], dist='norm')
        print('Anderson-Darling Test:', ad_test)

        return ks_test[1] > 0.05 and ad_test.fit_result.success
    except:
        return False


if __name__ == '__main__':

    # combine all df's:
    normal_stocks_bank = pd.DataFrame()
    df_files = os.listdir()
    df_files = [x for x in df_files if x.endswith('.csv') and 'stocks_bank' in x and 'normal' not in x]
    for f in df_files:
        df = pd.read_csv(f)
        try:
            normal_stocks_bank = pd.concat([normal_stocks_bank, df], ignore_index=True)
        except:
            continue

    sp500_stocks = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    sp500_stocks = sp500_stocks.Symbol.to_list()

    start_date = datetime.datetime(2020, 1, 1)
    end_date = datetime.datetime(2024, 7, 25)
    normal_stocks_bank = []
    stocks_bank = []

    for stock in range(1):  # sp500_stocks:
        print(stock)
        data = yf.download('C', start=start_date, end=end_date)
        is_data_normal = check_normality(data['Close'])
        print(f'Is data normal for {stock}: {is_data_normal}')
        if is_data_normal:
            normal_stocks_bank.append([stock, list(data['Close'])])
        stocks_bank.append([stock, list(data['Close'])])

    normal_stocks_bank_pd = pd.DataFrame(normal_stocks_bank, columns=['Stock', 'Prices'])
    normal_stocks_bank_pd.to_csv('normal_stocks_bank7.csv', index=False)

    stocks_bank_pd = pd.DataFrame(stocks_bank, columns=['Stock', 'Prices'])
    stocks_bank_pd.to_csv('c_bank.csv', index=False)

    # test:
    diff_stock = pd.DataFrame(np.diff(normal_stocks_bank_pd.iloc[3]['Prices']))
    fig = px.histogram(diff_stock, x=0)
    fig.show()
    print('Done')
