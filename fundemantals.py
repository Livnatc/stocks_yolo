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
'''


class Fundamentals:
    def __init__(self):
        self.stocks_list = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0].Symbol.to_list()
        self.data = None

    def get_stock_data(self, stock):
        try:
            self.data = yf.Ticker(stock)
        except:
            self.data = None
        return self.data

    def get_sales_increase(self):
        try:
            sales_inc = self.data.financials.loc['Total Revenue'].iloc[1] / self.data.financials.loc['Total Revenue'].iloc[2]
            return sales_inc > 1.1
        except:
            return 0

    def get_earning_increase(self):
        try:
            erning_inc = self.data.financials.loc['Net Income'].iloc[1] / self.data.financials.loc['Net Income'].iloc[2]
            return erning_inc > 1.1
        except:
            return 0

    def get_debt_lower(self):
        try:
            debt = self.data.balance_sheet.loc['Long Term Debt'].iloc[1] < self.data.balance_sheet.loc['Long Term Debt'].iloc[2]
            return debt
        except:
            return None

    def get_roe(self):
        try:
            total_assets = self.data.balance_sheet.loc['Total Assets'].iloc[1]
            total_liabilities = self.data.balance_sheet.loc['Stockholders Equity'].iloc[1]
            shareholders_equity = total_assets - total_liabilities
            roe = self.data.financials.loc['Net Income'].iloc[1] / shareholders_equity
            return roe > 1.1
        except:
            return 0

    def get_peg(self):
        try:
            return self.data.info['pegRatio']
        except:
            return 0

    def get_ps(self):
        try:
            return self.data.info['priceToSalesTrailing12Months']
        except:
            return 0

    def get_pb(self):
        try:
            return self.data.info['priceToBook']
        except:
            return 0

    def get_ev(self):
        try:
            return self.data.info['enterpriseToEbitda']
        except:
            return 0

    def get_pfcf(self):
        try:
            return self.data.info['open'] / (self.data.cashflow.loc['Free Cash Flow'].iloc[0] / self.data.info.get('sharesOutstanding'))
        except:
            return 0


if __name__ == '__main__':

    f = Fundamentals()
    for st in f.stocks_list:
        try:
            f.get_stock_data(st)
            sales_inc = f.get_sales_increase()
            erning_inc = f.get_earning_increase()
            debt = f.get_debt_lower()
            roe = f.get_roe()
            peg = f.get_peg()
            ps = f.get_ps()
            pb = f.get_pb()
            ev = f.get_ev()
            pfcf = f.get_pfcf()

            print(st, sales_inc, erning_inc, debt)
        except:
            pass


