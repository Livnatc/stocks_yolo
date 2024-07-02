import streamlit as st
import pandas as pd
import datetime
import os
import matplotlib.pyplot as plt
from fundemantals import Fundamentals


if __name__ == '__main__':
    st.title('Stock Market Analysis')
    st.write('This app uses moving average crossover strategy to find potential risers in the stock market.')

    # Load the results from the ema_strategy.py
    path_dir = 'potential_risers_csv'
    f = Fundamentals()
    for file in os.listdir(path_dir):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(path_dir, file))

            # Display the results
            st.write(f'### Results for {file}')
            fig, ax = plt.subplots()
            plt.style.use('seaborn-v0_8-pastel')
            revenue = list(df.iloc[:, 2])
            stocks = list(df.iloc[:, 1])
            ax.hist(revenue, bins=20)
            st.pyplot(fig)
            st.write('---')
            succcess_stocks = len([x for x in revenue if x > 1])
            st.write(f'Total sucsess {succcess_stocks} out of {len(df)}, accuracy: {succcess_stocks / len(df)}')
            st.write('### Potential Risers Fundamentals ###')

            fund_df = pd.DataFrame()
            for i, stock in enumerate(stocks):
                try:
                    f.get_stock_data(stock)
                    sales_inc = f.get_sales_increase()
                    erning_inc = f.get_earning_increase()
                    debt = f.get_debt_lower()
                    roe = f.get_roe()
                    peg = f.get_peg()
                    ps = f.get_ps()
                    pb = f.get_pb()
                    ev = f.get_ev()
                    pfcf = f.get_pfcf()
                except:
                    pass

                new_row = {'Stock': stock, 'Sales Increase': sales_inc, 'Earning Increase': erning_inc, 'Debt': debt, 'ROE': roe, 'PEG': peg, 'P/S': ps, 'P/B': pb, 'EV/EBITDA': ev, 'P/FCF': pfcf , 'Revenue': revenue[i]}
                # Convert the new row to a DataFrame
                new_row_df = pd.DataFrame([new_row])

                # Append the new row using concat
                fund_df = pd.concat([fund_df, new_row_df], ignore_index=True)
                # st.write(f'{stock} Sales Increase: {sales_inc}, Earning Increase: {erning_inc}, Debt: {debt}, ROE: {roe}, PEG: {peg}, P/S: {ps}, P/B: {pb}, EV/EBITDA: {ev}, P/FCF: {pfcf}')
            st.write(fund_df)
            st.write('---')

