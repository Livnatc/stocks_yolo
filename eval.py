import yfinance as yf

if __name__ == '__main__':

    # Get the holdings of the potential risers:
    with open('potential_risers.txt', 'r') as f:
        stocks_list = f.readlines()
        stocks_list = [x.strip() for x in stocks_list]

    rises_count = 0
    for stock in stocks_list:
        data = yf.download(stock, period="5d", interval='1d')
        if data['Close'][-1] > data['Close'][-2]:
            rises_count += 1
            print(f'{stock} is a potential riser')

    accuracy = rises_count / len(stocks_list)
    print(f'Accuracy: {accuracy}')
    print(f'Found {rises_count} potential risers out of {len(stocks_list)} stocks.')

