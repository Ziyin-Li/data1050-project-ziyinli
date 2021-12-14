import pandas_datareader as web
import datetime

stock_names=['AAPL','ADBE','ADP','AMAT','AMD','AMZN','AVGO','CRM','CSCO','EBAY','FB','GOOG','IBM','INTC','INTU','LRCX','MSFT','NFLX','NVDA','ORCL','PYPL','QCOM','TSLA','TWTR','TXN']

end = datetime.datetime.today() 
start = datetime.date(end.year-10,1,1) # collects data that are up to 10 years old

# +
# loop through the total of 25 different stocks in the stock_names list
# if no historical data returned on any pass, try to get the data again
# append historical data to the file

for i in range(0,len(stock_names)):
    try:
        df = web.DataReader(stock_names[i], 'yahoo', start, end)
        df.insert(0,'Stock',stock_names[i])
        df = df.drop(['Adj Close'], axis=1)
        df.to_csv('./stock_data.csv',mode = 'a',header=False)
    except Exception:
        continue
# -


