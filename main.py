import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas_datareader as pdr
import datetime as dt
import matplotlib.pyplot as plt

# this chart shows  how two non-correlated variables look like
X = np.random.randn(5000)
Y = np.random.randn(5000)
fig, ax = plt.subplots()
ax.scatter(X, Y, alpha=.2)


tickers = ['AAPL', 'TWTR', 'IBM', 'MSFT', '^GSPC']
start = dt.datetime(2020, 1, 1)
data = pdr.get_data_yahoo(tickers, start)
data = data['Adj Close']
log_returns = np.log(data/data.shift())


# this function calculate and visualize Linear Regression, data has to be converted to Numpy
def linear_regression(ticker_a, ticker_b):
    X = log_returns[ticker_a].iloc[1:].to_numpy().reshape(-1, 1)
    Y = log_returns[ticker_b].iloc[1:].to_numpy().reshape(-1, 1)
    lin_regr = LinearRegression()
    lin_regr.fit(X, Y)
    Y_pred = lin_regr.predict(X)
    alpha = lin_regr.intercept_[0]
    beta = lin_regr.coef_[0, 0]
    fig, ax = plt.subplots()
    ax.set_title("Alpha: " + str(round(alpha, 5)) + ", Beta: " + str(round(beta, 3)))
    ax.scatter(X, Y)
    ax.plot(X, Y_pred, c='r')
    plt.show()


# linear regression between AAPL and MSFT
linear_regression('AAPL', 'MSFT')