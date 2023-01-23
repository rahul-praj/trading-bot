# encoding: utf-8

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import pytz
import tulipy as ti
from datetime import datetime, timedelta, date
from scipy.signal import argrelextrema
from logger import *
import sys, os, time, pytz
from sklearn import metrics
import yfinance as yf
from pandas_datareader import data as pdr

from alpha_vantage.timeseries import TimeSeries
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import REST, TimeFrame, TimeFrameUnit

BASE_URL = "https://paper-api.alpaca.markets"
API_KEY = "PKG9L2NYONVTK05ZTI75"
API_SECRET_KEY = "NeJpDtk99QqGDBDBMcN53b29gX3koFdQl5wnll9w"

api = tradeapi.REST(key_id=API_KEY, secret_key=API_SECRET_KEY,
                    base_url=BASE_URL, api_version='v2')

#alpha_vantage_api = 'GD6HUVV8J80WFW91'

class Trader:

    stopLossMargin = 0.05 # percentage margin for stop stop loss
    takeProfitMargin = 0.1 # percentage margin for the take profit
    max_equity = 1000

    def __init__(self, ticker, api):
        self.ticker = ticker
        self.api = api

    def historical_data(self, ticker, n=10):

        #ts = TimeSeries(key=alpha_vantage_api, output_format='pandas')
        #data, meta = ts.get_intraday(symbol = ticker, interval = '30min')

        startdate = pd.to_datetime('2018-07-15')
        enddate = pd.to_datetime(date.today())

        data = data.DataReader(ticker, 'stooq', startdate, enddate)

        data.reset_index(inplace=True)

        data = data.sort_values(by='Date')

        data['normalised_price'] = (data['Close'] - data['Low']) / (data['High'] - data['Low'])

        #lmax = argrelextrema(np.array(data['close']), np.greater_equal, order = 15)
        #lmin = argrelextrema(np.array(data['close']), np.less_equal, order = 15)

        lmax = data.iloc[argrelextrema(data['Close'].values, np.greater_equal, order = 15)[0]]['Close']
        lmin = data.iloc[argrelextrema(data['Close'].values, np.less_equal, order = 15)[0]]['Close']

        data['lmax'] = pd.DataFrame(lmax)
        data['lmin'] = pd.DataFrame(lmin)

        data['lmax'].fillna(0, inplace=True)
        data['lmin'].fillna(0, inplace=True)

        conditions = [
            (data['lmax'] != 0) & (data['lmin'] == 0),
            (data['lmax'] == 0) & (data['lmin'] != 0),
            (data['lmax'] == 0) & (data['lmin'] == 0)
        ]

        output = [
            2,
            1,
            0
        ]

        data['target'] = np.select(conditions, output)

        #data['target'] = np.where(data['lmax'].isnull(), 0, 1)
        #data['target'] = np.where(data['lmin'].isnull(), 0, 1)

        #data.loc[((data['lmax'] != 0) & (data['lmin'] != 0)), 'target'] == "1"

        data1 = data[data['target'] == 1]
        data2 = data[data['target'] == 2]

        # add momentum indicators to the dataframe

        #rsi
        rsi = ti.rsi(data['Close'].to_numpy(), 8)
        data['rsi'] = pd.DataFrame(rsi)

        #ema
        ema = ti.ema(data['Close'].to_numpy(),)
        data['ema'] = pd.DataFrame(ema)

        #


        fig, ax = plt.subplots()
        ax.plot(data['Date'], data['Close'])
        plt.title("Stock Price of {}".format(ticker))
        plt.ylabel("Closing Price ($)")
        plt.scatter(data1['Date'], data1['Close'], marker = 'o', s=50, facecolors = 'none', color='green')
        plt.scatter(data2['Date'], data2['Close'], marker = 'o', s=50, facecolors = 'none', color='red')

        plt.show()

        return data

    def ml_function(self, ticker):

        data = self.historical_data(ticker, n=10)

        # needs a machine learning algo that incorporates momentum indicators

        while True:

            lin = LinearRegression()
            x_var = data[['ema', 'volume', 'rsi', 'stoch_d']] ## need to add momentum indicator functions as well
            y_var = data['close']
            train_x, test_x, train_y, test_y = train_test_split(x_var, y_var, test_size=0.3)
            lin.fit(train_x, train_y)

            training_accuracy = lin.score(train_x, train_y)
            test_accuracy = lin.score(test_x, test_y)

            if (test_accuracy >= 0.9) or (training_accuracy - test_accuracy < 0.15):
                return True
            else:
                return False

            train_y_pred = lin.predict(train_x)

            r_square = metrics.r2_score(train_y, train_y_pred, squared=False)

            if r_square > 0.70:
                if train_y_pred < data['close'].iloc[0]:
                    trend == 'short'
                elif train_y_pred < data['close'].iloc[0]:
                    trend == 'long'
            else:
                return False
                sys.exit()

            return trend

    def run(self,ticker):
        # POINT ECHO
        while True:
            trend = self.ml_function(ticker)
        # ask the broker/API if we have an open position with "ticker"
            if trend == 'long' and self.current_position(ticker):
                lg.info('Open position exists, hold %s' % ticker)
                return False # aborting execution
            else:
                self.execute_trade(ticker)

                time.sleep(10) # wait 10 seconds

                successfulOperation = lg.info('Trade complete')

                time.sleep(60*60)
