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
from datetime import datetime, timedelta
from scipy.signal import argrelextrema
from logger import *
import sys, os, time, pytz
from sklearn import metrics

from alpha_vantage.timeseries import TimeSeries
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import REST, TimeFrame, TimeFrameUnit

BASE_URL = "https://paper-api.alpaca.markets"
API_KEY = "PK48PPBXKL5LD47Z784K"
API_SECRET_KEY = "2pFoHWpx45tzCSllAjYFrKSuIkZbLUXAkNCwJ0kX"
API_URL = "https://paper-api.alpaca.markets"

api = tradeapi.REST(key_id=API_KEY, secret_key=API_SECRET_KEY,
                    base_url=BASE_URL, api_version='v2')

alpha_vantage_api = 'GD6HUVV8J80WFW91'

class Trader:

    stopLossMargin = 0.05 # percentage margin for stop stop loss
    takeProfitMargin = 0.1 # percentage margin for the take profit
    max_equity = 1000

    def __init__(self, ticker, api):
        self.ticker = ticker
        self.api = api

    def historical_data(self, ticker, n=10):

        ts = TimeSeries(key=alpha_vantage_api, output_format='pandas')
        data, meta = ts.get_intraday(symbol = ticker, interval = '30min')

        data.reset_index(inplace=True)

        data.rename(columns={'1. open': 'open', '2. high': 'high', '3. low': 'low', '4. close': 'close', '5. volume': 'volume'}, inplace=True)

        data['normalised_price'] = (data['close'] - data['low']) / (data['high'] - data['low'])

        lmax = data.iloc[argrelextrema(data['close'].values, np.greater_equal, order = n)[0]]['close']
        lmin = data.iloc[argrelextrema(data['close'].values, np.less_equal, order = n)[0]]['close']

        data['lmax'] = pd.DataFrame(lmax)
        data['lmin'] = pd.DataFrame(lmin)

        data.reset_index(inplace=True)

        rsi = ti.rsi(data['close'].to_numpy(), 8)
        stoch_k, stoch_d = ti.stoch(data['high'].to_numpy(), data['low'].to_numpy(), data['close'].to_numpy(), 5, 3, 3)
        ema = ti.ema(data['close'].to_numpy(), 8)

        df = pd.DataFrame()

        df['rsi'] = pd.DataFrame(rsi)
        df['stoch_k'] = pd.DataFrame(stoch_k)
        df['stoch_d'] = pd.DataFrame(stoch_d)
        df['ema'] = pd.DataFrame(ema)

        df1 = pd.DataFrame([[np.nan] * len(df.columns)], columns=df.columns)
        df1 = df1.loc[df1.index.repeat(8)].reset_index(drop=True)

        df = df1.append(df, ignore_index=True)

        data = pd.concat([data, df], axis=1)

        data['target'] = np.where(data['lmin'].isnull(), 1, 0)

        data.dropna(subset=['rsi', 'stoch_k', 'stoch_d', 'normalised_price'], inplace=True)

        return data

    def ml_function(self, ticker):

        data = self.historical_data(ticker, n=10)

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

    def avg_entry_price(self, ticker):

        current_price = self.price_current(ticker)

        try:
            position = self.api.get_position(ticker)
            entryPrice = float(position.avg_entry_price)
        except:
            entryPrice = current_price

        return entryPrice

    def set_stoploss(self, ticker):

        entryPrice = self.avg_entry_price(ticker)

        if trend == 'long':
            stopLoss = entryPrice - (entryPrice * stopLossMargin)
            return stopLoss
        elif trend == 'short':
            stopLoss = entryPrice + (entryPrice * stopLossMargin)
            return stopLoss
        else:
            raise ValueError('Error returned at stoploss')

        return stopLoss

    def set_takeprofit(self, ticker):

        entryPrice = self.avg_entry_price(ticker)

        if trend == 'long':
            takeprofit = entryPrice + (entryPrice * takeProfitMargin)
            lg.info('Take profit set for long at %2f' % takeprofit)
            return takeprofit
        elif trend == 'short':
            takeprofit = entryPrice - (entryPrice * takeProfitMargin)
            lg.info('Take profit set for short at %2f' % takeprofit)
            return takeprofit
        else:
            raise ValueError('Error returned at takeprofit')

        return takeprofit

    def price_current(self, ticker):

        data = self.historical_data(ticker, n=10)

        try:
            position = self.api.get_position(ticker)
            price = float(position.current_price)
        except:
            price = data['close'].iloc[0]

        return price

    def current_position(self, ticker):

        try:
            position = self.api.get_position(ticker)
            lg.info('Open position exists for %s' % ticker)
        except:
            lg.info('No current open position for %s' % ticker)

        return position

    def execute_trade(self, ticker):

        price = self.price_current(ticker)

        try:
            # get average entry price
            entryPrice = self.avg_entry_price(ticker)
        except:
            entryPrice = price

        # set the take profit
        takeProfit = self.set_takeprofit(ticker)

        # set the stop loss
        stopLoss = self.set_stoploss(ticker)

        try:
            self.current_position(ticker)
            if trend == 'long':
                self.api.submit_order(
                    symbol = ticker,
                    qty = max_equity / price,
                    side = 'buy',
                    type = 'market',
                    time_in_force = 'gtc',
                    take_profit = {
                        limit_price: takeProfit
                    },
                    stop_loss = {
                        stop_price: stopLoss
                    }
                )
            else:
                pass
        except:
            if trend == 'short':
                pass
            else:
                self.api.close_position(
                    symbol = ticker,
                    percentage = 100
                )

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
