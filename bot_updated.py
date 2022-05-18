# encoding: utf-8

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import pytz
import datetime
import newtulipy as ti
from datetime import datetime, timedelta
from scipy.signal import argrelextrema
from logger import *

import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import REST, TimeFrame, TimeFrameUnit

BASE_URL = "https://paper-api.alpaca.markets"
API_KEY = "PK48PPBXKL5LD47Z784K"
API_SECRET_KEY = "2pFoHWpx45tzCSllAjYFrKSuIkZbLUXAkNCwJ0kX"
API_URL = "https://paper-api.alpaca.markets"

api = tradeapi.REST(key_id=API_KEY, secret_key=API_SECRET_KEY,
                    base_url=BASE_URL, api_version='v2')

class Trader:

    def __init__(self, ticker, api):
        self.ticker = ticker
        self.api = api

        self.data = None
        self.trend = None

        self.stopLossMargin = 0.05 # percentage margin for stop stop loss
        self.takeProfitMargin = 0.1 # percentage margin for the take profit
        self.maxSpentEquity = 1000 # total equity to spend in a single operation

    def historical_data(self, ticker, interval=1, limit=3000, data):

        timeNow = datetime.now(pytz.timezone('US/Eastern')) - timedelta(minutes = 15)
        timeStart = timeNow - timedelta(days = interval * limit)

        data = self.api.get_bars(ticker, TimeFrame(interval,TimeFrameUnit.Day), start=timeStart.isoformat(),
                                 end = timeNow.isoformat(), limit=limit).df

        data['normalised_price'] = (data['close'] - data['low']) / (data['high'] - data['low'])

        self.data = data

        return data

    def ml_function(self, ticker, interval, limit, n=10, trend):

        data = self.data

        while True:

            self.data['lmax'] = self.data.iloc[argrelextrema(self.data['close'].values, np.greater_equal, order = n)[0]]['close']
            self.data['lmin'] = self.data.iloc[argrelextrema(self.data['close'].values, np.less_equal, order = n)[0]]['close']

            self.data['rsi'] = ti.rsi(self.data['close'], 7)
            self.data['stoch_k'], self.data['stoch_d'] = ti.stoch(self.data['high'], self.data['low'], self.data['close'], 5, 3, 3)

            data_1 = self.data.dropna(subset = ['lmax', 'lmin', 'rsi', 'stoch_d'], how='all')

            if data_1['lmin'] is None:
                data_1['target'] == 1
            else:
                data_1['target'] == 0

            log = LogisticRegression(solver = 'liblinear')
            x_var = data_1[['normalised_price', 'volume', 'rsi', 'stoch_d']] ## need to add momentum indicator functions as well
            y_var = data_1['target']
            train_x, test_x, train_y, test_y = train_test_split(x_var, y_var, test_size=0.3)
            log.fit(train_x, train_y)

            # Use random search to optimise hyperparameters

            #Set up parameters to tune to for optimisation

            param_rand = {
                'C': np.linspace(1e-5, 1e4, 20),
                'penalty': ['l1','l2','none'],
                'solver': ['lbfgs', 'liblinear','newton-cg'],
            }

            #Set up random search parameters to tune

            random_cv = RandomizedSearchCV(log, param_rand, cv = 5)

            random_cv.fit(train_x, train_y)

            training_accuracy = random_cv.score(train_x, train_y)
            test_accuracy = random_cv.score(test_x, test_y)

            if (test_accuracy >= 0.9) or (training_accuracy - test_accuracy < 0.15):
                return True
            else:
                return False

            train_y_pred = random_cv.predict(train_x)
            target_names = ['class_0', 'class_1']
            classification = classification_report(train_y, train_y_pred, target_names=target_names, output_dict=True) # output_dict allows us to take key-value pairs in dictionary and unpack

            precision_long = classification['class_0']['precision']
            precision_short = classification['class_1']['precision']

            if train_y_pred < data['close'].iloc[-1]:
                trend == 'short'
                if precision_short > 0.95:
                    return True
                else:
                    return False
            elif train_y_pred > data['close'].iloc[-1]:
                trend == 'long'
                if precision_long > 0.95:
                    return True
                else:
                    return False
            else:
                sys.exit()

            self.trend = trend

            return trend

    def set_stoploss(self, entryPrice, stopLossMargin):

        if self.trend == 'long':
            stopLoss = entryPrice - (entryPrice * stopLossMargin)
            return stopLoss
        elif self.trend == 'short':
            stopLoss = entryPrice + (entryPrice * stopLossMargin)
            return stopLoss
        else:
            raise ValueError('Error returned at stoploss')

        return stopLoss

    def set_takeprofit(self, entryPrice, takeProfitMargin):

        if self.trend == 'long':
            takeprofit = entryPrice + (entryPrice * takeProfitMargin)
            lg.info('Take profit set for long at %2f' % takeprofit)
            return takeprofit
        elif self.trend == 'short':
            takeprofit = entryPrice - (entryPrice * takeProfitMargin)
            lg.info('Take profit set for short at %2f' % takeprofit)
            return takeprofit
        else:
            raise ValueError('Error returned at takeprofit')

        return takeprofit

    def avg_entry_price(self, ticker):

        while True:
            try:
                position = self.api.get_position(ticker)
                entryPrice = float(position.avg_entry_price)
            except ValueError:
                return False

        return entryPrice

    def price_current(self, ticker):

        try:
            position = self.api.get_position(ticker)
            price = float(position.current_price)
        except ValueError:
            price = self.data['close'].iloc[-1:]

        return price

    def current_position(self, ticker):

        while True:
            try:
                position = self.api.get_position(ticker)
                lg.info('Open position exists for %s' % ticker)
                break
            except ValueError:
                return False
                lg.info('No current open position for %s' % ticker)

         return position

    def open_position(self, ticker):

        trend = self.trend

        # get average entry price
        entryPrice = self.get_avg_entry_price(ticker)

        # set the take profit
        takeProfit = self.set_takeprofit(entryPrice,takeProfitMargin)

        # set the stop loss
        stopLoss = self.set_stoploss(entryPrice,trend)

        position = self.current_position(ticker)

        while True:
            try:
                self.api.get_position(ticker)
                if trend == 'long':
                    pass
                else:
                    self.api.submit_order(
                        symbol = ticker
                        notional = 2000
                        side = 'sell'
                        type = 'market'
                        time_in_force = 'gtc'
            except ValueError:
                if trend == 'long':
                    self.api.submit_order(
                        symbol = ticker
                        notional = 2000
                        side = 'buy'
                        type = 'market'
                        time_in_force = 'gtc'
                    )
                else:
                    pass
                    return False

    def exit_position(self, ticker):

        while True:

            try:
                self.api.close_position(ticker)
            except ValueError:
                return False
                sys.exit()

        self.api.close_position(
            symbol = ticker
            percentage = 100
        )
