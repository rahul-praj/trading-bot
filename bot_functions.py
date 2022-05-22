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
import tulipy as ti
from datetime import datetime, timedelta
from scipy.signal import argrelextrema
from logger import *
import sys, os, time, pytz

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

        self.stopLossMargin = 0.05 # percentage margin for stop stop loss
        self.takeProfitMargin = 0.1 # percentage margin for the take profit
        self.max_equity = 1000

    def historical_data(self, ticker, interval=1, limit=3000, n=10):

        timeNow = datetime.now(pytz.timezone('US/Eastern')) - timedelta(minutes = 15)
        timeStart = timeNow - timedelta(days = interval * limit)

        data = self.api.get_bars(ticker, TimeFrame(interval,TimeFrameUnit.Day), start=timeStart.isoformat(),
                                 end = timeNow.isoformat(), limit=limit).df

        data['normalised_price'] = (data['close'] - data['low']) / (data['high'] - data['low'])

        lmax = data.iloc[argrelextrema(data['close'].values, np.greater_equal, order = n)[0]]['close']
        lmin = data.iloc[argrelextrema(data['close'].values, np.less_equal, order = n)[0]]['close']

        data['lmax'] = pd.DataFrame(lmax)
        data['lmin'] = pd.DataFrame(lmin)

        data.reset_index(inplace=True)

        rsi = ti.rsi(data['close'].to_numpy(), 8)
        stoch_k, stoch_d = ti.stoch(data['high'].to_numpy(), data['low'].to_numpy(), data['close'].to_numpy(), 5, 3, 3)

        df = pd.DataFrame()

        df['rsi'] = pd.DataFrame(rsi)
        df['stoch_k'] = pd.DataFrame(stoch_k)
        df['stoch_d'] = pd.DataFrame(stoch_d)

        df1 = pd.DataFrame([[np.nan] * len(df.columns)], columns=df.columns)
        df1 = df1.loc[df1.index.repeat(8)].reset_index(drop=True)

        df = df1.append(df, ignore_index=True)

        data = pd.concat([data, df], axis=1)

        data['target'] = np.where(data['lmin'].isnull(), 1, 0)

        data.dropna(subset=['rsi', 'stoch_k', 'stoch_d', 'normalised_price'], inplace=True)

        return data

    def ml_function(self, ticker):

        data = self.historical_data(ticker, interval=1, limit=3000, n=10)

        while True:

            log = LogisticRegression()
            x_var = data[['normalised_price', 'volume', 'rsi', 'stoch_d']] ## need to add momentum indicator functions as well
            y_var = data['target']
            train_x, test_x, train_y, test_y = train_test_split(x_var, y_var, test_size=0.3)
            log.fit(train_x, train_y)

            # Use random search to optimise hyperparameters

            #Set up parameters to tune to for optimisation

            param_rand = {
                'C': np.linspace(1e-5, 1e4, 20),
                'penalty': ['l2'],
                'solver': ['lbfgs','liblinear','newton-cg'],
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

            if train_y_pred == 1:
                trend == 'short'
                if precision_short > 0.95:
                    return True
                else:
                    return False
            else:
                trend == 'long'
                if precision_long > 0.95:
                    return True
                else:
                    return False

            return trend

    def avg_entry_price(self, ticker):

        while True:
            try:
                position = self.api.get_position(ticker)
                entryPrice = float(position.avg_entry_price)
            except ValueError:
                return False

        return entryPrice

    def set_stoploss(self):

        stopLossMargin = self.stopLossMargin()

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

        stopLossMargin = self.stopLossMargin()

        takeProfitMargin = self.takeProfitMargin()

        entryPrice = self.avg_entry_price(ticker)

        if trend == 'long':
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

    def price_current(self, ticker):

        data = self.historical_data(ticker, interval=1, limit=3000, n=10)

        try:
            position = self.api.get_position(ticker)
            price = float(position.current_price)
        except ValueError:
            price = data['close'].iloc[-1:]

        return price

    def current_position(self, ticker):

        while True:
            try:
                position = self.api.get_position(ticker)
                lg.info('Open position exists for %s' % ticker)
            except:
                lg.info('No current open position for %s' % ticker)
                return False

        return position

    def execute_trade(self, ticker):

        # get average entry price
        entryPrice = self.get_avg_entry_price(ticker)

        price = self.price_current(ticker)

        # set the take profit
        takeProfit = self.set_takeprofit(entryPrice,self.takeProfitMargin)

        # set the stop loss
        stopLoss = self.set_stoploss(entryPrice,trend)

        max_equity = self.max_equity()

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

    #POINT DELTA: LOOP until timeout reached (ex. 2h)

        trend = self.ml_function(ticker)

        # POINT ECHO
        while True:
        # ask the broker/API if we have an open position with "ticker"
            if trend == 'long' and self.current_position(ticker):
                lg.info('Open position exists, hold %s' % ticker)
                return False # aborting execution

                self.execute_trade(ticker)

                time.sleep(10) # wait 10 seconds

                successfulOperation = lg.info('Trade complete')

                break
