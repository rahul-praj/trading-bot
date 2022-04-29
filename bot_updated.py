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

import alpaca_trade_api as tradeapi

BASE_URL = "https://paper-api.alpaca.markets"
API_KEY = "PKZMYX5AINHNL0A52FKU"
API_SECRET_KEY = "cB9IqfVvYPIDXoyJF48b8FwPq0ClNmXj6GHxsDoN"
API_URL = "https://paper-api.alpaca.markets"



api = tradeapi.REST(key_id=API_KEY, secret_key=API_SECRET_KEY,
                    base_url=BASE_URL, api_version='v2')

## set take profit and stop loss

stopLossMargin = 0.05 # percentage margin for stop stop loss

takeProfitMargin = 0.1 # percentage margin for the take profit

maxSpentEquity = 1000 # total equity to spend in a single operation

class Trader:

    def __init__(self, ticker, api):
        self.ticker = ticker
        self.api = api

    def trade_type(self, ticker, interval, limit):

        timeNow = datetime.now(pytz.timezone('US/Eastern'))
        timeStart = timeNow - timedelta(minutes = interval * limit)
        data = self.api.get_bars(ticker, TimeFrame(interval, TimeFrameUnit.Minute), start=timeStart.isoformat(),
                                 end = timeNow.isoformat(), limit=limit).df

    def ml_function(self, data):

            while True:

                # pass through historical data from api here

                log = LogisticRegression(solver = 'liblinear')
                train_x, test_x, train_y, test_y = train_test_split(data, test_size=0.3)
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
                classification = classification_report(train_y, train_y_pred, target_names=target_names, output_dict=True) # double asterisk allows us to take key-value pairs in dictionary and unpack

                precision = classification['class_0']['precision']

                # for the long position
                if precision > 0.95:
                    return True
                else:
                    return False

                if train_y_pred < data['close'].iloc[-1]:
                    trend = 'short'
                elif train_y_pred > data['close'].iloc[-1]:
                    trend = 'long'
                else:
                    sys.exit()


    def set_stoploss(self, entryPrice, trend):

        stopLossMargin = 0.05

        if trend == 'long':
            stopLoss = entryPrice - (entryPrice * stopLossMargin)
            return stopLoss
        elif trend == 'short':
            stopLoss = entryPrice + (entryPrice * stopLossMargin)
            return stopLoss
        else:
            raise ValueError('Error returned at stoploss')

        return stopLoss

    def set_takeprofit(self, entryPrice, trend):

        takeProfitMargin = 0.1

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
