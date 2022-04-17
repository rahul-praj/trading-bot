# define ticker
# OUT: string

import alpaca_trade_api as tradeapi

import sys, os, time, pytz
import pandas as pd

from datetime import datetime
from math import ceil

class Trader():
    def __init__(self, ticker):
        lg.info('Trader initialized with ticker %s' % ticker)
        self.ticker = ticker

    def is_tradeable(self, ticker):

    # check if tradeable: ask the API if ticker is tradeable
      # IN: ticker (string)
      # OUT: True (exists) / False (does not exist)
      try:
          # get ticker from alpaca wrapper
          if not ticker.tradeable:
              lg.info('The ticker %s is not tradeable' % ticker)
              return FALSE
          else:
              lg.info('The ticker %s is tradeable!' % ticker)
              return True
      except:
          lg.error('The ticker %s is not answering well' % ticker)
          return False

    def set_stoploss(self, entryPrice, trend):
        # check stoploss: takes price as an input and sets the stoploss
            # IN: entry price
            # OUT: stop loss

        try:
            if trend == 'long':
                stopLoss = entryPrice - (entryPrice * gvars.stopLossMargin)
                return stopLoss
            elif trend == 'short':
                stopLoss = entryPrice + (entryPrice * gvars.stopLossMargin)
                return stopLoss
            else:
                raise ValueError

        except Exception as e:
            lg.error('The trend is undefined: %s' % str(trend))
            sys.exit()

        return stopLoss

    def set_takeprofit(self, entryPrice, trend):
    # set takeprofit: takes the price as an input and sets the takeprofit
        # IN: entry price
        # OUT: take profit

        try:
            if trend == 'long':
                takeprofit = entryPrice + (entryPrice * gvars.takeProfitMargin)
                lg.info('Take profit set for long at %2f' % takeprofit)
                return takeprofit
            elif trend == 'short':
                takeprofit = entryPrice - (entryPrice * gvars.takeProfitMargin)
                lg.info('Take profit set for short at %2f' % takeprofit)
                return takeprofit
            else:
                raise ValueError

        except Exception as e:
            lg.error('The trend is undefined: %s' % str(trend))
            sys.exit()

        return takeprofit

    # load historical stock data
        # IN: ticker, interval, entries limit
        # OUT: array with stock data (OHCL)

    def get_open_positions(self, tickerID):
    # get open positions
        # IN: tickerID
        # OUT: boolean (True = already open, False = Not open)
    # positions: ask alpaca wrapper for list of open positions
        for position in positions:
            if position.symbol == tickerID:
                return True
            else:
                return False

    # submit order
        # IN: Order data, order type
        # OUT: boolean (True = order went through, False = Order did not go through)

    # cancel orders
        # IN: order ID
        # OUT: boolean (True = order cancelled, False = order not cancelled)

    def check_position(self, ticker, notFound=False):
    # check position: check if position is open
        # IN: ticker
        # OUT: boolean (True = order is there, False = order not there)
        attempt = 1

        while attempt < gvars.maxAttemptsCP:
            try:
                currentPrice = position.current_Price
                lg.info('Position checked. Current price is %.2f' % currentPrice)
                return True
            except:

                if notFound:
                    lg.info('Position not found')
                    return False

                lg.info('Position not found, cannot check price, waiting...')
                time.sleep(gvars.sleepTimeCP) # wait 5 seconds
                attempt = attempt + 1

        lg,info('Position not found for %s, not waiting anymore' % ticker)
        return False

    def get_shares_amount(self, tickerPrice):
        # works out the number of shares I want to buy/sell
            # IN: tickerPrice
            # OUT: number of shares

        try:
            # get the total equity available
            # total equity = ask API for available equity

            # calculate the number of shares
            totalShares = int(gvars.maxSpentEquity / tickerPrice)

            return sharesQuantity

        except Exception as e:
            lg.error('Error at get shares amount')
            lg.error(e)
            sys.exit()

    def get_current_price(self, ticker):

        attempt = 1

        while attempt < gvars.maxAttemptsCP:
            try:
                currentPrice = position.current_Price
                lg.info('Position checked. Current price is %.2f' % currentPrice)
                return currentPrice
            except:
                lg.info('Position not found, waiting...')
                time.sleep(gvars.sleepTimeGCP) # wait 5 seconds
                attempt = attempt + 1

        lg.info('Position not found for %s, not waiting anymore' % ticker)
        return False

    def get_general_trend(self, ticker):
    # perform general trend analysis: detect interesting trend
        # IN: ticker
        # OUT: string(UP/DOWN/NO/TREND)

        lg.info('GENERAL TREND ANALYSIS ENTERED')

        attempt = 1
        maxAttempts = 10 # total time = maxAttempts * 10 min (as implemented)

        try:
            while True:

                # data = ask Alpaca for 30 min candles

                ema9 = ti.ema(data, 9)
                ema26 = ti.ema(data, 26)
                ema50 = ti.ema(data, 50)
                if (ema50 > ema26) and (ema26 > ema9):
                    lg.info('Trend detected for %s: long' % ticker)
                    return 'long'
                elif (ema50 < ema26) and (ema26 < ema9):
                    lg.info('Trend detected for %s: short' % ticker)
                    return 'short'
                elif attempt <= maxAttempts:
                    lg.info('Trend not clear for %s, waiting...' % ticker)
                    time.sleep(60*10)
                else:
                    lg.info('No trend detected for %s' % ticker)
                    return False
        except Exception as e:
            lg.error('Something when wrong at general trend')
            lg.error(e)
            sys.exit()

    def get_instant_trend(self, ticker, trend):

        lg.info('INSTANT ANALYSIS ENTERED')

        attempt = 1
        maxAttempts = 10

        try:
            while True:
                ema9 = ti.ema(data, 9)
                ema26 = ti.ema(data, 26)
                ema50 = ti.ema(data, 50)

                lg.info('%s instant trend EMAs = [%.2f, %.2f, %.2f]' % (ticker, ema9, ema26, ema50))

                if (trend == 'long') and (ema9 > ema26) and (ema26 > ema50):
                    lg.info('Long trend confirmed for %s' % ticker)
                    return True
                elif (trend == 'short') and (ema9 < ema26) and (ema26 < ema50):
                    lg.info('Short trend confirmed for %s' % ticker)
                    return False
                elif attempt <= maxAttempts:
                    lg.info('Position not found, waiting...')
                    time.sleep(30) # wait 5 minutes
                    attempt = attempt + 1
                else:
                    lg.info('Trend not detected for %s and timeout reached' % ticker)
        except Exception as e:
            lg.error('Something when wrong at instant trend')
            lg.error(e)
            sys.exit()

    # get instant trend: cofnrim the trend by the GT analysis
        # IN: output of general trend analysis (string), 5 min candle data
        # OUT: TRUE (confirmed) / FALSE (not confirmed)
        #IF fails, go back to start of loop

    def get_rsi(self, ticker, trend):

        lg.info('RSI ENTERED')

        attempt = 1
        maxAttempts = 10

        try:
            while True:
                rsi = ti.rsi(data, 14)

                lg.info('%s rsi = [%.2f]' % (ticker, rsi))

                if (trend == 'long') and (rsi > 50) and (rsi < 80):
                    lg.info('Long trend confirmed for %s' % ticker)
                    return True
                elif (trend == 'short') and (rsi < 50) and (rsi > 20):
                    lg.info('Short trend confirmed for %s' % ticker)
                    return False
                elif attempt <= maxAttempts:
                    lg.info('Position not found, waiting...')
                    time.sleep(20) # wait 5 minutes
                    attempt = attempt + 1
                else:
                    lg.info('Trend not detected for %s and timeout reached' % ticker)
        except Exception as e:
            lg.error('Error at rsi analysis')
            lg.error(e)
            sys.exit()

        # get RSI: perform RSI analysis
             # IN: output of general trend analysis (string), 5 min candle data
             # OUT: TRUE (confirmed) / FALSE (not confirmed)
             # IF fails, go back to start of loop

    def get_stochastic(self, ticker, trend):

        lg.info('STOCHASTIC ANALYSIS ENTERED')

        attempt = 1
        maxAttempts = 20

        try:
            while True:
                stoch_d, stoch_k = ti.stoch(high, low, close, 9, 6, 9)

                lg.info('%s stochastic = [%.2f, %.2f]' % (ticker, stoch_h, stoch_d))

                if (trend == 'long') and (stoch_k > stoch_d) and (stoch_k < 80) and (stoch_d < 80):
                    lg.info('Long trend confirmed for %s' % ticker)
                    return True
                elif (trend == 'short') and (stoch_k < stoch_d) and (stoch_k > 20) and (stoch_d > 20):
                    lg.info('Short trend confirmed for %s' % ticker)
                    return False
                elif attempt <= maxAttempts:
                    lg.info('Position not found, waiting...')
                    time.sleep(10) # wait 5 minutes
                    attempt = attempt + 1
                else:
                    lg.info('Trend not detected for %s and timeout reached' % ticker)
        except Exception as e:
            lg.error('Error at stochastic analysis')
            lg.error(e)
            sys.exit()
    # get stochastic: perform stochastic analysis
         # IN: ticker, trend
         # OUT: TRUE (confirmed) / FALSE (not confirmed)
         # IF fails, go back to start of loop

    # enter position mode: check the positions in parallel once inside the position

    def check_stochastic_crossing(self, ticker, trend):
        stoch_d, stoch_k = ti.stoch(high, low, close, 9, 6, 9)

        lg.info('%s stochastic = [%.2f, %.2f]' % (ticker, stoch_h, stoch_d))

        try:
            if (trend == 'long') and (stoch_k <= stoch_d):
                lg.info('Stochastic curves crossed: long k=%.2f, d=%.2f' % (stoch_k, stoch_d))
                return True
            elif (trend == 'short') and (stoch_d > stoch_k):
                lg.info('Stochastic curves crossed: short k=%.2f, d=%.2f' % (stoch_k, stoch_d))
                return True
            else:
                lg.info('Stochastic curves have not crossed')
                return False

        except Exception as e:
            lg.error('Stochastic crossing check error')
            lg.error(e)
            return True

    def enter_position_mode(self, ticker, trend):

        # entryPrice = ask Alpaca API for entry price

        takeProfitMargin = set_takeprofit(entryPrice, trend)

        # set the stop loss
        stopLoss = set_stoploss(entryPrice, trend)

        try:
            while True:

                currentPrice = get_current_price(ticker)

                if (trend == 'long') and (currentPrice >= takeprofit):
                    lg.info('Take profit met at %.2f. Current price is %2f' % (takeprofit, currentPrice))
                    return True

                elif (trend == 'short') and (currentPrice < takeprofit):
                    lg.info('Take profit met at %.2f. Current price is %2f' % (takeprofit, currentPrice))
                    return True

                elif (trend == 'long') and (currentPrice <= stopLoss):
                    lg.info('Stop loss met at %.2f. Current price is %.2f.' % (stopLoss, currentPrice))
                    return False

                elif (trend == 'short') and (currentPrice > stopLoss):
                    lg.info('Stop loss met at %.2f. Current price is %.2f.' % (stopLoss, currentPrice))
                    return False

                elif check_stochastic_crossing(self, ticker, trend):
                    lg.info('Stochastic curves crossed. Current price is %.2f' % currentPrice)

                elif attempt <= maxAttempts:
                    lg.info('Waiting inside position...')
                    time.sleep(20)

                # get out, time is out
                else:
                    lg.info('Timeout reached at enter position, too late')
                    return True

        except Exception as e:
            lg.error('Error at enter position function')
            lg.error(e)
            return True

    def run(self):

        while True:
        # check position: ask API if we have an open position of the ticker
          # IN: ticker (string)
          # OUT: True (exists) / False (does not exist)

          if check_position(self.ticker, notFound = True):
            lg.info('Open position exists, aborting...')
            return False

            #POINT DELTA

            while True:

                # get general trend: find a trend
                trend = get_general_trend(self.ticker)
                if not trend:
                    lg.info('No general trend found for %s, aborting...' % self.ticker)
                    return False
                    # LOOP

                # get instant trend
                if not get_instant_trend(self.ticker, trend):
                    lg,info('Instant trend failed')
                    # if failed go back to POINT DELTA (i.e. go back to the beginning)
                    continue

                #  perform RSI analysis
                if not get_rsi(self.ticker, trend):
                    lg,info('RSI not confirmed')
                    # if failed go back to POINT DELTA (i.e. go back to the beginning)
                    continue

                # perform stochastic analysis
                    # if failed go back to POINT DELTA (i.e. go back to the beginning)
                if not get_stochastic(self.ticker, trend):
                    lg,info('Stochastic not confirmed')
                    # if failed go back to POINT DELTA (i.e. go back to the beginning)
                    continue

                lg.info('All filters passed, continuing with order')
                break

            # get current price
            self.currentPrice = get_current_price(self.ticker)

            # get_shares_amount: decide the total amount to invest
            sharesQuantity = get_shares_amount(self.ticker, self.currentPrice)

            # get equity: decide the total amount to invest

            # submit order (limit order)
              # if false, retry until works

            # check POSITION
            if not check_position(self.ticker):
              # if false, go back to submit order
              continue

            # enter position mode
            successfulOperation = enter_position_mode(ticker, trend)

            # GET OUT
            while True:
                # SUBMIT ORDER (market order)

                #check the position is cleared
                if not check_position(self.ticker, notFound=True):
                    break

                time.sleep(10)

        # end of execution
        return successfulOperation

    # wait 15 mins
    # back to beginning
