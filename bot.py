# encoding: utf-8

# import needed libraries
from trading_bot import *
from logger import *
import sys

# initialize the logger
initialize_logger()

# check my trading account
def account_check():
    try:
        # get account info
    except Exception as e:
        lg.error('Could not get account error')
        lg.info(str(e))
        sys.exit()

# close current orders
def clean_open_orders():
    # get list of open clean_open_orders
    lg.info('List of open orders')
    lg.info(str(open_orders))

    for order in clean_open_orders:
        #close order
        lg.info('Order %s closed' % str(order.id))

    lg.info('Closing orders complete')

# define assets
    # IN: keyboard
    # OUT:  string

# execute trading bot
def main():

    initialize_logger()

# check account
    account_check()

# close current orders
    clean_open_orders()

    ticker = input('Ticker of asset: ')

    trader = Trader(ticker)
    tradingSuccess = trader.run()

    if not tradingSuccess:
        lg.info('Trading was not successful, locking asset')

if __name__ == '__main__':
    main()


    # IN: string (ticker)
    # OUT: boolean (True = success / False = failure)
