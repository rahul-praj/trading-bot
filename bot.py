# encoding: utf-8

# import needed libraries
from trading_bot import *
from logger import *
import sys

from alpaca_trade_api.rest import REST, TimeFrame

import gvars

# check my trading account
def account_check(api):
    try:
        account = api.get_account()
        if account.status == 'ACTIVE':
            lg.info('Account is not active')
            sys.exit()
    except Exception as e:
        lg.error('Could not get account error')
        lg.info(str(e))
        sys.exit()

    import pdb; pdb.set_trace()

# close current orders
def clean_open_orders(api):

    lg.info('Cancelling all orders')

    try:
        api.cancel_all_orders()
        lg.info('All orders cancelled')
    except Exception as e:
        lg.error('Error cancelling orders')
        lg.error(e)
        sys.exit()

# define assets
    # IN: keyboard
    # OUT:  string

# execute trading bot
def main():

    api = tradeapi.REST(gvars.API_KEY, gvars.API_SECRET_KEY, gvars.API_URL)

    initialize_logger()

# check account
    account_check(api)

# close current orders
    clean_open_orders(api)

    ticker = input('Ticker of asset: ')

    trader = Trader(ticker)
    tradingSuccess = trader.run()

    if not tradingSuccess:
        lg.info('Trading was not successful, locking asset')

if __name__ == '__main__':
    main()


    # IN: string (ticker)
    # OUT: boolean (True = success / False = failure)
