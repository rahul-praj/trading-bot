# encoding: utf-8

# import needed libraries
from trading_bot import *
from logger import *
import sys

import alpaca_trade_api as tradeapi

import gvars

# check my trading account
def account_check(api):
    try:
        account = api.get_account()
        if account.status != 'ACTIVE':
            lg.error('Account is not active')
            sys.exit()
    except Exception as e:
        lg.error('Could not get account error')
        lg.info(str(e))
        sys.exit()

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

def check_asset_ok(api, ticker):
    # check asset is ok for trading
        # IN: ticker
    try:
        asset = api.get_asset(ticker)
        if asset.tradable:
            lg.info('Asset exists and is tradeable')
            return True
        else:
            lg.info('Asset exists but not tradeable')
    except Exception as e:
        lg.error('Asset does not exist, error')
        lg.error(e)
        sys.exit()


# execute trading bot
def main():

    api = tradeapi.REST(gvars.API_KEY, gvars.API_SECRET_KEY, gvars.API_URL)

    initialize_logger()

# check account
    account_check(api)

# close current orders
    clean_open_orders(api)

    # ticker = input('Ticker of asset: ')
    ticker = 'TSLA'

    import pdb; pdb.set_trace()

    check_asset_ok(api, ticker)

    trader = Trader(ticker, api)
    tradingSuccess = trader.run(ticker)

    if not tradingSuccess:
        lg.info('Trading was not successful, locking asset')

if __name__ == '__main__':
    main()


    # IN: string (ticker)
    # OUT: boolean (True = success / False = failure)
