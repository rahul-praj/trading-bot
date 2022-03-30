# define asset
# OUT: string

# check position: ask API if we have an open position of the asset
  # IN: Asset (string)
  # OUT: True (exists) / False (does not exist)

# check if tradeable: ask the API if asset is tradeable
  # IN: Asset (string)
  # OUT: True (exists) / False (does not exist)

# load 30 min candles: ask API for 30 min candles
  # IN: asset, time range, candle size
  # OUT: open, high, close and low data for the candles

# perform general trend analysis: detect interesting trend
  # IN: 30 min candle data (close)
  # OUT: string(UP/DOWN/NO/TREND)

# LOOP

# STEP 1: load 5 min candles
  # IN: asset, time range, candle size
  # OUT: open, high, close and low data for the candles
  #IF fails, go back to start of loop

# STEP 2: get instant trend analysis
  # IN: output of general trend analysis (string), 5 min candle data
  # OUT: TRUE (confirmed) / FALSE (not confirmed)
  #IF fails, go back to start of loop

# STEP 3: perform RSI analysis
  # IN: output of general trend analysis (string), 5 min candle data
  # OUT: TRUE (confirmed) / FALSE (not confirmed)
  #IF fails, go back to start of loop

# STEP 4: perform stochastic analysis
  # IN: output of general trend analysis (string), 5 min candle data
  # OUT: TRUE (confirmed) / FALSE (not confirmed)
  #IF fails, go back to start of loop


# SUBMIT order:
# submit order (limit order)
  # IN: Number of shares to buy/ sell, asset, desired price
  # OUT: TRUE (confirmed) / FALSE (not confirmed), position ID
# check POSITION
  # IN: Position ID
  # OUT: OUT: TRUE (confirmed) / FALSE (not confirmed)

# LOOP until timeout reached (8h)

# ENTER POSITION
  # IF take profit --> close position
    # IN: Current gains (losses)
    # OUT: True / False
  # ELIF stop loss TRUE --> close POSITION
    # IN: Current gains (losses)
    # OUT: True / False
  # ELIF check stoch crossing (pull OHLC data) IF TRUE --> close POSITION
    #Step 1: Pull 5 min OHLC data
    # IN: Asset
    # OUT: OHLC data (5 min candles)

    #Step 2: see whether stochastic curves are crossing
    # IN: OHLC data
    # OUT: True / False

# GET OUT
# SUBMIT ORDER (market order)
# submit order: interact with API
  # IN: number of shares to operate with, asset, position ID
  # OUT: True (confirmed) / False (not confirmed)
# check POSITION: see if the position exists
  # IN: position ID
  # OUT: True (still exists) / False (does not exist)
  # IF FALSE, go back to submit order

# wait 15 mins
# back to beginning
