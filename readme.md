# A Python Based 4 Component Simulation of a Quant Platform, Market Data Service, Exchange and Quant Strategies

# FRONTEND DEMO

https://github.com/user-attachments/assets/d629bab6-6e52-467b-b6c3-cc841478caeb

To run frontend:

```
pip3 install flask
python3 Prototype/app.py
```

To run simulation:

```
pip3 install tensorflow # to cater for DL models
python3 Prototype/systemController
```

# Market Data Service - Group 1

- DONE PART -> This section is working for SINGLE STOCK PUBLISHING , current code can take in files with 2 specified options

  - Binary data file processing (DONE ONLY FOR STOCKS DATA)
  - Subscription-based market data (re-)publishing (DONE ONLY FOR STOCKS DATA)
  - Multi-processing for real-time data dissemination (DONE ONLY FOR STOCKS DATA)

  market_data_generator loads data from csv.gz files into data generator then feeds it into get_market_data_with_delay to submit to queues

  get_market_data_with_delay(
  stock_id = None, # IF specified, it will only load that stocks data
  start_on_date="202404" # IF specified, it will only load data from that files date
  )

- Updates needed from Group 1 ->
  How can we write multiple stocks (easy solution is to use single Queue and write as a list of data but up to you guys, many ways for this)
  How to write Futures Quotes and Prices

# Exchange Simulator - Group 2

- DONE PART -> This section is working for simple buy MO
  The exchange is able to take in a buy MO order , use the latest orderbook to create executions and fill order. Then update orderbook and send back execution and updated order object

  - Establishment of an order book for an individual stock (DONE)
  - Updating order book status upon real-time market data (DONE)
  - Handling submitted orders (DONE only for simple MO)
  - Responding to order submitters with executions; (DONE)

- Updates needed from Group 2

  Handle more types of orders, LO , MO , Futures etc. The flow should be simple to understand.

# Quant Platform - Group 3

- DONE PART -> This section is can take in orders from strategy, submit to exchange and get back updated information to feed into strategy again

  - Receiving market data and passing to subscribed strategies (DONE but no flow to add multiple strategies yet , need to define)
  - Receiving order submissions (if any) from strategies (DONE)
  - Utilities for sending a single stock and/or futures order to the exchange simulator and receiving info (DONE for stock orders)
  - Passing executions or cancellation info to strategies (DONE for stock orders)

- Updates needed from Group 3
  Add logic to handle multiple subscriptions of strategies, and add more complex thinkings

# Strategy - Group 4

- DONE PART -> This section is working for simple dumb strategy
  A simple buy MO is submitted via Trading Platform on every single call

  - A single stock strategy, or a single stock futures strategy, or an arbitrage strategy can be designed using the other as forecasting signals;
  - For whichever strategy, the project will calculate trading signals based on real-time market data;
  - If deciding to trade, submit an order and receive executions (DONE)
  - Calculating and reporting strategy PnL and other metrics in real time and after market close

- Updates needed from Group 4 ->

  Add PNL metrics , use actual strategy to create signal
