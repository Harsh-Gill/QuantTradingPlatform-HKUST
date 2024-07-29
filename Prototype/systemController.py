# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 10:26:05 2020

@author: hongsong chou
"""

import os
from multiprocessing import Process, Queue

from exchangeSimulator import ExchangeSimulator
from marketDataService import MarketDataService, SubscriptionManager
from quantTradingPlatform import TradingPlatform

# from datetime import datetime

if __name__ == '__main__':
    # delete existing .txt log files 
    for f in os.listdir():
        if f.endswith('.txt'):
            os.remove(f)

    ###########################################################################
    # Define instruments to be subscribed
    ###########################################################################
    futures = ['QLF1', 'GLF1', 'NYF1', 'HSF1', 'HCF1', 'NEF1', 'DBF1', 'IPF1', 'NLF1', 'RLF1']
    stocks = ['2610', '0050', '6443', '2498', '2618', '3374', '3035', '5347', '3264', '2392']

    ###########################################################################
    # Define all components
    ###########################################################################
    # Initialize market data subscription manager
    marketDataSubMgr = SubscriptionManager(futures=futures, stocks=stocks)

    platform_2_exchSim_order_q = Queue()
    exchSim_2_platform_execution_q = Queue()

    platform_2_strategy_md_q = Queue()
    strategy_2_platform_order_q = Queue()
    platform_2_strategy_execution_q = Queue()

    ###########################################################################
    # Start processes
    ###########################################################################
    Process(name='md', target=MarketDataService, args=(marketDataSubMgr,)).start()
    Process(name='sim', target=ExchangeSimulator,
            args=(marketDataSubMgr, platform_2_exchSim_order_q, exchSim_2_platform_execution_q,)).start()
    Process(name='platform', target=TradingPlatform,
            args=(marketDataSubMgr, platform_2_exchSim_order_q, exchSim_2_platform_execution_q,)).start()

''' !!!
# calculate simulation time diff
diff = 0

# call when exchange start to send orderbook (yyyy-mm-dd)
def calculate_diff(date)
    now = datetime.now()
    then = datetime(int(date[:4]), int(date[5:7]), int(date[8:10]), 9, 0, 0)
    diff = now - then

# return simulation time (call by other component)
def current_time():
    now = datetime.now()
    return (now + diff).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
# !!!
'''
