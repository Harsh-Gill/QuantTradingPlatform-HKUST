# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 10:12:21 2020

@author: hongsong chou
"""

import threading
import os
import time
from common.SingleStockExecution import SingleStockExecution


class ExchangeSimulator:
    futures = ['QLF1', 'GLF1', 'NYF1', 'HSF1', 'HCF1', 'NEF1', 'DBF1', 'IPF1', 'NLF1', 'RLF1']
    stocks = ['2610', '0050', '6443', '2498', '2618', '3374', '3035', '5347', '3264', '2392']
    subscriptionList = stocks + futures  # tickers (stock/future codes)

    # To store marketdata orderbook
    bidQuoteSnapshot = {}
    askQuoteSnapshot = {}
    for code in subscriptionList:
        bidQuoteSnapshot[code] = None
        askQuoteSnapshot[code] = None

    MO_remain_BUY = {}  # key:code, value: [orderid + restvolume]
    LO_remain_BUY = {}  # key:code, value: [orderid + restvolume, price]

    MO_remain_SELL = {}  # key:code, value: [orderid + restvolume]
    LO_remain_SELL = {}  # key:code, value: [orderid + restvolume, price]

    # To generate executionIDs, start from 1
    execID = 1

    m_QuoteSnapshot = threading.Lock();

    # m_MO_remain = threading.Lock();
    # m_LO_remain = threading.Lock();

    def __init__(self, marketData_2_exchSim_q, platform_2_exchSim_order_q, exchSim_2_platform_execution_q):
        print("[%d]<<<<< call ExchSim.init" % (os.getpid(),))

        t_md = threading.Thread(name='exchsim.on_md', target=self.consume_md, args=(marketData_2_exchSim_q,))
        t_md.start()

        t_order = threading.Thread(name='exchsim.on_order', target=self.consume_order,
                                   args=(platform_2_exchSim_order_q, exchSim_2_platform_execution_q,))
        t_order.start()

    def consume_md(self, marketData_2_exchSim_q):
        while True:
            res = marketData_2_exchSim_q.get()
            # print('[%d]ExchSim.consume_md' % (os.getpid()))
            # print(res.outputAsDataFrame())
            m_QuoteSnapshot.acquire()
            bidQuoteSnapshot[res.ticker] = [(res.bidPrice1, res.bidSize1),
                                            (res.bidPrice2, res.bidSize2),
                                            (res.bidPrice3, res.bidSize3),
                                            (res.bidPrice4, res.bidSize4),
                                            (res.bidPrice5, res.bidSize5)]
            askQuoteSnapshot[res.ticker] = [(res.askPrice1, res.askSize1),
                                            (res.askPrice2, res.askSize2),
                                            (res.askPrice3, res.askSize3),
                                            (res.askPrice4, res.askSize4),
                                            (res.askPrice5, res.askSize5)]

            ask_quote_all_level_size = [tup[1] for tup in askQuoteSnapshot[res.ticker]]
            bid_quote_all_level_size = [tup[1] for tup in bidQuoteSnapshot[res.ticker]]

            # The remained market orders
            # Extreme case: One market order consume all the quote size
            # Then the order with the rest size will go to the bottom
            # so no need for new design
            while len(MO_remain_BUY[res.ticker]) > 0 and sum(ask_quote_all_level_size) > 0:
                remain_order = MO_remain_BUY[res.ticker].pop(0)
                if res.currStatus == 'NEW':
                    self.process_order(remain_order, exchSim_2_platform_execution_q)

            while len(MO_remain_SELL[res.ticker]) > 0 and sum(bid_quote_all_level_size) > 0:
                remain_order = MO_remain_SELL[res.ticker].pop(0)
                if res.currStatus == 'NEW':
                    self.process_order(remain_order, exchSim_2_platform_execution_q)

            # The remained buy limited orders
            LO_BUY_index = 0
            LO_BUY_pop_index = []
            while LO_BUY_index < len(LO_remain_BUY[res.ticker]) and sum(ask_quote_all_level_size) > 0:
                res = LO_remain_BUY[res.ticker][LO_BUY_index]
                for level in range(1, 5 + 1):
                    if res.price >= askQuoteSnapshot[res.ticker][level - 1][0]:
                        if res.size > askQuoteSnapshot[res.ticker][level - 1][1]:
                            res.size -= askQuoteSnapshot[res.ticker][level - 1][1]
                            self.produce_execution(exchSim_2_platform_execution_q,
                                                   ticker=res.ticker,
                                                   direction="BI",
                                                   execID=self.execID,
                                                   orderID=res.orderID,
                                                   size=askQuoteSnapshot[res.ticker][level - 1][1],
                                                   price=askQuoteSnapshot[res.ticker][level - 1][0],
                                                   order_status="Partially Filled",
                                                   rest_size=res.size)
                            self.execID += 1
                            askQuoteSnapshot[res.ticker][level - 1] = 0
                            level += 1
                        else:
                            askQuoteSnapshot[res.ticker][level - 1][1] -= res.size
                            res.size = 0
                            self.produce_execution(exchSim_2_platform_execution_q,
                                                   ticker=res.ticker,
                                                   direction="BI",
                                                   execID=self.execID,
                                                   orderID=res.orderID,
                                                   size=res.size,
                                                   price=askQuoteSnapshot[res.ticker][level - 1][0],
                                                   order_status="Filled",
                                                   rest_size=0)  # size = res.size
                if res.size == 0:
                    LO_BUY_pop_index.append(LO_BUY_index)
                LO_BUY_index += 1

            LO_BUY_pop_index.sort(reverse=True)
            for pop_index in LO_BUY_pop_index:
                LO_remain_BUY.pop(pop_index)

            # The remained sell limited orders
            LO_SELL_index = 0
            LO_SELL_pop_index = []
            while LO_SELL_index < len(LO_remain_SELL[res.ticker]) and sum(bid_quote_all_level_size) > 0:
                res = LO_remain_SELL[res.ticker][LO_SELL_index]
                for level in range(1, 5 + 1):
                    if res.price <= bidQuoteSnapshot[res.ticker][level - 1][0]:
                        if res.size > bidQuoteSnapshot[res.ticker][level - 1][1]:
                            bidQuoteSnapshot[res.ticker][level - 1] = 0
                            res.size -= bidQuoteSnapshot[res.ticker][level - 1][1]
                            self.produce_execution(exchSim_2_platform_execution_q,
                                                   ticker=res.ticker,
                                                   direction="BI",
                                                   execID=self.execID,
                                                   orderID=res.orderID,
                                                   size=bidQuotaSnapshot[res.ticker][level - 1][1],
                                                   price=bidQuoteSnapshot[res.ticker][level - 1][0],
                                                   order_status="Partially Filled",
                                                   rest_size=res.size)  # size = askQuotaSnapshot[res.ticker][level-1][1], price = askQuotaSnapshot[res.ticker][level-1][0]
                            level += 1
                        else:
                            bidQuoteSnapshot[res.ticker][level - 1][1] -= res.size
                            res.size = 0
                            self.produce_execution(exchSim_2_platform_execution_q,
                                                   ticker=res.ticker,
                                                   direction="BI",
                                                   execID=self.execID,
                                                   orderID=res.orderID,
                                                   size=res.size,
                                                   price=bidQuoteSnapshot[res.ticker][level - 1][0],
                                                   order_status="Filled",
                                                   rest_size=0)
                if res.size == 0:
                    LO_SELL_pop_index.append(LO_SELL_index)
                LO_SELL_index += 1

            LO_SELL_pop_index.sort(reverse=True)
            for pop_index in LO_SELL_pop_index:
                LO_remain_SELL.pop(pop_index)

            m_QuotaSnapshot.release()

        #### exectionRestOrder

    def consume_order(self, platform_2_exchSim_order_q, exchSim_2_platform_execution_q):
        while True:
            res = platform_2_exchSim_order_q.get()  # res is an order
            # print('[%d]ExchSim.on_order' % (os.getpid()))
            # print(res.outputAsArray())
            m_QuoteSnapshot.acquire()
            self.process_order(res, exchSim_2_platform_execution_q)
            m_QuoteSnapshot.release()

    def process_order(self, order, exchSim_2_platform_execution_q):
        res = order
        if res.currStatus == 'NEW':
            if res.type == 'MO':  # market order
                if res.direction == 'BUY':
                    level = 1
                    while res.size > 0 and level <= 5:
                        if res.size > askQuoteSnapshot[res.ticker][level - 1][1]:
                            res.size -= askQuoteSnapshot[res.ticker][level - 1][1]
                            self.produce_execution(exchSim_2_platform_execution_q,
                                                   ticker=res.ticker,
                                                   direction="BI",
                                                   execID=self.execID,
                                                   orderID=res.orderID,
                                                   size=askQuoteSnapshot[res.ticker][level - 1][1],
                                                   price=askQuoteSnapshot[res.ticker][level - 1][0],
                                                   order_status="Partially Filled",
                                                   rest_size=res.size)
                            self.execID += 1
                            askQuoteSnapshot[res.ticker][level - 1] = 0
                            level += 1
                        else:
                            askQuoteSnapshot[res.ticker][level - 1][1] -= res.size
                            res.size = 0
                            self.produce_execution(exchSim_2_platform_execution_q,
                                                   ticker=res.ticker,
                                                   direction="BI",
                                                   execID=self.execID,
                                                   orderID=res.orderID,
                                                   size=res.size,
                                                   price=askQuoteSnapshot[res.ticker][level - 1][0],
                                                   order_status="Filled",
                                                   rest_size=0)  # size = res.size
                    if res.size > 0:
                        if res.ticker in MO_remain_BUY:
                            MO_remain_BUY[res.ticker].append((res.orderID, res.size))
                        else:
                            MO_remain_BUY[res.ticker] = [(res.orderID, res.size)]
                elif res.direction == 'SELL':
                    level = 1
                    while res.size > 0 and level <= 5:
                        if res.size > bidQuoteSnapshot[res.ticker][level - 1][1]:
                            res.size -= bidQuoteSnapshot[res.ticker][level - 1][1]
                            self.produce_execution(exchSim_2_platform_execution_q,
                                                   ticker=res.ticker,
                                                   direction="SI",
                                                   execID=self.execID,
                                                   orderID=res.orderID,
                                                   size=bidQuoteSnapshot[res.ticker][level - 1][1],
                                                   price=bidQuoteSnapshot[res.ticker][level - 1][0],
                                                   order_status="Partially Filled",
                                                   rest_size=res.size)
                            self.execID += 1
                            bidQuoteSnapshot[res.ticker][level - 1] = 0
                            level += 1
                        else:
                            bidQuoteSnapshot[res.ticker][level - 1][1] -= res.size
                            res.size = 0
                            self.produce_execution(exchSim_2_platform_execution_q,
                                                   ticker=res.ticker,
                                                   direction="SI",
                                                   execID=self.execID,
                                                   orderID=res.orderID,
                                                   size=res.size,
                                                   price=bidQuoteSnapshot[res.ticker][level - 1][0],
                                                   order_status="Filled",
                                                   rest_size=0)  # size = res.size
                    if res.size > 0:
                        if res.ticker in MO_remain_SELL:
                            MO_remain_SELL[res.ticker].append((res.orderID, res.size))
                        else:
                            MO_remain_SELL[res.ticker] = [(res.orderID, res.size)]

            elif res.type == 'LO':
                if res.direction == 'BUY':
                    level = 1
                    while res.size > 0 and level <= 5 and askQuoteSnapshot[res.ticker][level - 1][0] <= res.price:
                        if res.size > askQuoteSnapshot[res.ticker][level - 1][1]:
                            askQuoteSnapshot[res.ticker][level - 1] = 0
                            res.size -= askQuoteSnapshot[res.ticker][level - 1][1]
                            self.produce_execution(exchSim_2_platform_execution_q,
                                                   ticker=res.ticker,
                                                   direction="BI",
                                                   execID=self.execID,
                                                   orderID=res.orderID,
                                                   size=askQuotaSnapshot[res.ticker][level - 1][1],
                                                   price=askQuoteSnapshot[res.ticker][level - 1][0],
                                                   order_status="Partially Filled",
                                                   rest_size=res.size)  # size = askQuotaSnapshot[res.ticker][level-1][1], price = askQuotaSnapshot[res.ticker][level-1][0]
                            level += 1
                        else:
                            askQuoteSnapshot[res.ticker][level - 1][1] -= res.size
                            res.size = 0
                            self.produce_execution(exchSim_2_platform_execution_q,
                                                   ticker=res.ticker,
                                                   direction="BI",
                                                   execID=self.execID,
                                                   orderID=res.orderID,
                                                   size=res.size,
                                                   price=askQuoteSnapshot[res.ticker][level - 1][0],
                                                   order_status="Filled",
                                                   rest_size=0)  # size = res.size, price = askQuotaSnapshot[res.ticker][level-1][0]
                    if res.size > 0:
                        if res.ticker in LO_remain_BUY:
                            LO_remain_BUY[res.ticker].append((res.orderID, res.size))
                        else:
                            LO_remain_BUY[res.ticker] = [(res.orderID, res.size)]
                elif res.direction == 'SELL':
                    level = 1
                    while res.size > 0 and level <= 5 and bidQuoteSnapshot[res.ticker][level - 1][0] >= res.price:
                        if res.size > bidQuoteSnapshot[res.ticker][level - 1][1]:
                            bidQuoteSnapshot[res.ticker][level - 1] = 0
                            res.size -= bidQuoteSnapshot[res.ticker][level - 1][1]
                            self.produce_execution(exchSim_2_platform_execution_q,
                                                   ticker=res.ticker,
                                                   direction="BI",
                                                   execID=self.execID,
                                                   orderID=res.orderID,
                                                   size=bidQuotaSnapshot[res.ticker][level - 1][1],
                                                   price=bidQuoteSnapshot[res.ticker][level - 1][0],
                                                   order_status="Partially Filled",
                                                   rest_size=res.size)  # size = askQuotaSnapshot[res.ticker][level-1][1], price = askQuotaSnapshot[res.ticker][level-1][0]
                            level += 1
                        else:
                            bidQuoteSnapshot[res.ticker][level - 1][1] -= res.size
                            res.size = 0
                            self.produce_execution(exchSim_2_platform_execution_q,
                                                   ticker=res.ticker,
                                                   direction="BI",
                                                   execID=self.execID,
                                                   size=res.size,
                                                   orderID=res.orderID,
                                                   price=bidQuoteSnapshot[res.ticker][level - 1][0],
                                                   order_status="Filled",
                                                   rest_size=0)  # size = res.size, price = askQuotaSnapshot[res.ticker][level-1][0]
                    if res.size > 0:
                        if res.ticker in LO_remain_SELL:
                            LO_remain_SELL[res.ticker].append((res.orderID, res.size))
                        else:
                            LO_remain_SELL[res.ticker] = [(res.orderID, res.size)]
            else:
                pass
        # elif res.currStatus == 'Cancel':
        #    pass
        else:
            pass

    def produce_execution(self,
                          exchSim_2_platform_execution_q,
                          ticker,
                          direction,
                          execID,
                          orderID,
                          size=0,  # executed size
                          price=None,
                          order_status=None,
                          rest_size=0):  # rest size in the order

        execution = SingleStockExecution(ticker,
                                         '2024-07-30',
                                         time.asctime(time.localtime(time.time())),
                                         direction,
                                         execID,
                                         orderID,
                                         size,
                                         price,
                                         order_status,
                                         rest_size)

        # Only Traded Successfully the execution will be delivered
        self.execID += 1
        exchSim_2_platform_execution_q.put(execution)
        print('[%d]ExchSim.produce_execution' % (os.getpid()))
        print(execution.outputAsArray())
