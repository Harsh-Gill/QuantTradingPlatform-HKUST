# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 10:12:21 2020

@author: hongsong chou
"""

import numpy as np
import threading
import os
import time
from datetime import datetime
from common.SingleStockExecution import SingleStockExecution


class ExchangeSimulator:
    futures = ['QLF1', 'GLF1', 'NYF1', 'HSF1', 'HCF1', 'NEF1', 'DBF1', 'IPF1', 'NLF1', 'RLF1']
    stocks = ['2610', '0050', '6443', '2498', '2618', '3374', '3035', '5347', '3264', '2392']
    subscriptionList = stocks + futures  # tickers (stock/future codes)

    # m_MO_remain = threading.Lock();
    # m_LO_remain = threading.Lock();

    def __init__(self, marketData_2_exchSim_q, platform_2_exchSim_order_q, exchSim_2_platform_execution_q):
        print("[%d]<<<<< call ExchSim.init" % (os.getpid(),))

        # To store marketdata orderbook
        self.bidQuoteSnapshot = {}
        self.askQuoteSnapshot = {}

        # Time Adjustment
        self.time_delta = None

        self.MO_remain_BUY = {}  # key:code, value: [orderid + restvolume]
        self.LO_remain_BUY = {}  # key:code, value: [orderid + restvolume, price]

        self.MO_remain_SELL = {}  # key:code, value: [orderid + restvolume]
        self.LO_remain_SELL = {}  # key:code, value: [orderid + restvolume, price]

        for code in self.subscriptionList:
            self.bidQuoteSnapshot[code] = None
            self.askQuoteSnapshot[code] = None
            self.MO_remain_BUY[code] = []
            self.LO_remain_BUY[code] = []
            self.MO_remain_SELL[code] = []
            self.LO_remain_SELL[code] = []

        # To generate executionIDs, start from 1
        self.execID = 1

        self.m_QuoteSnapshot = threading.Lock()

        t_md = threading.Thread(name='exchsim.on_md', target=self.consume_md,
                                args=(marketData_2_exchSim_q, exchSim_2_platform_execution_q,))
        t_md.start()

        t_order = threading.Thread(name='exchsim.on_order', target=self.consume_order,
                                   args=(platform_2_exchSim_order_q, exchSim_2_platform_execution_q,))
        t_order.start()

    def consume_md(self, marketData_2_exchSim_q, exchSim_2_platform_execution_q):
        while True:
            res = marketData_2_exchSim_q.get()
            if res is None:
                continue
            else:
                if self.time_delta is None:
                    self.time_delta = datetime.now() - res.timeStamp
            # print('[%d]ExchSim.consume_md' % (os.getpid()))
            # print(res.outputAsDataFrame())
            self.m_QuoteSnapshot.acquire()  # Lock
            self.bidQuoteSnapshot[res.ticker] = np.array([[res.bidPrice1, res.bidSize1],
                                                          [res.bidPrice2, res.bidSize2],
                                                          [res.bidPrice3, res.bidSize3],
                                                          [res.bidPrice4, res.bidSize4],
                                                          [res.bidPrice5, res.bidSize5]])
            self.askQuoteSnapshot[res.ticker] = np.array([[res.askPrice1, res.askSize1],
                                                          [res.askPrice2, res.askSize2],
                                                          [res.askPrice3, res.askSize3],
                                                          [res.askPrice4, res.askSize4],
                                                          [res.askPrice5, res.askSize5]])
            # Prevent "None" as parameters
            if self.askQuoteSnapshot is None or self.bidQuoteSnapshot is None:
                continue

            if self.askQuoteSnapshot[res.ticker] is None or self.bidQuoteSnapshot[res.ticker] is None:
                continue

            # Deal with the remained market orders
            index_market_buy = 0
            poplist_market_buy = []
            while index_market_buy < len(self.MO_remain_BUY[res.ticker]) \
                    and np.sum(self.askQuoteSnapshot[res.ticker][:, 1]) > 0:
                orderID, size = self.MO_remain_BUY[res.ticker][index_market_buy]
                size = self.buy_market_order(res.ticker, orderID, size, exchSim_2_platform_execution_q)
                if size > 0:  # sizes of all ask levels are cleared
                    self.MO_remain_BUY[res.ticker][index_market_buy] = (orderID, size)
                    break
                elif size == 0:  # The market order's size is cleared
                    poplist_market_buy.append(index_market_buy)
                index_market_buy += 1
            poplist_market_buy.sort(reverse=True)  # Avoid the indexes changes during popping
            for pop_index in poplist_market_buy:
                self.MO_remain_BUY.pop(pop_index)

            index_market_sell = 0
            poplist_market_sell = []
            while index_market_sell < len(self.MO_remain_SELL[res.ticker]) \
                    and np.sum(self.bidQuoteSnapshot[res.ticker][:, 1]) > 0:
                orderID, size = self.MO_remain_SELL[res.ticker][index_market_sell]
                size = self.sell_market_order(res.ticker, orderID, size, exchSim_2_platform_execution_q)
                if size > 0:  # sizes of all ask levels are cleared
                    self.MO_remain_SELL[res.ticker][index_market_sell] = (orderID, size)
                    break
                elif size == 0:  # The market order's size is cleared
                    poplist_market_sell.append(index_market_sell)
                index_market_sell += 1
            poplist_market_sell.sort(reverse=True)  # Avoid the indexes changes during popping
            for pop_index in poplist_market_sell:
                self.MO_remain_SELL.pop(pop_index)

            # Deal with the remained limit orders
            index_limit_buy = 0
            poplist_limit_buy = []
            while index_limit_buy < len(self.LO_remain_BUY[res.ticker]) \
                    and np.sum(self.askQuoteSnapshot[res.ticker][:, 1]) > 0:
                orderID, size, price = self.LO_remain_BUY[res.ticker][index_limit_buy]
                size, price = self.buy_limit_order(res.ticker, orderID, size, price, exchSim_2_platform_execution_q)
                if size > 0:
                    self.LO_remain_BUY[res.ticker][index_limit_buy] = (orderID, size, price)
                elif size == 0:
                    poplist_limit_buy.append(index_limit_buy)
                index_limit_buy += 1
            poplist_limit_buy.sort(reverse=True)  # Avoid the indexes changes during popping
            for pop_index in poplist_limit_buy:
                self.LO_remain_BUY.pop(pop_index)

            index_limit_sell = 0
            poplist_limit_sell = []
            while index_limit_sell < len(self.LO_remain_SELL[res.ticker]) \
                    and np.sum(self.askQuoteSnapshot[res.ticker][:, 1]) > 0:
                orderID, size, price = self.LO_remain_SELL[res.ticker][index_limit_sell]
                size, price = self.sell_limit_order(res.ticker, orderID, size, price, exchSim_2_platform_execution_q)
                if size > 0:
                    self.LO_remain_SELL[res.ticker][index_limit_sell] = (orderID, size, price)
                elif size == 0:
                    poplist_limit_sell.append(index_limit_sell)
                index_limit_sell += 1
            poplist_limit_sell.sort(reverse=True)  # Avoid the indexes changes during popping
            for pop_index in poplist_limit_sell:
                self.LO_remain_SELL.pop(pop_index)

            # print("processed for comsume_md\n")
            self.m_QuoteSnapshot.release()

        #### exectionRestOrder

    def consume_order(self, platform_2_exchSim_order_q, exchSim_2_platform_execution_q):
        while True:
            res = platform_2_exchSim_order_q.get()  # res is an order
            if self.askQuoteSnapshot[res.ticker] is None or self.bidQuoteSnapshot[res.ticker] is None:
                continue
            print('[%d]ExchSim.on_order' % (os.getpid()))
            #print(res.outputAsArray())
            self.m_QuoteSnapshot.acquire()
            self.process_new_order(res, exchSim_2_platform_execution_q)
            #print("processed for comsume_order\n")
            #print("execID is ", self.execID)
            self.m_QuoteSnapshot.release()

    def buy_market_order(self, ticker, orderID, size, exchSim_2_platform_execution_q):
        level = 1
        while size > 0 and level <= 5:
            if 0 == self.askQuoteSnapshot[ticker][level - 1][1]:
                level += 1
                continue
            elif 0 < self.askQuoteSnapshot[ticker][level - 1][1] < size:
                size -= self.askQuoteSnapshot[ticker][level - 1][1]
                self.produce_execution(exchSim_2_platform_execution_q,
                                       ticker=ticker,
                                       direction="BI",
                                       execID=self.execID,
                                       orderID=orderID,
                                       size=int(self.askQuoteSnapshot[ticker][level - 1][1]),
                                       price=self.askQuoteSnapshot[ticker][level - 1][0],
                                       order_status="Partially Filled",
                                       rest_size=size)
                self.askQuoteSnapshot[ticker][level - 1] = 0
                level += 1
            elif 0 < size <= self.askQuoteSnapshot[ticker][level - 1][1]:
                self.askQuoteSnapshot[ticker][level - 1][1] -= size
                self.produce_execution(exchSim_2_platform_execution_q,
                                       ticker=ticker,
                                       direction="BI",
                                       execID=self.execID,
                                       orderID=orderID,
                                       size=size,
                                       price=self.askQuoteSnapshot[ticker][level - 1][0],
                                       order_status="Filled",
                                       rest_size=0)  # size = res.size
                size = 0
                break

        print("BUY Market Order :", ticker, orderID, size)
        print("execID is ", self.execID)
        return size

    def sell_market_order(self, ticker, orderID, size, exchSim_2_platform_execution_q):
        level = 1
        while size > 0 and level <= 5:
            if 0 == self.bidQuoteSnapshot[ticker][level - 1][1]:
                level += 1
                continue
            elif 0 < self.bidQuoteSnapshot[ticker][level - 1][1] < size:
                size -= self.bidQuoteSnapshot[ticker][level - 1][1]
                self.produce_execution(exchSim_2_platform_execution_q,
                                       ticker=ticker,
                                       direction="SI",
                                       execID=self.execID,
                                       orderID=orderID,
                                       size=self.bidQuoteSnapshot[ticker][level - 1][1],
                                       price=self.bidQuoteSnapshot[ticker][level - 1][0],
                                       order_status="Partially Filled",
                                       rest_size=size)
                self.bidQuoteSnapshot[ticker][level - 1] = 0
                level += 1
            elif 0 < size <= self.bidQuoteSnapshot[ticker][level - 1][1]:
                self.bidQuoteSnapshot[ticker][level - 1][1] -= size
                self.produce_execution(exchSim_2_platform_execution_q,
                                       ticker=ticker,
                                       direction="SI",
                                       execID=self.execID,
                                       orderID=orderID,
                                       size=size,
                                       price=self.bidQuoteSnapshot[ticker][level - 1][0],
                                       order_status="Filled",
                                       rest_size=0)  # size = res.size
                size = 0
                break

        print("MOS :", ticker, orderID, size)
        print("execID is ", self.execID)
        return size

    def buy_limit_order(self, ticker, orderID, size, price, exchSim_2_platform_execution_q):
        level = 1
        while size > 0 and level <= 5 and self.askQuoteSnapshot[ticker][level - 1][0] <= price:
            if 0 == self.askQuoteSnapshot[ticker][level - 1][1]:
                level += 1
                continue
            elif 0 < self.askQuoteSnapshot[ticker][level - 1][1] < size:
                self.askQuoteSnapshot[ticker][level - 1] = 0
                size -= self.askQuoteSnapshot[ticker][level - 1][1]
                self.produce_execution(exchSim_2_platform_execution_q,
                                       ticker=ticker,
                                       direction="BI",
                                       execID=self.execID,
                                       orderID=orderID,
                                       size=self.askQuoteSnapshot[ticker][level - 1][1],
                                       price=self.askQuoteSnapshot[ticker][level - 1][0],
                                       order_status="Partially Filled",
                                       rest_size=size)
                level += 1
            elif 0 < size <= self.askQuoteSnapshot[ticker][level - 1][1]:
                self.askQuoteSnapshot[ticker][level - 1][1] -= size
                self.produce_execution(exchSim_2_platform_execution_q,
                                       ticker=ticker,
                                       direction="BI",
                                       execID=self.execID,
                                       orderID=orderID,
                                       size=size,
                                       price=self.askQuoteSnapshot[ticker][level - 1][0],
                                       order_status="Filled",
                                       rest_size=0)
                size = 0
                break

        print("LOB :", ticker, orderID, size, price)
        print("execID is ", self.execID)
        return size, price

    def sell_limit_order(self, ticker, orderID, size, price, exchSim_2_platform_execution_q):
        level = 1
        while size > 0 and level <= 5 and self.bidQuoteSnapshot[ticker][level - 1][0] >= price:
            if 0 == self.bidQuoteSnapshot[ticker][level - 1][1]:
                level += 1
                continue
            elif 0 < self.bidQuoteSnapshot[ticker][level - 1][1] < size:
                self.bidQuoteSnapshot[ticker][level - 1] = 0
                size -= self.bidQuoteSnapshot[ticker][level - 1][1]
                self.produce_execution(exchSim_2_platform_execution_q,
                                       ticker=ticker,
                                       direction="SI",
                                       execID=self.execID,
                                       orderID=orderID,
                                       size=self.bidQuoteSnapshot[ticker][level - 1][1],
                                       price=self.bidQuoteSnapshot[ticker][level - 1][0],
                                       order_status="Partially Filled",
                                       rest_size=size)
                level += 1
            elif 0 < size <= self.bidQuoteSnapshot[ticker][level - 1][1]:
                self.bidQuoteSnapshot[ticker][level - 1][1] -= size
                self.produce_execution(exchSim_2_platform_execution_q,
                                       ticker=ticker,
                                       direction="SI",
                                       execID=self.execID,
                                       size=size,
                                       orderID=orderID,
                                       price=self.bidQuoteSnapshot[ticker][level - 1][0],
                                       order_status="Filled",
                                       rest_size=0)
                size = 0
                break

        print("LOS :", ticker, orderID, size, price)
        print("execID is ", self.execID)
        return size, price

    def process_new_order(self, order, exchSim_2_platform_execution_q):
        ticker = order.ticker
        orderID = order.orderID
        currStatus = order.currStatus.upper()
        type = order.type.upper()
        direction = order.direction.upper()
        size = order.size
        price = order.price

        print("Viewing order:", ticker, orderID, currStatus, type, direction, size, price)
        # print(self.askQuoteSnapshot[ticker], self.bidQuoteSnapshot[ticker])

        # Prevent "None" as parameters
        if self.askQuoteSnapshot[ticker] is None or self.bidQuoteSnapshot[ticker] is None:
            return

        if currStatus == 'NEW':
            # print("Identified as New order.")
            if type == 'MO':  # market order
                print("Identified as New MO order.")
                if direction == 'BUY':
                    print("Identified as New MO BUY order.")
                    size = self.buy_market_order(ticker, orderID, size, exchSim_2_platform_execution_q)
                    if size > 0:
                        if ticker in self.MO_remain_BUY:
                            self.MO_remain_BUY[ticker].append((orderID, size))
                        else:
                            self.MO_remain_BUY[ticker] = [(orderID, size)]
                elif direction == 'SELL':
                    print("Identified as New MO SELL order.")
                    size = self.sell_market_order(ticker, orderID, size, exchSim_2_platform_execution_q)
                    if size > 0:
                        if ticker in self.MO_remain_SELL:
                            self.MO_remain_SELL[ticker].append((orderID, size))
                        else:
                            self.MO_remain_SELL[ticker] = [(orderID, size)]

            elif type == 'LO':
                print("Identified as New LO order.")
                if direction == 'BUY':
                    print("Identified as New LO BUY order.")
                    size, price = self.buy_limit_order(ticker, orderID, size, price, exchSim_2_platform_execution_q)
                    if size > 0:
                        if ticker in self.LO_remain_BUY:
                            self.LO_remain_BUY[ticker].append((orderID, size, price))
                        else:
                            self.LO_remain_BUY[ticker] = [(orderID, size, price)]
                elif direction == 'SELL':
                    print("Identified as New LO SELL order.")
                    size, price = self.sell_limit_order(ticker, orderID, size, price, exchSim_2_platform_execution_q)
                    if size > 0:
                        if ticker in self.LO_remain_SELL:
                            self.LO_remain_SELL[ticker].append((orderID, size, price))
                        else:
                            self.LO_remain_SELL[ticker] = [(orderID, size, price)]
            else:
                pass
        # elif currStatus == 'Cancel':
        #    pass
        else:
            pass
        print("processed for process_new_order\n")

    def produce_execution(self,
                          exchSim_2_platform_execution_q,
                          ticker,
                          direction,
                          execID,
                          orderID,
                          size=0,  # executed size
                          price=0,
                          order_status=None,
                          rest_size=0):  # rest size in the order

        current_datetime = (datetime.now() - self.time_delta).strftime("%Y-%m-%d %H:%M:%S")
        date = current_datetime.split(" ")[0]
        submissionTime = current_datetime.split(" ")[1]

        execution = SingleStockExecution(ticker,
                                         date,
                                         submissionTime,
                                         direction,
                                         execID,
                                         orderID,
                                         int(size),
                                         float(price),
                                         order_status,
                                         rest_size)

        # Only Traded Successfully the execution will be delivered
        self.execID += 1
        exchSim_2_platform_execution_q.put(execution)
        print('[%d]ExchSim.produce_execution' % (os.getpid()))
        print("execID is ", self.execID)
        if order_status == "Filled":
            print(f"Order {orderID} fully executed at price {price}, size {size}")
        elif order_status == "Partially Filled":
            print(f"Order {orderID} partially executed at price {price}, size {size}, rest size {rest_size}")
        # print(execution.outputAsArray())


        # write to logs txt file
        with open('execution_log.txt', 'a') as f:
            f.write(f"{execution.outputAsArray()}\n")
