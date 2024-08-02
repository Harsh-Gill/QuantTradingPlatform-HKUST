# -*- coding: utf-8 -*-
"""
Version 1: Updated 29/07/2024 19:08

@author: HUANG Xiaofeng
       : WANG Yiqi
       : WU Zilin
       : Zhao Tianjun
"""

import threading
import os
from QuantStrategy import QuantStrategy, YFQuantStrategy,HarshQuantStrategy
from datetime import datetime
import time

class TradingPlatform:
    subscribeStockTicker = ['3035', '3374', '6443', '2392', '5347', '3264', '2618', '2498', '0050', '2610']
    subscribeFutureTicker = ['IPF1', 'QLF1', 'RLF1', 'GLF1', 'NLF1', 'NEF1', 'HSF1', 'HCF1', 'NYF1', 'DBF1']

    def __init__(self, marketData_2_platform_q, strategy_2_platform_order_q, platform_2_exchSim_order_q, \
                 exchSim_2_platform_execution_q):
        print("[%d]<<<<< call Platform.init" % (os.getpid(),))

        self.quantStrat = []
        self.order_id = 1
        self.time_delta = None
        self.mutex = threading.Lock()

        # Instantiate individual strategies
        self.quantStrat.append(YFQuantStrategy(1, "quantStrategy1", "Zhou Yufeng", \
                                             [('3035', 'IPF1'), ('3374', 'QLF1'), ('6443', 'RLF1'), ('2392', 'GLF1'), \
                                              ('5347', 'NLF1'), ('3264', 'NEF1'), ('2618', 'HSF1'), ('2498', 'HCF1'), \
                                              ('0050', 'NYF1'), ('2610', 'DBF1')], "2024-06-28", strategy_2_platform_order_q))
        
        
        # self.quantStrat.append(QuantStrategy(2, "quantStrategy2", "Zou Jie", \
        #                                      [('3035', 'IPF1'), ('3374', 'QLF1'), ('6443', 'RLF1'), ('2392', 'GLF1'), \
        #                                       ('5347', 'NLF1'), ('3264', 'NEF1'), ('2618', 'HSF1'), ('2498', 'HCF1'), \
        #                                       ('0050', 'NYF1'), ('2610', 'DBF1')], "2024-06-28", strategy_2_platform_order_q))
        self.quantStrat.append(HarshQuantStrategy(2, "quantStrategy2", "Zou Jie", \
                                             [('3035', 'IPF1'), ('3374', 'QLF1'), ('6443', 'RLF1'), ('2392', 'GLF1'), \
                                              ('5347', 'NLF1'), ('3264', 'NEF1'), ('2618', 'HSF1'), ('2498', 'HCF1'), \
                                              ('0050', 'NYF1'), ('2610', 'DBF1')], "2024-06-28", strategy_2_platform_order_q))
            
    
        self.quantStrat.append(QuantStrategy(3, "quantStrategy3", "Harsh",  \
                                             ['3035', '3374', '6443', '2392', '5347', '3264', '2618', '2498', '0050', '2610'], 
                                             "2024-06-28", 
                                             strategy_2_platform_order_q))

        t_md = threading.Thread(name='platform.on_marketData', target=self.consume_marketData, \
                                args=(marketData_2_platform_q,))
        t_md.start()

        t_exec = threading.Thread(name='platform.on_exec', target=self.handle_execution, \
                                  args=(exchSim_2_platform_execution_q,))
        t_exec.start()

        # overall order handling of all strategy (strategy 0)
        t_order = threading.Thread(name='platform.on_order', target=self.handle_order, \
                                   args=(platform_2_exchSim_order_q, strategy_2_platform_order_q))
        t_order.start()

        # store order status and execution by order id
        self.order_status_and_executions = {}

        # store order belonging to which strategy
        self.order_belonging = {}

        # store current position for overall platform
        self.position_and_cost = {}

        # Store unmatched orders, grouped by ticker symbol
        self.unmatched_orders = {}

        # store the last mid-quote
        self.mid_q_pair = [[None, None] for _ in range(len(TradingPlatform.subscribeStockTicker))]

        for i in range(len(self.quantStrat)):
            self.position_and_cost[self.quantStrat[i].getStratID()] = {}

            for stock_code in TradingPlatform.subscribeStockTicker:
                self.position_and_cost[self.quantStrat[i].getStratID()][stock_code] = [0, 0]

            for future_code in TradingPlatform.subscribeFutureTicker:
                self.position_and_cost[self.quantStrat[i].getStratID()][future_code] = [0, 0]

        # pnl calculation
        t_pnl = threading.Thread(name='platform.pnl', target=self.log_position_and_pnl)
        t_pnl.start()

    '''
    def cross_matching_and_order_submitting(self, orders_colleted):
        # Perform internal cross-matching and for the rest
        for ticker_price_key, order_lists in orders_colleted.items():
            # Continue matching until no more orders can be matched
            while len(order_lists[0]) > 0 and len(order_lists[1]) > 0:
                buy_order = order_lists[0][0]
                sell_order = order_lists[1][0]

                # Update total position and cost for each strategy
                stratID_buy = self.order_belonging[buy_order.orderID]
                stratID_sell = self.order_belonging[sell_order.orderID]

                # Match orders and calculate internal cross execution information
                quantity_to_match = min(buy_order.size, sell_order.size)
                execution_id = f"O_{buy_order.orderID}_{sell_order.orderID}"
                current_datetime = (datetime.now() - self.time_delta).strftime("%Y-%m-%d %H:%M:%S")

                with self.mutex:
                    if quantity_to_match == buy_order.size and quantity_to_match == sell_order.size:
                        self.order_status_and_executions[buy_order.orderID].append((execution_id, current_datetime.split(" ")[0], \
                                                                                    current_datetime.split(" ")[1], 'Filled', \
                                                                                    buy_order.price, quantity_to_match, \
                                                                                    buy_order.size, 0))
                        self.order_status_and_executions[sell_order.orderID].append((execution_id, current_datetime.split(" ")[0], \
                                                                                    current_datetime.split(" ")[1], 'Filled', \
                                                                                    sell_order.price, quantity_to_match, \
                                                                                    sell_order.size, 0))
                        order_lists[1].pop(0)
                        order_lists[0].pop(0)
                    elif quantity_to_match == buy_order.size:
                        self.order_status_and_executions[buy_order.orderID].append((execution_id, current_datetime.split(" ")[0], \
                                                                                    current_datetime.split(" ")[1], 'Filled', \
                                                                                    buy_order.price, quantity_to_match, \
                                                                                    buy_order.size, 0))
                        self.order_status_and_executions[sell_order.orderID].append((execution_id, current_datetime.split(" ")[0], \
                                                                                    current_datetime.split(" ")[1], 'PartiallyFilled', \
                                                                                    sell_order.price, quantity_to_match, \
                                                                                    quantity_to_match, sell_order.size - quantity_to_match))
                        order_lists[1][0].size -= quantity_to_match
                        order_lists[0].pop(0)
                    else:
                        self.order_status_and_executions[buy_order.orderID].append((execution_id, current_datetime.split(" ")[0], \
                                                                                    current_datetime.split(" ")[1], 'PartiallyFilled', \
                                                                                    buy_order.price, quantity_to_match, \
                                                                                    quantity_to_match, buy_order.size - quantity_to_match))
                        self.order_status_and_executions[sell_order.orderID].append((execution_id, current_datetime.split(" ")[0], \
                                                                                    current_datetime.split(" ")[1], 'Filled', \
                                                                                    sell_order.price, quantity_to_match, \
                                                                                    sell_order.size, 0))
                        order_lists[0][0].size -= quantity_to_match
                        order_lists[1].pop(0)

                    self.position_and_cost[stratID_buy][buy_order.ticker][0] += quantity_to_match
                    self.position_and_cost[stratID_buy][buy_order.ticker][1] += quantity_to_match * buy_order.price

                    self.position_and_cost[stratID_sell][sell_order.ticker][0] -= quantity_to_match
                    self.position_and_cost[stratID_sell][sell_order.ticker][1] -= quantity_to_match * sell_order.price

                    with open('execution_and_status.txt', 'a') as f2:
                        output = [buy_order.ticker, buy_order.direction, buy_order.orderID]
                        output.extend(list(self.order_status_and_executions[buy_order.orderID][-1]))
                        f2.write(str(output) + '\n')
                        output = [sell_order.ticker, sell_order.direction, sell_order.orderID]
                        output.extend(list(self.order_status_and_executions[sell_order.orderID][-1]))
                        f2.write(str(output) + '\n')

            while len(order_lists[0]) > 0:
                with self.mutex:
                    order = order_lists[0].pop(0)
                    self.platform_2_exchSim_order_q.put(order)
        
                    with open('submitted_orders.txt', 'a') as f1:
                        f1.write(str(order.outputAsArray()) + '\n')

            while len(order_lists[1]) > 0:
                with self.mutex:
                    order = order_lists[1].pop(0)
                    self.platform_2_exchSim_order_q.put(order)
        
                    with open('submitted_orders.txt', 'a') as f1:
                        f1.write(str(order.outputAsArray()) + '\n')
    '''

    # overall order handling
    def handle_order(self, platform_2_exchSim_order_q, strategy_2_platform_order_q):
        while True:
            #print('[%d]Platform.handle_order --' % (os.getpid()))
            order_message = strategy_2_platform_order_q.get()
            if order_message is None:
                pass
            else:
                stratID = order_message[0]
                #print("<><><><><><>< stratID: ", type(stratID))
                order = order_message[1]
                current_datetime = (datetime.now() - self.time_delta).strftime("%Y-%m-%d %H:%M:%S")
                order.orderID = self.order_id
                self.order_id += 1
                order.date = current_datetime.split(" ")[0]
                order.submissionTime = current_datetime.split(" ")[1]
                order.currStatusTime = current_datetime.split(" ")[1]
                order.currStatus = "NEW"

                # print(f"[{os.getpid()}]Platform.handle_order: {order.outputAsArray()}")

                with self.mutex:
                    self.order_status_and_executions[order.orderID] = [(order.ticker, order.date, order.submissionTime, \
                                                                        order.direction, order.price, order.size, order.type), \
                                                                       (None, order.date, order.currStatusTime, order.currStatus, None, None, 0, order.size)]
                    self.order_belonging[order.orderID] = stratID
                    with open('execution_and_status.txt', 'a') as f2:
                        output = [order.ticker, order.direction, order.orderID, None, order.date, order.currStatusTime, order.currStatus, None, None, 0, order.size]
                        f2.write(str(output) + '\n')
                
                platform_2_exchSim_order_q.put(order)

                with open('submitted_orders.txt', 'a') as f1:
                    f1.write(str(order.outputAsArray()) + '\n')

                '''
                # !!! migrate order (in one second)
                while True:
                    # Collect orders and process them after one second
                    start_time = time.time()
                    orders_collected = {}  # Temporarily store unmatched orders during collection

                    # Collect orders for one second
                    while time.time() - start_time < 1:
                        order_message = strategy_2_platform_order_q.get()
                        if order_message is not None:
                            stratID, order = order_message
                            current_datetime = (datetime.now() - self.time_delta).strftime("%Y-%m-%d %H:%M:%S")
                            order.orderID = self.order_id
                            self.order_id += 1
                            order.date = current_datetime.split(" ")[0]
                            order.submissionTime = current_datetime.split(" ")[1]
                            order.currStatusTime = current_datetime.split(" ")[1]
                            order.currStatus = "NEW"

                            with self.mutex:
                                self.order_status_and_executions[order.orderID] = [(order.ticker, order.date, order.submissionTime, \
                                                                                    order.direction, order.price, order.size, order.type), \
                                                                                   (None, order.date, order.currStatusTime, order.currStatus, None, None, 0, order.size)]
                                self.order_belonging[order.orderID] = stratID
                                with open('execution_and_status.txt', 'a') as f2:
                                    output = [order.ticker, order.direction, order.orderID, None, order.date, order.currStatusTime, order.currStatus, None, None, 0, order.size]
                                    f2.write(str(output) + '\n')

                            # Check if the order is a Market Order (MO) or Limit Order (LO)
                            if order.type == 'MO':  # Market Order
                                # Directly submit the Market Order to the exchange simulator queue
                                platform_2_exchSim_order_q.put(order)
                                with open('submitted_orders.txt', 'a') as f1:
                                    f1.write(str(order.outputAsArray()) + '\n')
                            elif order.type == 'LO':  # Limit Order
                                # Add the Limit Order to the unmatched orders dictionary
                                if (order.ticker, order.price) in orders_collected:
                                    if order.direction == "BUY":
                                        orders_collected[(order.ticker, order.price)][0].append(order)
                                    elif order.direction == "SELL":
                                        orders_collected[(order.ticker, order.price)][1].append(order)
                                else:
                                    if order.direction == "BUY":
                                        orders_collected[(order.ticker, order.price)] = [[order], []]
                                    elif order.direction == "SELL":
                                        orders_collected[(order.ticker, order.price)] = [[], [order]]

                    # Attempt internal cross-matching with the collected unmatched orders
                    self.cross_matching_and_order_submitting(orders_collected)

                '''
                # !!! cancel order
                #else:
                #    pass


    def consume_marketData(self, marketData_2_platform_q):
            start_time = time.time()
            while True:
                # print('[%d]Platform.consume_marketData' % (os.getpid()))
                marketData = marketData_2_platform_q.get()

                if marketData is None:
                    pass
                else:
                    if self.time_delta is None:
                        self.time_delta = datetime.now() - marketData.timeStamp

                    if marketData.ticker in TradingPlatform.subscribeStockTicker:
                        index = TradingPlatform.subscribeStockTicker.index(marketData.ticker)

                        self.mid_q_pair[index][0] = (marketData.askPrice1 + marketData.bidPrice1) / 2
                        if time.time() - start_time < 5:
                            continue
                        else:
                            start_time = time.time()
                            if self.mid_q_pair[index][1] is not None:
                                temp_thread_qs1 = threading.Thread(name='QuantStrategy1' + (datetime.now() - self.time_delta).strftime("%Y-%m-%d %H:%M:%S"), \
                                                                target=self.quantStrat[0].run, args=((marketData.ticker, TradingPlatform.subscribeFutureTicker[index], \
                                                                    self.mid_q_pair[index][0], self.mid_q_pair[index][1]), None))
                                temp_thread_qs1.start()
                                
                                # temp_thread_qs2 = threading.Thread(name='QuantStrategy2' + (datetime.now() - self.time_delta).strftime("%Y-%m-%d %H:%M:%S"), \
                                #                                    target=self.quantStrat[1].run, args=((marketData.ticker, TradingPlatform.subscribeFutureTicker[index], \
                                #                                    self.mid_q_pair[index][0], self.mid_q_pair[index][1]), None))
                                # temp_thread_qs2.start()

                            temp_thread_qs3 = threading.Thread(name='QuantStrategy3' + (datetime.now() - self.time_delta).strftime("%Y-%m-%d %H:%M:%S"), \
                                                                target=self.quantStrat[2].run, args=(marketData, None))
                            temp_thread_qs3.start()


                    elif marketData.ticker in TradingPlatform.subscribeFutureTicker:
                        index = TradingPlatform.subscribeFutureTicker.index(marketData.ticker)
                        self.mid_q_pair[index][1] = (marketData.askPrice1 + marketData.bidPrice1) / 2


    def handle_execution(self, exchSim_2_platform_execution_q):
        print('[%d]Platform.handle_execution' % (os.getpid(),))
        
        while True:
            execution = exchSim_2_platform_execution_q.get()
            if execution is None:
                pass
            else:
                StartID = self.order_belonging[execution.orderID]
                with self.mutex:
                    if execution.direction == 'ACK':
                        self.order_status_and_executions[execution.orderID].append((execution.execID, execution.date, \
                                                                                    execution.timeStamp, 'ACK', None, None, \
                                                                                    0, self.order_status_and_executions[execution.orderID][0][5]))
                    elif execution.direction == 'BI':
                        if execution.size == self.order_status_and_executions[execution.orderID][-1][-1]:
                            self.order_status_and_executions[execution.orderID].append((execution.execID, execution.date, \
                                                                                        execution.timeStamp, 'Filled', \
                                                                                        execution.price, execution.size, \
                                                                                        self.order_status_and_executions[execution.orderID][-1][-2] + execution.size, 0))
                            self.position_and_cost[StartID][execution.ticker][0] += execution.size
                            self.position_and_cost[StartID][execution.ticker][1] += execution.price * execution.size
                        else:
                            self.order_status_and_executions[execution.orderID].append((execution.execID, execution.date, \
                                                                                        execution.timeStamp, 'PartiallyFilled', \
                                                                                        execution.price, execution.size, \
                                                                                        self.order_status_and_executions[execution.orderID][-1][-2] + execution.size, \
                                                                                        self.order_status_and_executions[execution.orderID][-1][-1] - execution.size))
                            self.position_and_cost[StartID][execution.ticker][0] += execution.size
                            self.position_and_cost[StartID][execution.ticker][1] += execution.price * execution.size
                        temp_thread_execution2Strat = threading.Thread(name='QuantStrategy'+ str(StartID) + '_Execution' + (datetime.now() - self.time_delta).strftime("%Y-%m-%d %H:%M:%S"), \
                                                                       target=self.quantStrat[StartID-1].run, args=(None, execution))
                    elif execution.direction == 'SI':
                        if execution.size == self.order_status_and_executions[execution.orderID][-1][-1]:
                            self.order_status_and_executions[execution.orderID].append((execution.execID, execution.date, \
                                                                                        execution.timeStamp, 'Filled', \
                                                                                        execution.price, execution.size, \
                                                                                        self.order_status_and_executions[execution.orderID][-1][-2] + execution.size, 0))
                            self.position_and_cost[StartID][execution.ticker][0] -= execution.size
                            self.position_and_cost[StartID][execution.ticker][1] -= execution.price * execution.size
                        else:
                            self.order_status_and_executions[execution.orderID].append((execution.execID, execution.date, \
                                                                                        execution.timeStamp, 'PartiallyFilled', \
                                                                                        execution.price, execution.size, \
                                                                                        self.order_status_and_executions[execution.orderID][-1][-2] + execution.size, \
                                                                                        self.order_status_and_executions[execution.orderID][-1][-1] - execution.size))
                            self.position_and_cost[StartID][execution.ticker][0] -= execution.size
                            self.position_and_cost[StartID][execution.ticker][1] -= execution.price * execution.size
                        temp_thread_execution2Strat = threading.Thread(name='QuantStrategy'+ str(StartID) + '_Execution' + (datetime.now() - self.time_delta).strftime("%Y-%m-%d %H:%M:%S"), \
                                                                       target=self.quantStrat[StartID-1].run, args=(None, execution))
                    # !!! elif execution.direction == 'Cancel':
                    # !!! elif execution.direction == 'Cancel Fail':
                
                    else:
                        pass

                with open('execution_and_status.txt', 'a') as f2:
                    output = [execution.ticker, self.order_status_and_executions[execution.orderID][0][3], execution.orderID]
                    output.extend(list(self.order_status_and_executions[execution.orderID][-1]))
                    f2.write(str(output) + '\n')
                self.quantStrat[StartID-1].run(None, execution)


    def log_position_and_pnl(self):
        while True:
            time.sleep(5)
            if self.time_delta is None:
                continue
            else:
                current_datetime = (datetime.now() - self.time_delta).strftime("%Y-%m-%d %H:%M:%S")
                with self.mutex:
                    with open('position_and_pnl_log.txt', 'a') as f:
                        for stratID in self.position_and_cost:
                            total_pnl = 0
                            for ticker, position_cost in self.position_and_cost[stratID].items():
                                total_size = position_cost[0]
                                total_cost = position_cost[1]
                                if ticker in TradingPlatform.subscribeStockTicker:
                                    index = TradingPlatform.subscribeStockTicker.index(ticker)
                                    if self.mid_q_pair[index][0] is not None:
                                        pnl = self.mid_q_pair[index][0] * total_size - total_cost
                                    else:
                                        pnl = 0
                                    total_pnl += pnl
                                else:
                                    index = TradingPlatform.subscribeFutureTicker.index(ticker)
                                    if self.mid_q_pair[index][1] is not None:
                                        pnl = self.mid_q_pair[index][1] * total_size - total_cost
                                    else:
                                        pnl = 0
                                    total_pnl += pnl
                                f.write(f"Current time {current_datetime}, Strategy {stratID}, Ticker {ticker}, Total Cost {total_cost}, Total Size {total_size}, PnL {pnl}\n")
                            f.write(f"Current time {current_datetime}, Strategy {stratID}, Total PnL {total_pnl}\n")
