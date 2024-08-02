#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Created on Thu Jun 20 10:26:05 2020

@author: hongsong chou
"""

import os
import time
from common.OrderBookSnapshot_FiveLevels import OrderBookSnapshot_FiveLevels
from common.Strategy import Strategy
from common.SingleStockOrder import SingleStockOrder
from common.SingleStockExecution import SingleStockExecution
import random
import lightgbm as lgb
import pandas as pd
import numpy as np
import datetime
import threading
import numpy as np
from joblib import load
import pandas as pd
from tensorflow import keras
import time

class QuantStrategy(Strategy):
    
    def __init__(self, stratID, stratName, stratAuthor, ticker, day, strategy_2_platform_order_q):
        super(QuantStrategy, self).__init__(stratID, stratName, stratAuthor) #call constructor of parent
        self.ticker = ticker #public field
        self.day = day #public field
        self.strategy_2_platform_order_q = strategy_2_platform_order_q

        
    def getStratDay(self):
        return self.day
    
    def run(self, marketData, execution):
        time_now = time.time()

        # get as formatted pretty time
        time_now_pretty = time.asctime(time.localtime(time_now))

        print(f'Time now: {time_now_pretty} Market Data {marketData}')

        if (marketData is None) and (execution is None):
            return None
        elif (marketData is None) and ((execution is not None) and (isinstance(execution, SingleStockExecution))):
            #handle executions
            # print('[%d] ----- H ------ Strategy.handle_execution' % (os.getpid()))
            print(execution.outputAsArray())
            return None
        elif ((marketData is not None) and (isinstance(marketData, OrderBookSnapshot_FiveLevels))) and (execution is None):
            #handle new market data, then create a new order and send it via quantTradingPlatform.
            # return SingleStockOrder('testTicker','2019-07-05',time.asctime(time.localtime(time.time())))
            # return SingleStockOrder('2610','2019-07-05',time.asctime(time.localtime(time.time())))
            # print('[%d] ----- H ------ Strategy.handle_execution' % (os.getpid()))

            # send single stock order to platform
            order_object = SingleStockOrder(marketData.ticker,'2019-07-05',time.asctime(time.localtime(time.time())))

            # randomly do buy or sell
            if random.randint(0,1) == 0:
                order_object.direction = 'BUY'
            else:
                order_object.direction = 'SELL'


            order_object.type = 'MO'
            order_object.size = 3


            order_to_put = [
                self.getStratID(),
                order_object
            ]


      
            self.strategy_2_platform_order_q.put(order_to_put)
            print(f"{os.getpid()} Strat ID: {self.getStratID()}QuantStrategy.run Order Info : ! {order_object.outputAsArray()}")
        
            return

        else:
            return None
                


class HarshQuantStrategy(Strategy):
    
    def __init__(self, stratID, stratName, stratAuthor, ticker, day, strategy_2_platform_order_q):
        super(HarshQuantStrategy, self).__init__(stratID, stratName, stratAuthor) #call constructor of parent
        self.ticker = ticker #public field
        self.day = day #public field
        self.strategy_2_platform_order_q = strategy_2_platform_order_q

        
    def getStratDay(self):
        return self.day
    

    def load_selected_model_and_scaler(
            self,
            model_name,
            scaler_name
    ):
        return keras.models.load_model(model_name), load(scaler_name)  


    def predict_last_px_cnn_ltsm(self,ticker,input_row, model_type='CNN'):
        model_path = f"/Users/hssingh/mafs5360-final-project-v2/PrototypeWIP/nn_models/{ticker}_cnn_lastpx_model.h5"
        scaler_path = f"/Users/hssingh/mafs5360-final-project-v2/PrototypeWIP/nn_models/{ticker}_scaler.joblib"

        loaded_model, scaler = self.load_selected_model_and_scaler(
            model_path,
            scaler_path
        )

        # Convert input_row to a DataFrame
        input_df = pd.DataFrame([input_row])
        
        # Scale the features using the same scaler
        input_scaled = scaler.transform(input_df)
        
        # Reshape for CNN or LSTM
        if model_type == 'CNN':
            input_reshaped = input_scaled.reshape(-1, input_scaled.shape[1], 1)  # Shape for CNN: (samples, features, 1)
        elif model_type == 'LSTM':
            input_reshaped = input_scaled.reshape(-1, 1, input_scaled.shape[1])  # Shape for LSTM: (samples, timesteps, features)
        else:
            raise ValueError("model_type should be either 'CNN' or 'LSTM'")
        
        # Make prediction
        prediction = loaded_model.predict(input_reshaped)
        
        return prediction[0][0]  # Return the predicted value




    def run(self, marketData, execution):
        time_now = time.time()

        # get as formatted pretty time
        time_now_pretty = time.asctime(time.localtime(time_now))

        print(f'Time now: {time_now_pretty} Market Data {marketData}')

        if (marketData is None) and (execution is None):
            return None
        elif (marketData is None) and ((execution is not None) and (isinstance(execution, SingleStockExecution))):
            #handle executions
            # print('[%d] ----- H ------ Strategy.handle_execution' % (os.getpid()))
            print(execution.outputAsArray())
            return None
        elif ((marketData is not None) and (isinstance(marketData, OrderBookSnapshot_FiveLevels))) and (execution is None):
            #handle new market data, then create a new order and send it via quantTradingPlatform.
            # return SingleStockOrder('testTicker','2019-07-05',time.asctime(time.localtime(time.time())))
            # return SingleStockOrder('2610','2019-07-05',time.asctime(time.localtime(time.time())))
            # print('[%d] ----- H ------ Strategy.handle_execution' % (os.getpid()))


            midq = (marketData.askPrice1 + marketData.bidPrice1) / 2


            # get ticker
            ticker = marketData.ticker

            # get row information in this order
            row = [marketData.askPrice5, marketData.askPrice4, marketData.askPrice3, marketData.askPrice2, marketData.askPrice1,
                     marketData.bidPrice1, marketData.bidPrice2, marketData.bidPrice3, marketData.bidPrice4, marketData.bidPrice5,
                     marketData.askSize5, marketData.askSize4, marketData.askSize3, marketData.askSize2, marketData.askSize1,
                     marketData.bidSize1, marketData.bidSize2, marketData.bidSize3, marketData.bidSize4, marketData.bidSize5]
            
            # get prediction
            prediction = self.predict_last_px_cnn_ltsm(ticker, row, model_type='CNN')

            if prediction > midq:
                direction = 'BUY'
            else:
                direction = 'SELL'


            # send single stock order to platform
            order_object = SingleStockOrder(marketData.ticker,'2019-06-28',time.asctime(time.localtime(time.time())))

            order_object.direction = direction

            order_object.type = 'MO'
            order_object.size = 1


            order_to_put = [
                self.getStratID(),
                order_object
            ]


      
            self.strategy_2_platform_order_q.put(order_to_put)
            print(f"{os.getpid()} Strat ID: {self.getStratID()}QuantStrategy.run Order Info : ! {order_object.outputAsArray()}")
        
            return

        else:
            return None
                


class ZJQuantStrategy(Strategy):
    
    def __init__(self, stratID, stratName, stratAuthor, ticker, day, strategy_2_platform_order_q):
        super(ZJQuantStrategy, self).__init__(stratID, stratName, stratAuthor) #call constructor of parent
        self.ticker = ticker #public field
        self.day = day #public field
        self.initial_cash = 100000.0

        self.strategy_2_platform_order_q = strategy_2_platform_order_q

        self.cash = pd.DataFrame(columns=['date', 'timestamp', 'cash'])

        self.networth = pd.DataFrame(columns=['date','timestamp','networth'])

        self.current_position = pd.DataFrame(columns=['ticker', 'quantity', 'price','amount'])

        self.submitted_order = pd.DataFrame(
            columns=['date', 'submissionTime', 'ticker', 'orderID', 'currStatus', 'currStatusTime', 'direction',
                     'price', 'size', 'type'])

        self.metrics = pd.DataFrame(columns=['cumulative_return', 'portfolio_volatility', 'max_drawdown'])
        
        # Load model 
        self.all_market_data = {}
        self.last_position_time = {}
        self.future2stock = {'DBF': '2610'} #, 'GLF': 2392, 'HCF': 2498, 'HSF': 2618, 'IPF': 3035, 'NEF': 3264, 'NLF': 5347, 'NYF': '0050', 'QLF': 3374, 'RLF': 6443}
        self.stock2future = {v: k for k, v in self.future2stock.items()}
        self.future_tickers = self.future2stock.keys()
        self.stock_tickers = list(self.future2stock.values())
        self.if_enough_data = False
        self.target = 100
        self.models = {}
        for future_ticker in self.future2stock.keys():
            self.models[future_ticker] = lgb.Booster(model_file=f'./model_params/model_{future_ticker}_Y_M_{self.target}.txt')
        
    def getStratDay(self):
        return self.day
    
    def run(self, marketData, execution,marketDataObj=None):
        if (marketData is None) and (execution is None):
            return None
        elif (marketData is None) and ((execution is not None) and (isinstance(execution, SingleStockExecution))):
            #handle executions
            print('[%d] Strategy.handle_execution' % (os.getpid()))
            date, ticker, timeStamp, execID, orderID, direction, price, size, comm = execution.outputAsArray()
            execID = str(execID)
            orderID = str(orderID)
            direction = direction.lower()
            # locate the row in self.submitted_order with orderID, then update the currStatus and currStatusTime, check if the size of executed order is the same as the size of submitted order, if less than, the status is PartiallyFilled, if equal, the status is Filled, if more than, return Non and print error
            submitted_order = self.submitted_order[self.submitted_order['orderID']==orderID]

            #check if the orderID is in self.submitted_order
            if submitted_order is None:
                print('Error: orderID not in submitted_order')
                return None
            else:
                #locate the row in self.submitted_order with orderID
                submitted_size = submitted_order.size
                #check if the size of executed order is the same as the size of submitted order
                if size < submitted_size:
                    submitted_order.currStatus = 'PartiallyFilled'
                    #update the size of submitted order
                    submitted_order.size = submitted_order.size - size
                elif size > submitted_size:
                    print('Error: size of executed order is more than the size of submitted order')
                    return None
                else:
                    # update the currStatus and currStatusTime
                    submitted_order.currStatus = 'Filled'
                submitted_order.currStatusTime = timeStamp
            current_position = self.current_position[self.current_position['ticker']==ticker].iloc[-1]

            # update current position with ticker
            if direction == 'buy':
                if current_position is not None:
                    current_position.quantity = current_position.quantity + size
                    current_position.inception_timestamp=timeStamp
                else:
                    new_position = pd.DataFrame({'price': price, 'inception_timestamp': timeStamp, 'ticker': ticker, 'quantity': size})
                    current_position = pd.concat([current_position, new_position])
            elif direction == 'sell':
                if current_position is not None:
                    current_position.quantity = current_position.quantity - size
                    current_position.inception_timestamp = timeStamp
                else:
                    new_position = pd.DataFrame({'price': price, 'inception_timestamp': timeStamp, 'ticker': ticker, 'quantity': size})
                    current_position = pd.concat([current_position, new_position])
            else:
                print('Error: direction is neither Buy nor Sell')
                return None

            cash_position = self.current_position.iloc[-1]

            if direction == 'buy':
                cash_position.quantity = cash_position.quantity - (price * size + comm)
            elif direction == 'sell':
                cash_position.quantity = cash_position.quantity + price * size - comm

            positions = self.current_position

            current_networth = pd.DataFrame({'date': date, 'timestamp': timeStamp, 'networth': 0})

            for position in positions:
                current_networth.networth = current_networth.networth + position.price * position.quantity

            self.networth = pd.concat([self.networth, current_networth])
            networthes = self.networth

            #update self.metrics with self.networth if self.networth has more than 1 row
            if networthes.shape[0] > 1:
                cumulative_return = (networthes.iloc[-1]['networth'] / networthes.iloc[0]['networth'] - 1) * 100

                # calculate portfolio volatility
                portfolio_volatility = networthes['networth'].pct_change().std() * 100

                # calculate max drawdown
                max_drawdown = 0
                for i in range(1, len(networthes)):
                    if networthes.iloc[i]['networth'] > networthes.iloc[i - 1]['networth']:
                        continue
                    else:
                        drawdown = (networthes.iloc[i]['networth'] / networthes.iloc[i - 1]['networth'] - 1) * 100
                        if drawdown < max_drawdown:
                            max_drawdown = drawdown

                metrics = self.metrics
                if metrics is None:
                    metrics = pd.DataFrame({'cumulative_return': cumulative_return, 'portfolio_volatility': portfolio_volatility,
                                      'max_drawdown': max_drawdown})
                else:
                    metrics.cumulative_return = cumulative_return
                    metrics.portfolio_volatility = portfolio_volatility
                    metrics.max_drawdown = max_drawdown
            print(execution.outputAsArray())
            return None
        # elif ((marketData is not None) and (isinstance(marketData, OrderBookSnapshot_FiveLevels))) and (execution is None):
        elif (marketData is not None) and (execution is None):
            #handle new market data, then create a new order and send it via quantTradingPlatform.
            # current_market_data = marketData.outputAsDataFrame()

            current_market_data = marketDataObj.outputAsDataFrame()
            if current_market_data.iloc[0]['askPrice1'] == 0 or current_market_data.iloc[0]['bidPrice1'] == 0:
                print('Error: askPrice1 or bidPrice1 is empty')
                return None

            current_date = current_market_data.iloc[0]['date']
            current_time = current_market_data.iloc[0]['time']

            #handle new market data, then create a new order and send it via quantTradingPlatform if needed
            #update networth and current_position if there is an open position related to this ticker
            ticker = current_market_data.iloc[0]['ticker']
            # if it is a futrue ticker with month, transfer it 
            if ticker not in self.stock2future.keys():
                ticker = ticker[:3]
                if ticker not in self.stock2future.keys():
                    #raise ValueError('Future ticker probelm')
                    pass
            related_position = self.current_position[self.current_position['ticker']==ticker]
            if related_position is not None:
                #get the current price of the ticker
                current_price = (current_market_data.iloc[0]['askPrice1'] + current_market_data.iloc[0]['bidPrice1']) / 2
                #update the price of the position
                related_position.price = current_price
                #update the networth
                positions = self.current_position
                current_networth = pd.DataFrame([[current_date, current_time, 0]], columns=['date','timestamp','networth'])
                for position in positions:
                    current_networth.networth = current_networth.networth + position.price * position.quantity

            networthes = self.networth
            if self.current_position_dataframe.empty:
                self.current_position_dataframe = pd.DataFrame({'ticker':'cash','quantity':10000.0,'price':1}, index=[0])

            if self.metrics.empty:
                self.metrics = pd.DataFrame({'cumulative_return':0,'portfolio_volatility':0,'max_drawdown':0}, index=[0])

            #update networth and current_position_dataframe if there is an open position, and save all dataframe to a local csv file.
            #update current cash with the cash from the last row of self.cash
            current_cash = self.cash.iloc[-1]['cash']
            #check if self.cash has the same time as current_time, if not, add a new row to self.cash
            if self.cash.iloc[-1]['timestamp'] != current_time:
                self.cash = pd.concat([self.cash, pd.DataFrame({'date': current_date, 'timestamp': current_time, 'cash': current_cash}, index=[0])])

            current_networth = current_cash
            if len(self.current_position) > 0:
                #loop through all the positions (keys of self.current_position) to find the current price from current_market_data, the current price is the average price of askPrice1 and bidPrice1
                for ticker in self.current_position.keys():
                    #check if the ticker is in the current_market_data, if not, skip this ticker
                    if ticker not in current_market_data['ticker'].values:
                        continue
                    current_price = (current_market_data.loc[current_market_data['ticker'] == ticker]['askPrice1'].values[0] + current_market_data.loc[current_market_data['ticker'] == ticker]['bidPrice1'].values[0])/2
                    self.position_price = pd.concat([self.position_price, pd.DataFrame({'date':current_date, 'timestamp':current_time, 'ticker':ticker, 'price':current_price}, index=[0])])

                for ticker in self.current_position.keys():
                    #check if ticker is in self.position_price, if not, skip this ticker
                    if ticker not in self.position_price['ticker'].values:
                        continue
                    #update current_networth with the current price of ticker
                    current_networth += self.current_position[ticker] * self.position_price.loc[self.position_price['ticker'] == ticker].iloc[-1]['price']


            # check if self.networth has the same date as current_date, if not, add a new row to self.networth, if so, update the networth
            if self.networth.iloc[-1]['timestamp'] != current_time:
                self.networth = pd.concat([self.networth, pd.DataFrame({'date': current_date, 'timestamp': current_time,'networth': current_networth}, index=[0])])
            else:
                self.networth.loc[self.networth['timestamp'] == current_time, 'networth'] = current_networth

            print(self.current_position)

            #construct self.current_position_dataframe from self.current_position, price and current cash, ticker by ticker
            self.current_position_dataframe = pd.DataFrame(columns=['ticker','quantity','price'])
            for ticker in self.current_position.keys():
                #check if ticker is in self.position_price, if the price of this ticker is 0 and concate
                if ticker not in self.position_price['ticker'].values:
                    self.current_position_dataframe = pd.concat([self.current_position_dataframe, pd.DataFrame({'ticker':ticker,'quantity':self.current_position[ticker],'price':0}, index=[0])])
                else:
                    self.current_position_dataframe = pd.concat([self.current_position_dataframe, pd.DataFrame({'ticker':ticker,'quantity':self.current_position[ticker],'price':self.position_price.loc[self.position_price['ticker'] == ticker].iloc[-1]['price']}, index=[0])])

            self.current_position_dataframe = pd.concat([self.current_position_dataframe, pd.DataFrame({'ticker':'cash','quantity':current_cash,'price':1}, index=[0])])
            #update self.metrics with self.networth if self.networth has more than 1 row
            if networthes.shape[0] > 1:
                cumulative_return = (networthes.iloc[-1]['networth'] / networthes.iloc[0]['networth'] - 1) * 100

                # calculate portfolio volatility
                portfolio_volatility = networthes['networth'].pct_change().std() * 100

                # calculate max drawdown
                max_drawdown = 0
                for i in range(1, len(networthes)):
                    if networthes.iloc[i]['networth'] > networthes.iloc[i - 1]['networth']:
                        continue
                    else:
                        drawdown = (networthes.iloc[i]['networth'] / networthes.iloc[i - 1]['networth'] - 1) * 100
                        if drawdown < max_drawdown:
                            max_drawdown = drawdown

                metrics = self.metrics
                if metrics is None:
                    metrics = pd.DataFrame({'cumulative_return': cumulative_return, 'portfolio_volatility': portfolio_volatility, 'max_drawdown': max_drawdown})
                else:
                    metrics.cumulative_return = cumulative_return
                    metrics.portfolio_volatility = portfolio_volatility
                    metrics.max_drawdown = max_drawdown

            tradeOrder = None
            current_cash_query = self.cash.iloc[-1]
            current_cash = 0

            if current_cash_query is not None:
                current_cash = current_cash_query['cash']

            # Save market data to self.all_market_data
            if ticker not in self.all_market_data.keys():
                self.all_market_data[ticker] = current_market_data
            self.all_market_data[ticker] = pd.concat([self.all_market_data[ticker], current_market_data], ignore_index=True)
            # Check future/stock
            # Check if future ticker consists of month
            if ticker in self.future_tickers.keys():
                #save all self dataframe to a local csv file
                self.networth.to_csv('./networth.csv', index=False)
                self.cash.to_csv('./cash.csv', index=False)
                self.position_price.to_csv('./position_price.csv', index=False)
                self.current_position_dataframe.to_csv('./current_position.csv', index=False)
                self.metrics.to_csv('./metrics.csv', index=False)
            
            # Save market data
            # Check if we have this ticker's data, if not, we create a new dataframe, if have, we concat the new data to the old dataframe
            if ticker not in self.all_market_data.keys():
                self.all_market_data[ticker] = current_market_data
            self.all_market_data[ticker] = pd.concat([self.all_market_data[ticker], current_market_data], ignore_index=True)
            
            # Since we just trade stock, we can check if it is future ticker, if it, we just save the data and return None
            if ticker in self.future_tickers.keys():
                return None
            # Check if we have enough data to make decision
            if self.if_enough_data == False:
                stk_time_delta = (self.all_market_data[ticker]['time'].iloc[-1] - self.all_market_data[ticker]['time'].iloc[0]).total_seconds()
                future_time_delta = (self.all_market_data[self.stock2future[ticker]]['time'].iloc[-1] - self.all_market_data[self.stock2future[ticker]]['time'].iloc[0]).total_seconds()
                if stk_time_delta < 110 or future_time_delta < 110:
                    return None
            self.if_enough_data = True
            # Check if stock in current position
            current_position = self.current_position[self.current_position['ticker']==ticker].iloc[-1]
            if current_position is not None and current_position.quantity != 0:
                # if holding time < 10s, we will return None
                last_position_time = current_position.inception_timestamp
                if (current_time - last_position_time).total_seconds() < 10:
                    # Keep this position
                    return None
                else:
                    # Balance the position
                    direction = 'buy' if current_position.quantity < 0 else 'sell'
                    quantity = abs(self.current_position[ticker])
                    tradeOrder = SingleStockOrder(ticker, datetime.datetime.now().strftime('%Y-%m-%d'), datetime.datetime.now(), 
                                                  datetime.datetime.now(), 'New', direction, current_price,quantity , 'MO')
                    
                    self.strategy_2_platform_order_q.put([self.getStratID(), tradeOrder])
                    # return tradeOrder
            # if we don't have this ticker's position, we will make a new order decision
            # get the latest 100 seconds market data and downsample by 10s and get the last data in each 10s
            delay_100s = current_time - datetime.timedelta(seconds=100)
            input_stock_data = self.all_market_data[ticker].loc[self.all_market_data[ticker]['time'] > delay_100s].resample('10s', on='time').last().reset_index()
            input_future_data = self.all_market_data[self.stock2future[ticker]].loc[self.all_market_data[self.stock2future[ticker]]['time'] > delay_100s].resample('10s', on='time').last().reset_index()
            # get feature
            features_df = self.generate_features(input_stock_data, input_future_data)
            # get prediction
            prediction = self.models[ticker].predict(features_df).iloc[-1]
            # make decision
            if prediction > 0:
                direction = 'buy'
            elif prediction < 0:
                direction = 'sell'
            else:
                return None
            quantity = self.initial_cash * 0.1 // current_price
            tradeOrder = SingleStockOrder(ticker, datetime.datetime.now().strftime('%Y-%m-%d'), datetime.datetime.now(),
                                            datetime.datetime.now(), 'New', direction, current_price, quantity, 'MO')
            date, ticker, submissionTime, orderID, currStatus, currStatusTime, direction, price, size, type = tradeOrder.outputAsArray()
            new_order = pd.DataFrame([[date, submissionTime, ticker, orderID, currStatus, currStatusTime, direction, price, size, type]], 
                                    columns=['date', 'submissionTime', 'ticker', 'orderID', 'currStatus', 'currStatusTime', 'direction',
                                            'price', 'size', 'type'])
            self.submitted_order = pd.concat([self.submitted_order, new_order])

            self.strategy_2_platform_order_q.put([self.getStratID(), tradeOrder])
            # return tradeOrder
        else:
            return None

    def generate_features(self, futureData_date, stockData_date):
        basicCols = ['date', 'time', 'sAskPrice1','sBidPrice1','sWmidQ', 'fAskPrice1','fBidPrice1', 'fWmidQ', 'spreadRatio']
        featureCols = []
        TIME_LIST = [10, 30, 50, 100, 200, 500]

        for i in TIME_LIST:
            
            featureCols.extend([f'fHisRtn_{i}'])
            featureCols.extend([f'spreadRatio_{i}'])
            featureCols.extend([f'volumeImbalanceRatio_{i}'])
            featureCols.extend([f'depthImb5_{i}'])
            featureCols.extend([f'sHisRtn_{i}'])
            featureCols.extend([f'stockSpreadRatio_{i}'])
            featureCols.extend([f'stockVolumeImbalanceRatio_{i}'])
            featureCols.extend([f'stockDepthImb5_{i}'])
            
            for j in range(1, 6):
                featureCols.extend([f'fAskSize{j}_{i}'])
                featureCols.extend([f'fBidSize{j}_{i}'])
                featureCols.extend([f'sAskSize{j}_{i}'])
                featureCols.extend([f'sBidSize{j}_{i}'])

        df = pd.DataFrame(index=stockData_date.index, columns=basicCols+featureCols)
        df['date'] = stockData_date['date']
        df['time'] = stockData_date['time']   
                    
        #========= Size Normalization and Price from 1 to 5 =========
        for i in range(1, 6):
            df[f'fAskPrice{i}'] = futureData_date[f'askPrice{i}']
            df[f'fBidPrice{i}'] = futureData_date[f'bidPrice{i}']
            df[f'fAskSize{i}'] = futureData_date[f'askSize{i}'] / futureData_date[['askSize1', 'askSize2', 'askSize3', 'askSize4', 'askSize5']].max(axis=1)
            df[f'fBidSize{i}'] = futureData_date[f'bidSize{i}'] / futureData_date[['bidSize1', 'bidSize2', 'bidSize3', 'bidSize4', 'bidSize5']].max(axis=1)
            
            df[f'sAskPrice{i}'] = stockData_date[f'askPrice{i}']
            df[f'sBidPrice{i}'] = stockData_date[f'bidPrice{i}']
            df[f'sAskSize{i}'] = stockData_date[f'askVolume{i}'] / stockData_date[[f'askVolume{depth}' for depth in range(1, 6)]].max(axis=1)
            df[f'sBidSize{i}'] = stockData_date[f'bidVolume{i}'] / stockData_date[[f'bidVolume{depth}' for depth in range(1, 6)]].max(axis=1)
        
        #====================== weighted MidQ ========================
        df['fWmidQ'] = (df['fAskPrice1'] * futureData_date['bidSize1'] + df['fBidPrice1'] * futureData_date['askSize1']) / (futureData_date['bidSize1'] + futureData_date['askSize1'])
        df['sAskPrice1'] = stockData_date['askPrice1']
        df['sBidPrice1'] = stockData_date['bidPrice1']
        df['sWmidQ'] = (stockData_date['askPrice1'] * stockData_date['bidVolume1'] + stockData_date['bidPrice1'] * stockData_date['askVolume1']) / (stockData_date['askVolume1'] + stockData_date['bidVolume1'])
        
        #====================== Spread Ratio =========================
        ask = np.array([df[f'fAskPrice{i}'] * df[f'fAskSize{i}'] * (1 - (i - 1) / 5) for i in range(1, 6)]).sum(axis=0)
        bid = np.array([df[f'fBidPrice{i}'] * df[f'fBidSize{i}'] * (1 - (i - 1) / 5) for i in range(1, 6)]).sum(axis=0)
        df['spreadRatio'] = (ask - bid) / (ask + bid)

        ask = np.array([df[f'sAskPrice{i}'] * df[f'sAskSize{i}'] * (1 - (i - 1) / 5) for i in range(1, 6)]).sum(axis=0)
        bid = np.array([df[f'sBidPrice{i}'] * df[f'sBidSize{i}'] * (1 - (i - 1) / 5) for i in range(1, 6)]).sum(axis=0)
        df['stockSpreadRatio'] = (ask - bid) / (ask + bid)

        #================= Order Imbalance Ratio =====================
        delta_size_bid = np.where(df['fBidPrice1'] < df['fBidPrice1'].shift(1), 0, np.where(df['fBidPrice1'] == df['fBidPrice1'].shift(1), df['fBidSize1'] - df['fBidSize1'].shift(1), df['fBidSize1']))
        delta_size_ask = np.where(df['fAskPrice1'] > df['fAskPrice1'].shift(1), 0, np.where(df['fAskPrice1'] == df['fAskPrice1'].shift(1), df['fAskSize1'] - df['fAskSize1'].shift(1), df['fAskSize1']))
        df['fOrderImbalance'] = delta_size_bid - delta_size_ask
        df['fOrderImbalance'] = df['fOrderImbalance'].rolling(10, min_periods=10).apply(lambda x: (x[-1] - x.mean()) / x.std())

        delta_size_bid = np.where(df['sBidPrice1'] < df['sBidPrice1'].shift(1), 0, np.where(df['sBidPrice1'] == df['sBidPrice1'].shift(1), df['sBidSize1'] - df['sBidSize1'].shift(1), df['sBidSize1']))
        delta_size_ask = np.where(df['sAskPrice1'] > df['sAskPrice1'].shift(1), 0, np.where(df['sAskPrice1'] == df['sAskPrice1'].shift(1), df['sAskSize1'] - df['sAskSize1'].shift(1), df['sAskSize1']))
        df['stockOrderImbalance'] = delta_size_bid - delta_size_ask
        df['stockOrderImbalance'] = df['stockOrderImbalance'].rolling(10, min_periods=10).apply(lambda x: (x[-1] - x.mean()) / x.std())

        #====================== Depth Imbalance ======================
        bid_part = np.log(futureData_date[[f'bidSize{i}' for i in range(1, 6)]].sum(axis=1))
        ask_part = np.log(futureData_date[[f'askSize{i}' for i in range(1, 6)]].sum(axis=1))
        df['depthImb5'] = (bid_part - ask_part) / (bid_part + ask_part)

        bid_part = np.log(stockData_date[[f'bidVolume{i}' for i in range(1, 6)]].sum(axis=1))
        ask_part = np.log(stockData_date[[f'askVolume{i}' for i in range(1, 6)]].sum(axis=1))
        df['stockDepthImb5'] = (bid_part - ask_part) / (bid_part + ask_part)

        for i in TIME_LIST:
            df[f'fHisRtn_{i}'] = np.log(df['fWmidQ']) - np.log(df['fWmidQ'].shift(i))
            df[f'spreadRatio_{i}'] = df['spreadRatio'].rolling(i).mean()
            df[f'volumeImbalanceRatio_{i}'] = df['fOrderImbalance'].rolling(i).mean()
            df[f'depthImb5_{i}'] = df['depthImb5'].rolling(i).mean()

            df[f'Y_M_{i}'] = np.log(df['sWmidQ'].shift(-i)) - np.log(df['sWmidQ'])
            df[f'sHisRtn_{i}'] = np.log(df['sWmidQ']) - np.log(df['sWmidQ'].shift(i))
            df[f'stockSpreadRatio_{i}'] = df['stockSpreadRatio'].rolling(i).mean()
            df[f'stockVolumeImbalanceRatio_{i}'] = df['stockOrderImbalance'].rolling(i).mean()
            df[f'stockDepthImb5_{i}'] = df['stockDepthImb5'].rolling(i).mean()
            
            for j in range(1, 6):
                df[f'fAskSize{j}_{i}'] = df[f'fAskSize{j}'].shift(i)
                df[f'fBidSize{j}_{i}'] = df[f'fBidSize{j}'].shift(i)
                df[f'sAskSize{j}_{i}'] = df[f'sAskSize{j}'].shift(i)
                df[f'sBidSize{j}_{i}'] = df[f'sBidSize{j}'].shift(i)
                df[f'sAskSize{j}_{i}'] = df[f'sAskSize{j}'].shift(i)
                df[f'sBidSize{j}_{i}'] = df[f'sBidSize{j}'].shift(i)

        # Convert inf to nan to 0
        df = df[featureCols].iloc[[-1]]
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        # get the last row
        return df
        

class YFQuantStrategy(Strategy):
    
    def __init__(self, stratID, stratName, stratAuthor, ticker, day, strategy_2_platform_order_q):
        super(YFQuantStrategy, self).__init__(stratID, stratName, stratAuthor) #call constructor of parent
        self.ticker = ticker #public field
        self.day = day #public field
        self.strategy_1_status = 0 # this is to record the status of strategy1 from yufeng
        self.strategy_2_platform_order_q = strategy_2_platform_order_q # this is the queue to send order to platform
        self.mutex = threading.Lock()

    def getStratDay(self):
        return self.day
    
    def strategy_1_calculate(self, stock_price, futures_price, strategy_1_status):
        if stock_price == 0 :
            return None
        if (futures_price / stock_price >= 1.001) and (strategy_1_status == 0):
            return 1
        elif (futures_price / stock_price < 1) and (strategy_1_status == 1):
            return 0
        elif (futures_price / stock_price <= 0.995) and (strategy_1_status == 0):
            return -1
        elif (futures_price / stock_price > 1) and (strategy_1_status == -1):
            return 0
        else: 
            return None
    
    def run(self, marketData, execution):
        with self.mutex:
            if (marketData is None) and (execution is None):
                return None
            elif (marketData is None) and ((execution is not None) and (isinstance(execution, SingleStockExecution))):
                #handle executions
                print('[%d] Strategy.handle_execution' % (os.getpid()))
                print(execution.outputAsArray())
                return None
            # elif ((marketData is not None) and (isinstance(marketData, OrderBookSnapshot_FiveLevels))) and (execution is None):
            elif (marketData is not None) and (execution is None):

                #handle new market data, then create a new order and send it via quantTradingPlatform.
                # return SingleStockOrder('testTicker','2019-07-05',time.asctime(time.localtime(time.time())))
                

                # get the stock, futures, stock price and futures price
                ticker_stock, ticker_futures, stock_price, futures_price = marketData
                # generate the signal
                strategy_1_signal = self.strategy_1_calculate(stock_price, futures_price, self.strategy_1_status)
                

                print(f'[{os.getpid()}] --- YF ---- Strategy.handle_market_data - SIGNAL STRATEGY = ', strategy_1_signal)

                if strategy_1_signal == 1:
                    order_object_1 = SingleStockOrder( ticker_stock, '2024-06-25', time.asctime(time.localtime(time.time())))
                    # set .direction, .type, .size
                    order_object_1.direction = 'BUY'
                    order_object_1.type = 'MO'
                    order_object_1.size = 1

                    order_object_2 = SingleStockOrder( ticker_futures, '2024-06-25', time.asctime(time.localtime(time.time())))
                    # set .direction, .type, .size
                    order_object_2.direction = 'SELL'
                    order_object_2.type = 'MO'
                    order_object_2.size = 1

                    # put the order to the queue
                    self.strategy_2_platform_order_q.put([self.getStratID(), order_object_1])
                    self.strategy_2_platform_order_q.put([self.getStratID(), order_object_2])


                    # long 100 stock, and short 100 futures
                    # self.strategy_2_platform_order_q.put([self.getStratID(), SingleStockOrder(
                    #                 orderID = int(time.time())
                    #                 , ticker = ticker_stock
                    #                 , date = '2024-06-25'
                    #                 , submissionTime = time.asctime(time.localtime(time.time()))
                    #                 , direction = 'BUY'
                    #                 , type = 'MO'
                    #                 , size = 1)])
                    # self.strategy_2_platform_order_q.put([self.getStratID(), SingleStockOrder(orderID = int(time.time())
                    #                 , ticker = ticker_futures
                    #                 , date = '2024-06-25'
                    #                 , submissionTime = time.asctime(time.localtime(time.time()))
                    #                 , direction = 'SELL'
                    #                 , type = 'MO'
                    #                 , size = 1)])
                    self.strategy_1_status = 1
                elif strategy_1_signal == -1:
                    # short 100 stock, and long 100 futures

                    order_object_1 = SingleStockOrder( ticker_stock, '2024-06-25', time.asctime(time.localtime(time.time())))
                    # set .direction, .type, .size
                    order_object_1.direction = 'SELL'
                    order_object_1.type = 'MO'
                    order_object_1.size = 1


                    order_object_2 = SingleStockOrder( ticker_futures, '2024-06-25', time.asctime(time.localtime(time.time())))
                    # set .direction, .type, .size
                    order_object_2.direction = 'BUY'
                    order_object_2.type = 'MO'
                    order_object_2.size = 1

                    # put the order to the queue
                    self.strategy_2_platform_order_q.put([self.getStratID(), order_object_1])
                    self.strategy_2_platform_order_q.put([self.getStratID(), order_object_2])


                    # self.strategy_2_platform_order_q.put([self.getStratID(), SingleStockOrder(orderID = int(time.time())
                    #                 , ticker = ticker_stock
                    #                 , date = '2024-06-25'
                    #                 , submissionTime = time.asctime(time.localtime(time.time()))
                    #                 , direction = 'SELL'
                    #                 , type = 'MO'
                    #                 , size = 1)])
                    # self.strategy_2_platform_order_q.put([self.getStratID(), SingleStockOrder(orderID = int(time.time())
                    #                 , ticker = ticker_futures
                    #                 , date = '2024-06-25'
                    #                 , submissionTime = time.asctime(time.localtime(time.time()))
                    #                 , direction = 'BUY'
                    #                 , type = 'MO'
                    #                 , size = 1)])
                    self.strategy_1_status = -1
                elif strategy_1_signal == 0 and self.strategy_1_status == 1:
                    # short 100 stock, and long 100 futures

                    order_object_1 = SingleStockOrder( ticker_stock, '2024-06-25', time.asctime(time.localtime(time.time())))
                    # set .direction, .type, .size
                    order_object_1.direction = 'SELL'
                    order_object_1.type = 'MO'
                    order_object_1.size = 1


                    order_object_2 = SingleStockOrder( ticker_futures, '2024-06-25', time.asctime(time.localtime(time.time())))
                    # set .direction, .type, .size
                    order_object_2.direction = 'BUY'
                    order_object_2.type = 'MO'
                    order_object_2.size = 1

                    # put the order to the queue
                    self.strategy_2_platform_order_q.put([self.getStratID(), order_object_1])
                    self.strategy_2_platform_order_q.put([self.getStratID(), order_object_2])

                    # self.strategy_2_platform_order_q.put([self.getStratID(), SingleStockOrder(orderID = int(time.time())
                    #                 , ticker = ticker_stock
                    #                 , date = '2024-06-25'
                    #                 , submissionTime = time.asctime(time.localtime(time.time()))
                    #                 , direction = 'SELL'
                    #                 , type = 'MO'
                    #                 , size = 1)])
                    # self.strategy_2_platform_order_q.put([self.getStratID(), SingleStockOrder(orderID = int(time.time())
                    #                 , ticker = ticker_futures
                    #                 , date = '2024-06-25'
                    #                 , submissionTime = time.asctime(time.localtime(time.time()))
                    #                 , direction = 'BUY'
                    #                 , type = 'MO'
                    #                 , size = 1)])
                    self.strategy_1_status = 0
                elif strategy_1_signal == 0 and self.strategy_1_status == -1:
                    # long 100 stock, and short 100 futures

                    order_object_1 = SingleStockOrder( ticker_stock, '2024-06-25', time.asctime(time.localtime(time.time())))
                    # set .direction, .type, .size
                    order_object_1.direction = 'BUY'
                    order_object_1.type = 'MO'
                    order_object_1.size = 1


                    order_object_2 = SingleStockOrder( ticker_futures, '2024-06-25', time.asctime(time.localtime(time.time())))
                    # set .direction, .type, .size
                    order_object_2.direction = 'SELL'
                    order_object_2.type = 'MO'
                    order_object_2.size = 1

                    # put the order to the queue
                    self.strategy_2_platform_order_q.put([self.getStratID(), order_object_1])
                    self.strategy_2_platform_order_q.put([self.getStratID(), order_object_2])

                    # self.strategy_2_platform_order_q.put([self.getStratID(), SingleStockOrder(orderID = int(time.time())
                    #                  , ticker = ticker_stock
                    #                  , date = '2024-06-25'
                    #                  , submissionTime = time.asctime(time.localtime(time.time()))
                    #                  , direction = 'BUY'
                    #                  , type = 'MO'
                    #                  , size = 1)])
                    # self.strategy_2_platform_order_q.put([self.getStratID(), SingleStockOrder(orderID = int(time.time())
                    #                  , ticker = ticker_futures
                    #                  , date = '2024-06-25'
                    #                  , submissionTime = time.asctime(time.localtime(time.time()))
                    #                  , direction = 'SELL'
                    #                  , type = 'MO'
                    #                  , size = 1)])
                    self.strategy_1_status = 0
                
            else:
                return None
                    
            