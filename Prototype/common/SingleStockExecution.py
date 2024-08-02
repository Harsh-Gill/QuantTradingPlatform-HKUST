# -*- coding: utf-8 -*-
"""
Created on Sat Jul 3 07:11:28 2019

@author: hongs
"""

import pandas as pd
import random

class SingleStockExecution():

    def __init__(self,
                 ticker,
                 date,
                 timeStamp,
                 direction,
                 execID,
                 orderID,
                 size=0,  # executed size
                 price=None,
                 order_status=None,
                 rest_size=0):
        self.execID = execID
        self.orderID = orderID
        self.ticker = ticker
        self.date = date
        self.timeStamp = timeStamp
        self.direction = direction  # "BI", "SI", "ACK", "Cancel Successfully But Partially Filled"
        # "Cancel Failed fully filled", "Cancel Successfully not filled"
        self.price = price
        self.size = size
        self.comm = 0  # commission for this transaction

        self.order_status = order_status  # "Filled" "Partially Filled" "not filled"
        self.rest_size = rest_size

    def outputAsArray(self):
        output = []
        output.append(self.date)
        output.append(self.ticker)
        output.append(self.timeStamp)
        output.append(self.execID)
        output.append(self.orderID)
        output.append(self.direction)
        output.append(self.price)
        output.append(self.size)
        output.append(self.comm)

        return output



