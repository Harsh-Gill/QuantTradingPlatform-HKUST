from multiprocessing import Process, Queue

from exchangeSimulator import ExchangeSimulator
from marketDataService import MarketDataService
from quantTradingPlatform import TradingPlatform
import os

if __name__ == '__main__':
    # get all .txt file and delete
    for file in os.listdir():
        if file.endswith('.txt'):
            os.remove(file)


    ###########################################################################
    # Define instruments to be subscribed
    ###########################################################################
    futures = ['IPF1', 'QLF1', 'RLF1', 'GLF1', 'NLF1', 'NEF1', 'HSF1', 'HCF1', 'NYF1', 'DBF1']
    stocks = ['3035', '3374', '6443', '2392', '5347', '3264', '2618', '2498', '0050', '2610']

    ###########################################################################
    # Define all components
    ###########################################################################
    marketData_2_exchSim_q = Queue()
    marketData_2_platform_q = Queue()

    platform_2_exchSim_order_q = Queue()
    exchSim_2_platform_execution_q = Queue()

    platform_2_strategy_md_q = Queue()
    strategy_2_platform_order_q = Queue()
    platform_2_strategy_execution_q = Queue()

    ###########################################################################
    # Start processes
    ###########################################################################
    Process(name='md', target=MarketDataService,
            args=(marketData_2_exchSim_q, marketData_2_platform_q, futures, stocks)).start()
    Process(name='sim', target=ExchangeSimulator,
            args=(marketData_2_exchSim_q, platform_2_exchSim_order_q, exchSim_2_platform_execution_q,)).start()
    Process(name='platform', target=TradingPlatform,
            args=(
                marketData_2_platform_q, strategy_2_platform_order_q,platform_2_exchSim_order_q, exchSim_2_platform_execution_q,)).start()

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
