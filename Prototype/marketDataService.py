import os
import time
from multiprocessing import Process, Queue

import pandas as pd

from common.OrderBookSnapshot_FiveLevels import OrderBookSnapshot_FiveLevels

START_DATE = pd.Timestamp('2024-06-28')
WAIT_AFTER_RUNNING = 10


class MarketDataService:
    def __init__(self, marketData_2_exchSim_q, marketData_2_platform_q, futures=None, stocks=None,
                 chunksize=1000):
        print(f"[{os.getpid()}] <<<<< call MarketDataService.init")
        if futures is None:
            futures = []
        if stocks is None:
            stocks = []

        self.INIT_TS = START_DATE.timestamp() + 32400
        self.subscribed_futures = futures
        self.subscribed_stocks = stocks

        self._data_path = './processedData_2024'
        self._support_instruments = self._get_instruments()
        self._months = ["202406"]
        self._chunksize = chunksize
        self._start_time = time.time()

        # Validate if all subscribed instruments are supported
        for _stock in self.subscribed_stocks:
            if _stock not in self._support_instruments['stocks']:
                raise ValueError(f"No data for stock {_stock}")
        for _futures in self.subscribed_futures:
            if _futures not in self._support_instruments['futures']:
                raise ValueError(f"No data for futures {futures}")

        self.produce_multi_insts_market_data(marketData_2_exchSim_q, marketData_2_platform_q)
        # self.produce_single_inst_market_data('futures', 'DBF1', marketData_2_exchSim_q, marketData_2_platform_q)
        # self.produce_single_inst_market_data('stocks', '2618', marketData_2_exchSim_q, marketData_2_platform_q)

    def _get_instruments(self):
        # Get the list of supported instruments from the data directory
        futures_instruments = list(
            set([f.split('_')[0] for f in os.listdir(os.path.join(self._data_path, 'futuresQuotes')) if
                 f.endswith('.csv.gz')]))
        stocks_instruments = list(set([f.split('_')[0] for f in os.listdir(os.path.join(self._data_path, 'stocks')) if
                                       f.endswith('.csv.gz')]))
        return {'futures': futures_instruments, 'stocks': stocks_instruments}

    @staticmethod
    def _convert_time(time_int):
        hour = time_int // 1e7
        minute = (time_int % 1e7) // 1e5
        second = (time_int % 1e5) // 1e3
        microsecond = time_int % 1e3 * 1e3
        return pd.Timedelta(hours=hour, minutes=minute, seconds=second, microseconds=microsecond)

    @staticmethod
    def _wait_until(timestamp):
        while True:
            now = time.time()
            if now >= timestamp:
                break
            sleep_time = timestamp - now
            if sleep_time > 0.1:
                time.sleep(sleep_time - 0.1)
            else:
                pass

    def _process_chunk(self, chunk):
        chunk = chunk[chunk['time'] >= 9e7]
        chunk = chunk[chunk['date'] >= START_DATE]
        if chunk.empty:
            return chunk
        chunk = chunk.copy()
        chunk.loc[:, 'time'] = chunk['date'] + chunk['time'].apply(self._convert_time)
        chunk.rename(columns=lambda x: x.replace('BP', 'bidPrice').replace('BV', 'bidSize')
                     .replace('SP', 'askPrice').replace('SV', 'askSize'), inplace=True)
        return chunk

    def read_data_by_chunk(self, category, instrument, month, chunksize=5000):
        """
        Read data in chunks and process each chunk
        """
        file_name = f"{instrument}_md_{month}_{month}.csv.gz"
        file_path = os.path.join(self._data_path, 'futuresQuotes' if category == 'futures' else 'stocks', file_name)
        first_chunk_flg = True
        chunks = pd.read_csv(file_path, compression='gzip', parse_dates=['date'], chunksize=chunksize)
        for chunk in chunks:
            chunk = self._process_chunk(chunk)
            while chunk.empty:
                try:
                    chunk = next(chunks)
                    chunk = self._process_chunk(chunk)
                except StopIteration:
                    return
            yield first_chunk_flg, chunk
            first_chunk_flg = False

    def produce_single_inst_market_data(self, category, instrument, marketData_2_exchSim_q, marketData_2_platform_q):
        print(f"[{os.getpid()}] <<<<< Starting processing for instrument: {instrument}")
        for month in self._months:
            for first_chunk_flg, chunk in self.read_data_by_chunk(category, instrument, month, self._chunksize):
                print(f"[{os.getpid()}] <<<<< Processing chunk for instrument: {instrument}, month: {month}")
                if first_chunk_flg:
                    self._wait_until(self._start_time + WAIT_AFTER_RUNNING)
                for i, row in chunk.iterrows():
                    bidPrice = [row[f'bidPrice{i}'] for i in range(1, 6)]
                    askPrice = [row[f'askPrice{i}'] for i in range(1, 6)]
                    bidSize = [row[f'bidSize{i}'] for i in range(1, 6)]
                    askSize = [row[f'askSize{i}'] for i in range(1, 6)]
                    ts = row['time'].timestamp()
                    obs = OrderBookSnapshot_FiveLevels(instrument, row['date'], row['time'], bidPrice, askPrice,
                                                       bidSize, askSize)
                    self._wait_until(self._start_time + WAIT_AFTER_RUNNING + ts - self.INIT_TS)
                    # print("put quote:", obs.outputAsDataFrame())
                    marketData_2_exchSim_q.put(obs)
                    marketData_2_platform_q.put(obs)
                    # print(f"PID: {os.getpid()} Instrument id : {instrument} Time: {ts} \n{obs.outputAsDataFrame()}")

                    # save to logs called market_data_output.txt
                    with open('market_data_output.txt', 'a') as f:
                        # Increase the display width to show more columns
                        pd.set_option('display.max_columns', 100)
                        f.write(obs.outputAsDataFrame().to_string(index=False) + '\n')

    def produce_multi_insts_market_data(self, marketData_2_exchSim_q, marketData_2_platform_q):
        """
        Create and start subprocesses for each instrument
        """
        processes = []
        for instrument in self.subscribed_futures:
            p = Process(target=self.produce_single_inst_market_data,
                        args=('futures', instrument, marketData_2_exchSim_q, marketData_2_platform_q))
            p.start()
            processes.append(p)
        for instrument in self.subscribed_stocks:
            p = Process(target=self.produce_single_inst_market_data,
                        args=('stock', instrument, marketData_2_exchSim_q, marketData_2_platform_q))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()


if __name__ == '__main__':
    # futures = []
    futures = ['QLF1', 'GLF1', 'NYF1', 'HSF1', 'HCF1', 'NEF1', 'DBF1', 'IPF1', 'NLF1', 'RLF1']
    stocks = ['2610', '0050', '6443', '2498', '2618', '3374', '3035', '5347', '3264', '2392']
    marketData_2_exchSim_q = Queue()
    marketData_2_platform_q = Queue()
    DMS = MarketDataService(marketData_2_exchSim_q, marketData_2_platform_q, futures, stocks)
