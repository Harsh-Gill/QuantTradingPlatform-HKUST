import os
import time
from multiprocessing import Process, Queue

import pandas as pd

from common.OrderBookSnapshot_FiveLevels import OrderBookSnapshot_FiveLevels

FILTER_BEFORE_9 = True

INIT_TS = 1711962000
if not FILTER_BEFORE_9:
    INIT_TS -= 800


class SubscriptionManager:
    def __init__(self, futures=None, stocks=None):
        if futures is None:
            futures = []
        if stocks is None:
            stocks = []
        self.futures = futures
        self.stocks = stocks
        self.queues = {
            'futures': {instrument: Queue() for instrument in futures},
            'stock': {instrument: Queue() for instrument in stocks}
        }

    def get_queue(self, category, instrument):
        return self.queues.get(category, {}).get(instrument)


class MarketDataService:
    def __init__(self, market_data_subscription_manager: SubscriptionManager, chunksize=1000):
        print(f"[{os.getpid()}] <<<<< call MarketDataService.init")
        self._data_path = '.\processedData_2024'
        self._support_instruments = self._get_instruments()
        self._months = ["202404", "202405", "202406"]
        self._chunksize = chunksize
        self._start_time = time.time()

        self.subscription_manager = market_data_subscription_manager
        # self.produce_single_inst_market_data('futures', 'HSF1')

        # Validate if all subscribed instruments are supported
        for stock in self.subscription_manager.stocks:
            if stock not in self._support_instruments['stocks']:
                raise ValueError(f"No data for stock {stock}")
        for futures in self.subscription_manager.futures:
            if futures not in self._support_instruments['futures']:
                raise ValueError(f"No data for futures {futures}")

        self.produce_multi_insts_market_data()

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

    def _process_chunk(self, chunk, filter_before_9):
        if filter_before_9:
            chunk = chunk[chunk['time'] >= 9e7]
        if chunk.empty:
            return chunk
        chunk = chunk.copy()
        chunk.loc[:, 'time'] = chunk['date'] + chunk['time'].apply(self._convert_time)
        chunk.rename(columns=lambda x: x.replace('BP', 'bidPrice').replace('BV', 'bidSize')
                     .replace('SP', 'askPrice').replace('SV', 'askSize'), inplace=True)
        chunk = chunk.iloc[1:] if not chunk.empty else chunk
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
            chunk = self._process_chunk(chunk, FILTER_BEFORE_9)

            while chunk.empty:
                try:
                    chunk = next(chunks)
                    chunk = self._process_chunk(chunk, FILTER_BEFORE_9)
                except StopIteration:
                    return

            yield first_chunk_flg, chunk
            first_chunk_flg = False

    def produce_single_inst_market_data(self, category, instrument):
        print(f"[{os.getpid()}] <<<<< Starting processing for instrument: {instrument}")
        for month in self._months:
            for first_chunk_flg, chunk in self.read_data_by_chunk(category, instrument, month, self._chunksize):
                print(f"[{os.getpid()}] <<<<< Processing chunk for instrument: {instrument}, month: {month}")
                if first_chunk_flg:
                    self._wait_until(self._start_time + 10)
                for i, row in chunk.iterrows():
                    bidPrice = [row[f'bidPrice{i}'] for i in range(1, 6)]
                    askPrice = [row[f'askPrice{i}'] for i in range(1, 6)]
                    bidSize = [row[f'bidSize{i}'] for i in range(1, 6)]
                    askSize = [row[f'askSize{i}'] for i in range(1, 6)]
                    ts = row['time'].timestamp()

                    self._wait_until(self._start_time + 10 + ts - INIT_TS)
                    print('Put quote:', instrument, row['time'])
                    self.subscription_manager.get_queue(category, instrument).put(
                        OrderBookSnapshot_FiveLevels(instrument, row['date'], row['time'], bidPrice, askPrice, bidSize,
                                                     askSize))

    def produce_multi_insts_market_data(self):
        """
        Create and start subprocesses for each instrument
        """
        processes = []
        for instrument in self.subscription_manager.futures:
            p = Process(target=self.produce_single_inst_market_data, args=('futures', instrument))
            p.start()
            processes.append(p)
        for instrument in self.subscription_manager.stocks:
            p = Process(target=self.produce_single_inst_market_data, args=('stock', instrument))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()


if __name__ == '__main__':
    sm = SubscriptionManager(
        futures=['QLF1', 'GLF1', 'NYF1', 'HSF1', 'HCF1', 'NEF1', 'DBF1', 'IPF1', 'NLF1', 'RLF1'],
        stocks=['2610', '0050', '6443', '2498', '2618', '3374', '3035', '5347', '3264', '2392'])
    DMS = MarketDataService(sm, )
