import pandas as pd
from binance.client import Client
from dateutil import parser
from SRC.CORE.ApiClient import ApiClient
from SRC.CORE._CONSTANTS import SYMBOL_TRADING_KEY, START_CACHE_PROCESS_KEY, END_CACHE_PROCESS_KEY
from SRC.CORE._FUNCTIONS import PAIRS
from SRC.CORE.utils import get_item_from_list_dict

binance_api_key = '[REDACTED]'    #Enter your own API-key here
binance_api_secret = '[REDACTED]' #Enter your own API-secret here


class BinanceApiClient(ApiClient):
    def __init__(self, symbol, process_symbol, kline_size):
        self.binance_client = Client(api_key=binance_api_key, api_secret=binance_api_secret)
        self.api_key = binance_api_key
        self.api_secret = binance_api_secret
        self.symbol = symbol
        self.process_symbol = process_symbol
        self.kline_size = kline_size.lower()


    def get_historical_klines(self, start_point, end_point):
        klines = self.binance_client.get_historical_klines(self.symbol, self.kline_size, start_point.strftime("%d %b %Y %H:%M:%S"), end_point.strftime("%d %b %Y %H:%M:%S"))

        return klines

    def get_end_point(self):
        end_point = pd.to_datetime(self.binance_client.get_klines(symbol=self.symbol, interval=self.kline_size)[-1][0], unit='ms')

        return end_point

    def minutes_of_new_data(self, data):
        from SRC.CORE._CONSTANTS import CACHED__START_DATE

        if len(data) > 0:
            start_point = parser.parse(data["timestamp"].iloc[-1])
        else:
            start_point = CACHED__START_DATE

        end_point = self.get_end_point()

        pair_config = get_item_from_list_dict(PAIRS(), SYMBOL_TRADING_KEY, self.process_symbol)
        if START_CACHE_PROCESS_KEY in pair_config:
            start_point = pair_config[START_CACHE_PROCESS_KEY]

        if END_CACHE_PROCESS_KEY in pair_config:
            end_point = pair_config[END_CACHE_PROCESS_KEY]

        return start_point, end_point