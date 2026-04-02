import functools
import traceback
import hashlib
import hmac
import json
import multiprocessing
import os
import threading
import subprocess
import time
from collections import Counter, defaultdict
from functools import lru_cache
from async_lru import alru_cache
import aiohttp
import binance.exceptions
import pandas as pd
import plotly.graph_objects as go
from IPython.core.display_functions import clear_output
from binance.client import Client as BinanceClient
from binance.enums import SIDE_BUY, ORDER_TYPE_MARKET, SIDE_SELL, FUTURE_ORDER_TYPE_MARKET
from dateutil import parser
from SRC.LIBRARIES.new_utils import is_close_or_lower, is_close_or_higher
from SRC.CORE._CONSTANTS import _SYMBOL_JOIN, EXCHANGE_MP_INFO_FILE_PATH, EXCHANGE_INFO_FILE_PATH, SYMBOLS_INFO_FILE_PATH, MARGIN_ACCOUNT_INFO_FILE_PATH, _DISCRETIZATION, \
    _SYMBOL_MARGIN_LOAN_WATCHER_FILE_PATH, EXCHANGE_INFO_FUTURES_FILE_PATH, USE_PROXY_CLIENT, project_root_dir, _IS_BINANCE_PROD, _LONG
from SRC.CORE._CONSTANTS import _KIEV_TIMESTAMP, LEVERAGE
from SRC.CORE._CONSTANTS import _SYMBOL_TRADINGVIEW_RECOMMENDATION_DF_FILE_PATH
from SRC.CORE._CONSTANTS import _UTC_TIMESTAMP, _SYMBOL
from SRC.CORE.debug_utils import log_context, printmd, produce_measure, is_cloud, get_execution_caller_present, decor_log_mock_calls, DEBUG, print_call_stack, CONSOLE_SPLITTED, IS_DEBUG, EXCEPTION_SPLITTED, ERROR, EXCEPTION, NOTICE_SPLITTED, \
    is_autotrading
from SRC.CORE.file_utils import remove_all_files_from_folder
from SRC.CORE.utils import datetime_h_m__d_m, datetime_h_m_s__d_m, hashabledict, datetime_Y_m_d__h_m_s, _float_5
from SRC.LIBRARIES.new_data_utils import candelify, featurize_gradient, featurize_gradient_extremums, linear_interpolate
from SRC.LIBRARIES.new_utils import _symbol_join, merge_dicts, merge_dict_s, func_multi_process
from SRC.LIBRARIES.new_utils import find_first_list_item, timed_cache, get_round_order, multiprocess_cache, json_file_cache, tryall_delegate, floor, check_env_true, func_multi_thread_executor, is_close_to_zero, run_safety_interrupter
from SRC.LIBRARIES.time_utils import TIME_DELTA, localize_kiev_tz, round_down_to_nearest_sec
from SRC.LIBRARIES.time_utils import block_until_next
from SRC.LIBRARIES.time_utils import get_datetime_splitters
from SRC.LIBRARIES.time_utils import kiev_now, utc_now
from binance.enums import FUTURE_ORDER_TYPE_STOP_MARKET, FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET
from SRC.CORE._CONSTANTS import _BACKTEST
from SRC.CORE._CONSTANTS import _MARGIN, _FUTURES, _LONG, _SHORT
from SRC.CORE.utils import _float_5
from SRC.LIBRARIES.new_utils import print_log_trades

try:
    from binance.streams import BinanceSocketManager
    from binance.streams import AsyncClient
except:
    from binance import BinanceSocketManager, Client as BinanceClient, FUTURE_ORDER_TYPE_MARKET, FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET, FUTURE_ORDER_TYPE_STOP_MARKET
    from binance import AsyncClient

CLOSE_POSITION_FEE_EXTRACTOR_RATIO = 1.00075


try:
    from tradingview_ta import TA_Handler, Interval, Exchange, get_multiple_analysis
    import tradingview_ta
except:
    subprocess.run(["pip install tradingview_ta"], shell=True)
    from tradingview_ta import TA_Handler, Interval, Exchange
    import tradingview_ta

try:
    from tabulate import tabulate
except:
    subprocess.run(["pip install tabulate"], shell=True)
    from tabulate import tabulate

try:
    import mplfinance as mpf
except:
    subprocess.run(["pip install mplfinance"], shell=True)
    import mplfinance as mpf


def ensure_binance_prod_context(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not check_env_true(_IS_BINANCE_PROD):
            execution_caller = get_execution_caller_present()
            raise RuntimeError(f"!! DONT USE METHOD [{execution_caller}] | BINANCE PROD FLAG [--binance_prod] IS NOT SET !!")

        result = func(*args, **kwargs)

        return result
    return wrapper


def produce_exit_market_order_params(algo_order):
    new_client_order_id = algo_order['clientAlgoId']
    symbol = algo_order['symbol']
    quantity = float(algo_order['quantity'])
    type = FUTURE_ORDER_TYPE_MARKET
    side = algo_order['side']
    positionSide = algo_order['positionSide']
    price = float(algo_order['triggerPrice'])

    exit_market_order_params = {
        'newClientOrderId': new_client_order_id,
        "symbol": symbol,
        "quantity": str(quantity),
        "type": type,
        "side": side,
        "positionSide": positionSide
    }

    return exit_market_order_params


class MyBinanceClient(BinanceClient):
    def __init__(self, api_key=None, api_secret=None):
        super().__init__(api_key, api_secret)

        self.client = super()

    @ensure_binance_prod_context
    @decor_log_mock_calls(name='BINANCE PROD | create_margin_order', depth=4)
    def create_margin_order(self, **params):
        return self.client.create_margin_order(**params)

    @ensure_binance_prod_context
    @decor_log_mock_calls(name='BINANCE PROD | get_margin_order', depth=4)
    def get_margin_order(self, **params):
        margin_order = self.client.get_margin_order(**params)

        return margin_order

    @decor_log_mock_calls(name='BINANCE PROD | get_margin_account', depth=4, shorten_return=200)
    def get_margin_account(self, **params):
        margin_account = self.client.get_margin_account(**params)

        return margin_account

    @ensure_binance_prod_context
    @decor_log_mock_calls(name='BINANCE PROD | futures_create_order', depth=4)
    def futures_create_order(self, **params):
        return self.client.futures_create_order(**params)

    @ensure_binance_prod_context
    @decor_log_mock_calls(name='BINANCE PROD | futures_get_order', depth=4)
    def futures_get_order(self, **params):
        futures_order  = self.client.futures_get_order(**params)

        return futures_order

    @decor_log_mock_calls(name='BINANCE PROD | futures_account', depth=4, shorten_return=200)
    def futures_account(self, **params):
        futures_account = self.client.futures_account(**params)

        return futures_account

    def futures_account_balance(self, **params):
        futures_account_balance = self.client.futures_account_balance(**params)

        return futures_account_balance

    def futures_account_trades(self, **params):
        futures_account_trades = self.client.futures_account_trades(**params)

        return futures_account_trades

    def futures_get_all_orders(self, **params):
        futures_all_orders = self.client.futures_get_all_orders(**params)

        return futures_all_orders

    def futures_get_open_orders(self, **params):
        futures_open_orders = self.client.futures_get_open_orders(**params)

        return futures_open_orders

    def futures_asset_balance(self, **params):
        stable_asset = params['stable_asset']
        account_balance = self.futures_account_balance()
        stable_asset_balance = [bal for bal in account_balance if bal['asset'] == stable_asset][0]
        stable_asset_available_balance = stable_asset_balance['availableBalance']

        return float(stable_asset_available_balance)

    def transfer__SPOT__ISOLATED_MARGIN(self, asset, amount, toSymbol):
        self.universal_transfer(type='MAIN_MARGIN', asset=asset, amount=amount, toSymbol=toSymbol)
        self.universal_transfer(type='MARGIN_ISOLATEDMARGIN', asset=asset, amount=amount, toSymbol=toSymbol)

    def transfer__ISOLATED_MARGIN__SPOT(self, asset, amount, fromSymbol):
        self.universal_transfer(type='ISOLATEDMARGIN_MARGIN', asset=asset, amount=amount, fromSymbol=fromSymbol)
        self.universal_transfer(type='MARGIN_MAIN', asset=asset, amount=amount, fromSymbol=fromSymbol)


def produce_binance_client():
    client = MyBinanceClient()

    return client

@lru_cache(maxsize=None)
def produce_binance_client_singleton_cached():
    client = produce_binance_client()

    return client


def produce_binance_client_singleton():
    client = produce_binance_client_singleton_cached()

    return client


@alru_cache(maxsize=None)
async def async_produce_binance_async_client_singleton():
    async_client = await async_produce_binance_async_client()

    return async_client


@lru_cache(maxsize=None)
def produce_binance_client_singleton_wrapper(print_traces=False):
    if print_traces and IS_DEBUG():
        print_call_stack()

    client = produce_binance_client()

    return client


@lru_cache(maxsize=None)
def get_quantity_round_order(symbol):
    symbol_info = get_symbol_info_file_cached(symbol)
    price_tick_size, quantity_step_size, min_stable_allowed = get_tick_step_info_spot(hashabledict(symbol_info))
    quantity_round_order = get_round_order(quantity_step_size)

    return quantity_round_order


@lru_cache(maxsize=None)
def get_price_round_order(symbol):
    symbol_info = get_symbol_info_file_cached(symbol)
    price_tick_size, quantity_step_size, min_stable_allowed = get_tick_step_info_spot(hashabledict(symbol_info))
    price_round_order = get_round_order(price_tick_size)

    return price_round_order


def get_margin_assets_info(symbol_info):
    margin_account = get_margin_account()
    stable_asset = find_first_list_item(margin_account['userAssets'], 'asset', symbol_info['quoteAsset'])
    coin_asset = find_first_list_item(margin_account['userAssets'], 'asset', symbol_info['baseAsset'])

    return stable_asset, coin_asset


def get_futures_assets_info(symbol_info):
    futures_account = get_futures_account()
    stable_asset = find_first_list_item(futures_account['assets'], 'asset', symbol_info['quoteAsset'])
    coin_asset = find_first_list_item(futures_account['assets'], 'asset', symbol_info['baseAsset'])

    return stable_asset, coin_asset


def get_symbol_info(symbol):
    client = produce_binance_client_singleton()

    sym_join = _symbol_join(symbol)
    symbol_info = client.get_symbol_info(sym_join)

    return symbol_info


@json_file_cache(f"{SYMBOLS_INFO_FILE_PATH()}", ttl=60 * 60)
def get_symbol_info_file_cached(symbol):
    symbol_info = get_symbol_info(symbol)

    return symbol_info


def get_exchange_symbol_info(symbol, exchange_info=None):
    if exchange_info is None:
        exchange_info = get_exchange_info_file_cached()

    symbol_info = find_first_list_item(exchange_info['symbols'], 'symbol', symbol)
    if symbol_info is None:
        return hashabledict(get_symbol_info_file_cached(symbol))

    return hashabledict(symbol_info)


def get_futures_exchange_symbol_info(symbol, exchange_info=None):
    if exchange_info is None:
        exchange_info = get_futures_exchange_info_file_cached()

    symbol_info = find_first_list_item(exchange_info['symbols'], 'symbol', symbol)
    if symbol_info is None:
        return hashabledict(get_symbol_info_file_cached(symbol))

    return hashabledict(symbol_info)


def get_margin_account():
    client = produce_binance_client_singleton()
    margin_account = client.get_margin_account()

    return margin_account


def get_futures_account():
    client = produce_binance_client_singleton()
    futures_account = client.futures_account()

    return futures_account


@json_file_cache(f"{MARGIN_ACCOUNT_INFO_FILE_PATH()}", ttl=60 * 60, lock_timeout_secs=20)
def get_margin_account_cached():
    return get_margin_account()


def get_exchange_info():
    client = produce_binance_client_singleton()

    return client.get_exchange_info()


@json_file_cache(f"{EXCHANGE_INFO_FILE_PATH()}", ttl=60 * 30, lock_timeout_secs=20)
def get_exchange_info_file_cached():
    return get_exchange_info()


# @timed_cache(ttl=60 * 15)
@multiprocess_cache(f"{EXCHANGE_MP_INFO_FILE_PATH()}", ttl=60 * 15)
def get_exchange_info_time_cached():
    return get_exchange_info_file_cached()


def get_futures_exchange_info():
    client = produce_binance_client_singleton()

    return client.futures_exchange_info()


@json_file_cache(f"{EXCHANGE_INFO_FUTURES_FILE_PATH()}", ttl=60 * 30, lock_timeout_secs=20)
def get_futures_exchange_info_file_cached():
    return get_futures_exchange_info()


@timed_cache(ttl=60 * 15)
def get_futures_exchange_info_time_cached():
    return get_futures_exchange_info_file_cached()


@timed_cache(ttl=60 * 60)
def get_tick_step_info_spot(symbol_info):
    price_tick_size = float([k for k in symbol_info['filters'] if k['filterType'] == 'PRICE_FILTER'][0]['tickSize'])
    quantity_step_size = float([k for k in symbol_info['filters'] if k['filterType'] == 'LOT_SIZE'][0]['stepSize'])
    min_stable_allowed = float([k for k in symbol_info['filters'] if k['filterType'] == 'NOTIONAL'][0]['minNotional'])

    return price_tick_size, quantity_step_size, min_stable_allowed


@timed_cache(ttl=60 * 60)
def get_tick_step_info_futures(symbol_info):
    price_tick_size = float([k for k in symbol_info['filters'] if k['filterType'] == 'PRICE_FILTER'][0]['tickSize'])
    quantity_step_size = float([k for k in symbol_info['filters'] if k['filterType'] == 'LOT_SIZE'][0]['stepSize'])
    min_stable_allowed = float([k for k in symbol_info['filters'] if k['filterType'] == 'MIN_NOTIONAL'][0]['notional'])

    return price_tick_size, quantity_step_size, min_stable_allowed


def get_market_symbol_ticker_current_price(symbol, market_type):
    if '-' in market_type:
        market_type = market_type.split('-')[1]

    if market_type == _FUTURES:
        return get_futures_symbol_ticker_current_price(symbol)
    elif market_type == _MARGIN:
        return  get_margin_symbol_ticker_current_price(symbol)
    elif market_type == _BACKTEST:
        return  get_margin_symbol_ticker_current_price(symbol)
    else:
        raise AssertionError(f"!!! NO MARKET TYPE: {market_type} ALLOWED !!!")


def get_margin_symbol_ticker_current_price(symbol):
    client = produce_binance_client_singleton()

    current_price = float(client.get_symbol_ticker(symbol=symbol)['price'])

    return current_price


def get_futures_symbol_ticker_current_price(symbol):
    client = produce_binance_client_singleton()

    current_price = float(client.futures_symbol_ticker(symbol=symbol)['price'])

    return current_price


def get_margin_isolated_total_net_USDT(client, symbol, price):
    coin_stable_asset = client.get_isolated_margin_account(symbols=symbol)['assets'][0]

    quote_asset = coin_stable_asset['quoteAsset']
    base_asset = coin_stable_asset['baseAsset']
    total_net_USDT = (
            float(quote_asset['netAsset']) +
            float(base_asset['netAsset']) * price
    )

    return total_net_USDT


def calculate_reduceonly_order_exit_position_metrics(entry_order, exit_reduceonly_order, grouped_trades):
    reduceonly_exit_order_metrics = calculate_metrics(exit_reduceonly_order, grouped_trades)

    reduceonly_total_qty = reduceonly_exit_order_metrics['total_qty']
    reduceonly_total_fee = reduceonly_exit_order_metrics['total_fee']
    reduceonly_total_pnl = reduceonly_exit_order_metrics['total_pnl']
    reduceonly_update_kiev_dt = reduceonly_exit_order_metrics['update_kiev_dt']

    pos_side = entry_order['positionSide']
    entry_quantity = float(entry_order['executedQty'])

    result_exit_qty_part = entry_quantity / reduceonly_total_qty

    total_qty = reduceonly_total_qty * result_exit_qty_part
    total_fee = reduceonly_total_fee * result_exit_qty_part
    total_pnl = reduceonly_total_pnl * result_exit_qty_part

    return {
        'total_qty': total_qty,
        'total_fee': total_fee,
        'total_pnl': total_pnl,
        'pos_side': pos_side,
        'update_kiev_dt': reduceonly_update_kiev_dt,
    }


def futures_cross_position_information(symbol):
    client = produce_binance_client_singleton()

    position_s = client.futures_position_information(symbol=symbol)

    symbol_position_s = []
    for pos in position_s:
        symbol_position_s.append({
            'symbol': pos['symbol'],
            'positionSide': pos['positionSide'],
            'positionAmt': str(pos['positionAmt']),
        })

    return symbol_position_s


def margin_cross_position_information(symbol):
    cross_margin_position_s = get_cross_margin_positions()
    symbol_position_s = []
    for pos in cross_margin_position_s:
        if pos['symbol'] == symbol:
            symbol_position_s.append(pos)

    return symbol_position_s


def get_cross_margin_positions():
    client = produce_binance_client_singleton()

    quote = "USDT"

    account = client.get_margin_account()
    tickers = client.get_symbol_ticker()
    prices = {t['symbol']: float(t['price']) for t in tickers}

    position_s = []

    for a in account['userAssets']:
        asset = a['asset']
        if asset == quote:
            continue

        net = float(a['netAsset'])

        if abs(net) < 1e-8:
            continue

        symbol = f"{asset}{quote}"
        price = prices.get(symbol)

        if not price:
            continue

        value_usdt = abs(net) * price

        if net > 0:
            positionSide = _LONG
        else:
            positionSide = _SHORT

        position_s.append({
            "symbol": symbol,
            "positionSide": positionSide,
            "positionAmt": str(net)
        })

    return position_s


def calculate_metrics(order, grouped_trades):
    order_id = order['orderId']
    pos_side = order['positionSide']
    update_kiev_dt = localize_kiev_tz(order['updateTime'])

    order_trades = grouped_trades[order_id]
    total_fee = sum(float(t['commission']) for t in order_trades)
    total_pnl = sum(float(t['realizedPnl']) for t in order_trades)
    total_qty = sum(float(t['qty']) for t in order_trades)

    return {
        'total_fee': total_fee,
        'total_pnl': total_pnl,
        'total_qty': total_qty,
        'pos_side': pos_side,
        'update_kiev_dt': update_kiev_dt,
    }


@timed_cache(ttl=3)
def get_futures_transactions_cumulative_balance_movement(client, symbol_join):
    transaction_s = get_futures_closed_transactions(client, symbol_join)
    cum_balance_movement = sum([transaction['balance_movement'] for transaction in transaction_s])
    stable_asset_balance = cum_balance_movement

    return stable_asset_balance


def get_transaction_exit_from_entry_order(entry_order, long_manual_order_s, short_manual_order_s):
    exit_order = None

    position_side = entry_order['positionSide']
    quantity = entry_order['origQty']
    update_time = entry_order['updateTime']

    if entry_order['positionSide'] == _LONG:
        exit_order_ = [o for o in long_manual_order_s if o['positionSide'] == position_side and o['origQty'] == quantity and o['updateTime'] > update_time]
        if len(exit_order_) > 0:
            exit_order = exit_order_[0]
    elif entry_order['positionSide'] == _SHORT:
        exit_order_ = [o for o in short_manual_order_s if o['positionSide'] == position_side and o['origQty'] == quantity and o['updateTime'] > update_time]
        if len(exit_order_) > 0:
            exit_order = exit_order_[0]
    else:
        raise AssertionError(f"NOT ALLOWED [{entry_order['positionSide']}]: {entry_order}")

    return exit_order


def get_transaction_entry_exit_order(grouped_order_s, order_s, long_manual_order_s, short_manual_order_s):
    entry_order = [to for to in grouped_order_s if ('type' in to and to['type'] == FUTURE_ORDER_TYPE_MARKET) or ('orderType' in to and to['orderType'] == FUTURE_ORDER_TYPE_MARKET)][0]
    algo_loss_order_ = [to for to in grouped_order_s if ('type' in to and to['type'] == FUTURE_ORDER_TYPE_STOP_MARKET) or ('orderType' in to and to['orderType'] == FUTURE_ORDER_TYPE_STOP_MARKET)]
    algo_profit_order_ = [to for to in grouped_order_s if ('type' in to and to['type'] == FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET) or ('orderType' in to and to['orderType'] == FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET)]

    exit_algo_order = None
    exit_order = None
    algo_loss_order = None
    algo_profit_order = None

    if len(algo_loss_order_) > 0:
        algo_loss_order = algo_loss_order_[0]

    if len(algo_profit_order_) > 0:
        algo_profit_order = algo_profit_order_[0]

    if algo_loss_order and algo_profit_order:
        if algo_loss_order['algoStatus'] == 'FINISHED':
            loss_order_ = [order for order in order_s if str(order['orderId']) == str(algo_loss_order['actualOrderId'])]
            if len(loss_order_) > 0:
                exit_order = loss_order_[0]
            else:
                exit_order = get_transaction_exit_from_entry_order(entry_order, long_manual_order_s, short_manual_order_s)
            exit_algo_order = algo_loss_order
        elif algo_profit_order['algoStatus'] == 'FINISHED':
            profit_order_ = [order for order in order_s if str(order['orderId']) == str(algo_profit_order['actualOrderId'])]
            if len(profit_order_) > 0:
                exit_order = profit_order_[0]
            else:
                exit_order = get_transaction_exit_from_entry_order(entry_order, long_manual_order_s, short_manual_order_s)
            exit_algo_order = algo_profit_order
        elif algo_loss_order['algoStatus'] == 'CANCELED' and algo_profit_order['algoStatus'] == 'CANCELED':
            start_transaction_time = entry_order['updateTime']
            end_transaction_time = max([algo_loss_order['updateTime'], algo_profit_order['updateTime']])
            if entry_order['positionSide'] == _LONG:
                long_reduceonly_order_ = [order for order in long_manual_order_s if start_transaction_time < order['updateTime'] < end_transaction_time]
                if len(long_reduceonly_order_) > 0:
                    exit_order = long_reduceonly_order_[0]
            elif entry_order['positionSide'] == _SHORT:
                short_reduceonly_order_ = [order for order in short_manual_order_s if start_transaction_time < order['updateTime'] < end_transaction_time]
                if len(short_reduceonly_order_) > 0:
                    exit_order = short_reduceonly_order_[0]
            else:
                raise AssertionError(f"NOT ALLOWED [{entry_order['positionSide']}]: {entry_order}")
    else:
        exit_order = get_transaction_exit_from_entry_order(entry_order, long_manual_order_s, short_manual_order_s)

    return entry_order, exit_order, exit_algo_order


def get_futures_closed_transactions(client, symbol_join):
    start_time = 0
    if 'START_DT_STR' in os.environ:
        start_time = parser.parse(os.environ['START_DT_STR']).timestamp() * 1000

    order_s = client.futures_get_all_orders(symbol=symbol_join)
    algo_order_s = client.futures_get_all_algo_orders(symbol=symbol_join)
    trades = client.futures_account_trades(symbol=symbol_join)

    # order_s = [o for o in order_s if o['updateTime'] > start_time]
    manual_order_s = [order for order in order_s if 'web_' in order['clientOrderId']]
    long_manual_order_s = [order for order in manual_order_s if order['positionSide'] == _LONG]
    short_manual_order_s = [order for order in manual_order_s if order['positionSide'] == _SHORT]

    if len(trades) == 0 or len(order_s) == 0:
        return []

    grouped_trades = defaultdict(list)
    for trade in trades:
        grouped_trades[trade['orderId']].append(trade)

    grouped_order_wrapper_d = defaultdict(list)
    for order in [*order_s, *algo_order_s]:
        if 'clientOrderId' in order and 'crypto-' in order['clientOrderId']:
            transaction_id = order['clientOrderId'].split("-")[1]
        elif 'clientAlgoId' in order and 'crypto-' in order['clientAlgoId']:
            transaction_id = order['clientAlgoId'].split("-")[1]
        else:
            continue

        grouped_order_wrapper_d[transaction_id].append(order)

    def filter_transaction_groups(transaction_order_group):
        result_s = []
        if len(transaction_order_group['order_s']) > 2:
            result_s.append(True)

        algo_order_s = [order for order in transaction_order_group['order_s'] if 'clientAlgoId' in order]
        finished_algo_order_s = [algo_order for algo_order in algo_order_s if algo_order['algoStatus'] == 'FINISHED']
        canceled_algo_order_s = [algo_order for algo_order in algo_order_s if algo_order['algoStatus'] == 'CANCELED']

        if any(finished_algo_order_s) or all(canceled_algo_order_s):
            result_s.append(True)

        is_matched = all(result_s)

        return is_matched

    transaction_orders_group_s = [{"transaction_id": k, "order_s": v} for k, v in grouped_order_wrapper_d.items()]
    transaction_orders_s = [tog for tog in transaction_orders_group_s if filter_transaction_groups(tog)]

    if len(transaction_orders_s) == 0:
        return []

    start_time = transaction_orders_s[0]['order_s'][0]['updateTime']
    end_time = transaction_orders_s[-1]['order_s'][-1]['updateTime']

    funding_fees = client.futures_income_history(symbol=symbol_join, incomeType='FUNDING_FEE')

    funding_rate_request_params = dict(
        symbol=symbol_join,
        startTime=int(start_time),
        endTime=int(end_time),
        limit=1000
    )
    funding_rates = client.futures_funding_rate(**funding_rate_request_params)

    if isinstance(funding_rates, dict) and 'status' in funding_rates and funding_rates['status'] == 'ERROR':
        funding_rates = []
        ERROR(f"FUNDING RATE ERROR: {funding_rate_request_params}")

    for funding_fee in funding_fees:
        funding_fee['kiev_dt'] = round_down_to_nearest_sec(localize_kiev_tz(funding_fee['time']))

    for funding_rate in funding_rates:
        funding_rate['funding_kiev_dt'] = round_down_to_nearest_sec(localize_kiev_tz(funding_rate['fundingTime']))

    transaction_s = []
    for transaction_orders in transaction_orders_s:
        try:
            transaction_id = transaction_orders['transaction_id']
            orders = transaction_orders['order_s']

            entry_order, exit_order, exit_algo_order = get_transaction_entry_exit_order(orders, order_s, long_manual_order_s, short_manual_order_s)

            if exit_order is None:
                continue

            entry_order_metrics = calculate_metrics(entry_order, grouped_trades)

            if 'web_' in exit_order['clientOrderId']:
                exit_order_metrics = calculate_reduceonly_order_exit_position_metrics(entry_order, exit_order, grouped_trades)
            else:
                exit_order_metrics = calculate_metrics(exit_order, grouped_trades)

            pos_qty = exit_order_metrics['total_qty']
            pos_side = entry_order_metrics['pos_side']

            trans_pnl = entry_order_metrics['total_pnl'] + exit_order_metrics['total_pnl']
            trans_fee = entry_order_metrics['total_fee'] + exit_order_metrics['total_fee']

            opened_kiev_dt = entry_order_metrics['update_kiev_dt']
            closed_kiev_dt = exit_order_metrics['update_kiev_dt']

            fund_fee = 0
            for funding_rate in funding_rates:
                funding_fees_ = [funding_fee for funding_fee in funding_fees if funding_fee['time'] == funding_rate['fundingTime']]
                if len(funding_fees_) > 0:
                    funding_fee = funding_fees_[0]
                else:
                    continue

                fund_kiev_dt = funding_rate['funding_kiev_dt']
                fund_rate = float(funding_rate['fundingRate'])
                mark_price = float(funding_rate['markPrice'])
                if opened_kiev_dt <= fund_kiev_dt <= closed_kiev_dt:
                    fund_fee += (pos_qty * mark_price) * (fund_rate * -1 if pos_side == _LONG else fund_rate)

            total_fee = trans_fee + fund_fee
            balance_movement = trans_pnl - total_fee

            if exit_algo_order is None:
                pos_result = 'PROFIT' if balance_movement > 0 else 'LOSS'
            else:
                if exit_algo_order['orderType'] == FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET:
                    pos_result = 'PROFIT'
                elif exit_algo_order['orderType'] == FUTURE_ORDER_TYPE_STOP_MARKET:
                    pos_result = 'LOSS'
                else:
                    raise AssertionError(f"WRONG ORDER TYPE: {exit_algo_order}")

            transaction = {
                'symbol_join': symbol_join,
                'transaction_id': transaction_id,
                'pos_side': pos_side,
                'pos_result': pos_result,
                'pos_qty': pos_qty,
                'trans_fee': trans_fee,
                'fund_fee': fund_fee,
                'pnl': trans_pnl,
                'total_fee': total_fee,
                'balance_movement': balance_movement,
                'opened_kiev_dt': opened_kiev_dt,
                'closed_kiev_dt': closed_kiev_dt,
                'order_s': orders,
            }

            transaction_s.append(transaction)
        except Exception as ex:
            from SRC.WEBAPP.libs.BinanceFuturesIsolatedTrader import write_uncompleted_transations

            write_uncompleted_transations(transaction_orders, symbol_join)

    return transaction_s

def get_futures_opened_transaction_grouped_algo_order_d(client, symbol_join):
    opened_algo_order_s = client.futures_get_open_algo_orders(symbol=symbol_join)

    grouped_algo_order_d = defaultdict(dict)
    for algo_order in opened_algo_order_s:
        transaction_id = algo_order['clientAlgoId'].split("-")[1]
        order_type = algo_order['orderType']
        position_side = algo_order['positionSide']
        quantity = algo_order['quantity']
        grouped_algo_order_d[transaction_id][order_type] = algo_order
        grouped_algo_order_d[transaction_id]['position_side'] = position_side
        grouped_algo_order_d[transaction_id]['quantity'] = quantity

    return grouped_algo_order_d

def get_margin_opened_transaction_grouped_order_d(client, symbol):
    opened_order_s = client.get_open_margin_orders(symbol=symbol)

    grouped_opened_order_d = defaultdict(dict)

    for opened_order in opened_order_s:
        transaction_id = str(opened_order['transaction_id'])
        order_type = opened_order['type']
        side = opened_order['side']
        grouped_opened_order_d[transaction_id]['side'] = side
        grouped_opened_order_d[transaction_id][order_type] = opened_order

    return grouped_opened_order_d

def get_symbol_asset_balance_info(symbol):
    symbol_info = get_symbol_info_file_cached(symbol)
    stable_asset, coin_asset = get_margin_assets_info(symbol_info)

    return stable_asset, coin_asset


def get__exist_quantity__price__by__existing_market_order_id(symbol, order_id):
    client = produce_binance_client_singleton()

    sym_join = _symbol_join(symbol)
    symbol_info = get_symbol_info_file_cached(symbol)
    stable_asset, coin_asset = get_margin_assets_info(symbol_info)

    order = client.get_margin_order(symbol=sym_join, orderId=order_id, isIsolated=False)
    side = order['side']
    enter_quantity = float(order['executedQty'])
    cummulativeQuoteQty = float(order['cummulativeQuoteQty'])
    price = cummulativeQuoteQty / enter_quantity
    exit_quantity = enter_quantity / CLOSE_POSITION_FEE_EXTRACTOR_RATIO
    coin_asset_free = float(coin_asset['free'])

    if side == SIDE_BUY:
        exit_quantity = coin_asset_free if coin_asset_free < exit_quantity else exit_quantity

    quantity_round_order = get_quantity_round_order(symbol)
    exit_quantity = floor(exit_quantity, quantity_round_order)

    return exit_quantity, price, side


def calc_exit_ratios_prices(symbol, enter_price, profit_price, profit_loss_ratio):
    price_round_order = get_price_round_order(symbol)

    if profit_price > enter_price:
        take_profit_ratio = profit_price / enter_price
        stop_loss_ratio = 1 + ((take_profit_ratio - 1) / profit_loss_ratio)
        loss_price = enter_price / stop_loss_ratio

        return take_profit_ratio, stop_loss_ratio, loss_price

    if profit_price < enter_price:
        take_profit_ratio = enter_price / profit_price
        stop_loss_ratio = 1 + ((take_profit_ratio - 1) / profit_loss_ratio)
        loss_price = enter_price * stop_loss_ratio

        return take_profit_ratio, stop_loss_ratio, loss_price

    raise RuntimeError(f"Enter price: {_float_5(enter_price)} and profit price: {_float_5(profit_price)} should be different")


def calc_mean_price(deal_result):
    mean_price = sum([float(fill['qty']) * float(fill['price'])  for fill in deal_result['fills']]) / sum([float(fill['qty']) for fill in deal_result['fills']])

    return mean_price


def get_max_borrow_asset(asset, recv_window_ms=None):
    client = produce_binance_client_singleton()

    if recv_window_ms is None:
        max_borrow = float(client.get_max_margin_loan(asset=asset)['amount'])
    else:
        max_borrow = float(client.get_max_margin_loan(asset=asset, recvWindow=recv_window_ms)['amount'])

    return max_borrow


def get_position_stable_amount(stable_asset, symbol, part):
    raise NotImplementedError

    # client = produce_binance_client_singleton()
    #
    # balance = stable_balance_futures(client, stable_asset)
    # stable_amount = balance / part
    # leverage = get_symbol_futures_isolated_leverage(client, symbol)
    # stable_margined_amount = stable_amount * leverage
    #
    # return stable_margined_amount


def borrow_asset(asset, quantity):
    client = produce_binance_client_singleton()

    print(client.get_margin_loan_details(asset=asset))
    print(client.get_margin_repay_details(asset=asset))

    client.create_margin_loan(asset=asset, amount=str(quantity))

    print(client.get_margin_loan_details(asset=asset))
    print(client.get_margin_repay_details(asset=asset))


def repay_asset(asset, quantity):
    client = produce_binance_client_singleton()

    print(client.get_margin_loan_details(asset=asset))
    print(client.get_margin_repay_details(asset=asset))

    client.repay_margin_loan(asset=asset, amount=str(quantity))

    print(client.get_margin_loan_details(asset=asset))
    print(client.get_margin_repay_details(asset=asset))


def get_spot_total_usdt(ticker_info):
    client = produce_binance_client_singleton()

    return get_account_usdt_equivalent(client.get_account()['balances'], ticker_info)


def get_margin_total_usdt(ticker_info):
    client = produce_binance_client_singleton()

    return get_account_usdt_equivalent(client.get_margin_account()['userAssets'], ticker_info)


def get_account_usdt_equivalent(account_balances, ticker_info):
    # ticker_prices = {ticker['symbol']: float(ticker['price']) for ticker in ticker_info}
    ticker_prices = {ticker['symbol']: float(ticker['lastPrice']) for ticker in ticker_info}

    coin_values = []
    for coin_balance in account_balances:
        coin_symbol = coin_balance['asset']
        unlocked_balance = float(coin_balance['free'])
        locked_balance = float(coin_balance['locked'])

        if coin_symbol == 'USDT' and unlocked_balance + locked_balance > 1:
            coin_values.append(('USDT', (unlocked_balance + locked_balance)))
        elif unlocked_balance + locked_balance > 0.0:
            if (any(coin_symbol + 'USDT' in i for i in ticker_prices)):
                ticker_symbol = coin_symbol + 'USDT'
                ticker_price = ticker_prices.get(ticker_symbol)
                if ticker_price is None:
                    continue

                coin_usdt_value = (unlocked_balance + locked_balance) * ticker_price
                if coin_usdt_value > 1:
                    coin_values.append((coin_symbol, coin_usdt_value))
            elif (any(coin_symbol + 'BTC' in i for i in ticker_prices)):
                ticker_symbol = coin_symbol + 'BTC'
                ticker_price = ticker_prices.get(ticker_symbol)
                if ticker_price is None:
                    continue

                coin_usdt_value = (unlocked_balance + locked_balance) * ticker_price * ticker_prices.get('BTCUSDT')
                if coin_usdt_value > 1:
                    coin_values.append((coin_symbol, coin_usdt_value))

    coin_values.sort(key=lambda x: x[1], reverse=True)

    return coin_values


def get_margin_debt_details(ticker_info):
    client = produce_binance_client_singleton()

    margin_account = client.get_margin_account()

    total_debt_usdt = 0.0
    asset_debt_s = []
    for asset_data in margin_account['userAssets']:
        asset = asset_data['asset']
        symbol = f"{asset}USDT"
        pair_info = find_first_list_item(ticker_info, 'symbol', symbol)
        if pair_info is None:
            continue

        borrowed = float(asset_data['borrowed'])
        interest = float(asset_data['interest'])
        asset_debt = borrowed + interest

        # asset_debt_usdt = asset_debt * float(pair_info['price'])
        asset_debt_usdt = asset_debt * float(pair_info['lastPrice'])

        if asset_debt_usdt > 1:
            asset_debt_s.append({'asset': asset, 'debt': asset_debt, 'debt_usdt': asset_debt_usdt})
            total_debt_usdt += asset_debt_usdt

    return {'total_usdt': total_debt_usdt, 'asset_debt_usdt_s': asset_debt_s}


def get_total_account_usdt_equity(exclude_symbol_s=[], print_out=False):
    client = produce_binance_client_singleton()

    ticker_info = client.get_all_tickers()
    tickers = get_active_coin_usdt_tickers(exclude_symbol_s)

    spot_coins_usdt_value = get_spot_total_usdt(tickers)
    margin_assets_usdt_value = get_margin_total_usdt(tickers)
    margin_debt_details = get_margin_debt_details(tickers)

    margin_total_usdt_dept = margin_debt_details['total_usdt']
    margin_asset_debt_usdt_s = margin_debt_details['asset_debt_usdt_s']

    info = []

    info.append(f"**SPOT**")
    for asset, usdt_value in spot_coins_usdt_value:
        info.append(f"- {asset}: ${usdt_value:.2f}")

    spot_grand_usdt_total = sum(map(lambda coin_usdt_value: coin_usdt_value[1], spot_coins_usdt_value))
    info.append(f"Grand SPOT Total: ${spot_grand_usdt_total:.2f}")
    info.append("\r\n")

    info.append(f"**MARGIN**")
    for asset, usdt_value in margin_assets_usdt_value:
        info.append(f"- {asset}: ${usdt_value:.2f}")

    margin_grand_usdt_total = sum(map(lambda coin_usdt_value: coin_usdt_value[1], margin_assets_usdt_value))
    info.append(f"Grand MARGIN Total: ${margin_grand_usdt_total:.2f}")

    info.append(f"**MARGIN DEBT**")
    for asset_debt in margin_asset_debt_usdt_s:
        if asset_debt['debt_usdt'] > 1:
            info.append(f"- {asset_debt['asset']}: {asset_debt['debt']} => ${asset_debt['debt_usdt']:.2f}")

    info.append(f"Debt MARGIN Total: ${margin_total_usdt_dept:.2f}")
    info.append("\r\n")

    total_account_usdt_equity = spot_grand_usdt_total + margin_grand_usdt_total - margin_total_usdt_dept
    info.append(f"**TOTAL ACCOUNT USDT EQUITY:** ${total_account_usdt_equity:.2f}")

    if print_out:
        printmd("\r\r".join(info))

    return total_account_usdt_equity


def produce_total_account_usdt_equity_figure(dt_s, usdt_equity_s):
    x_ticks_count_max = 3
    x_take_each_n = int(len(dt_s) / x_ticks_count_max) + 1 if len(dt_s) / x_ticks_count_max > 1 else 1

    x_ticks_sorted = sorted(dt_s)
    x_tickvals = [*[dt_s[0]], *[dt_s[-1]]]
    x_ticktext = [datetime_h_m__d_m(x_tickval) for x_tickval in x_tickvals]

    y_ticks_count_max = 4
    y_take_each_n = int(len(usdt_equity_s) / y_ticks_count_max) + 1 if len(usdt_equity_s) / y_ticks_count_max > 1 else 1

    y_ticks_sorted = sorted(usdt_equity_s)
    y_ticks_taken = y_ticks_sorted[::y_take_each_n]
    y_ticks_last = y_ticks_sorted[-1]
    y_tickvals = y_ticks_taken
    y_ticktext = [f"${int(round(y_tickval, 0))}" for y_tickval in y_tickvals]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dt_s, y=usdt_equity_s, mode="lines"))
    x_annotation = x_ticks_sorted[int(len(x_ticks_sorted) / 2)]
    y_annotation = y_ticks_sorted[int(len(y_ticks_sorted) / 2)]
    text_annotation = f"{datetime_h_m_s__d_m(dt_s[-1])} | <b>${round(usdt_equity_s[-1], 2)}</b>"
    annotation = dict(x=x_annotation, y=y_annotation, text=text_annotation, xanchor="center", font=dict(size=12, color='black'), bgcolor='cyan', showarrow=False, bordercolor="black", borderwidth=1)
    fig.add_annotation(**annotation)

    days_splitter_s = get_datetime_splitters(dt_s)
    for days_splitter in days_splitter_s:
        fig.add_vline(x=days_splitter, line_width=1, line_dash="dash", line_color="fuchsia")

    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(tickmode="array", tickvals=y_tickvals, ticktext=y_ticktext)
    fig.update_layout(height=150, margin=dict(l=0, r=0, t=30, b=0))
    fig.update_layout(title=f"{' - '.join(x_ticktext)}")

    return fig


def get_active_coin_usdt_tickers(exclude_symbol_s=[], include_symbol_s=[]):
    client = produce_binance_client_singleton()

    exchange_info = get_exchange_info_time_cached()
    tickers = client.get_ticker()

    trading_symbol_s = [symbol['symbol'] for symbol in exchange_info['symbols'] if symbol['status'] == 'TRADING' and _symbol_join(symbol['symbol']) not in [_symbol_join(exclude_symbol) for exclude_symbol in exclude_symbol_s]]
    active_coin_usdt_tickers = [ticker for ticker in tickers if ticker['symbol'].endswith('USDT') and ticker['symbol'] in trading_symbol_s]
    if len(include_symbol_s) > 0:
        active_coin_usdt_tickers = [ticker for ticker in active_coin_usdt_tickers if ticker['symbol'] in include_symbol_s]

    return active_coin_usdt_tickers


def get_margin_symbol_s():
    client = produce_binance_client_singleton()

    margin_all_pairs = client.get_margin_all_pairs()
    usdt_pair_s = [pair for pair in margin_all_pairs if pair['symbol'].endswith('USDT') and pair['isMarginTrade'] and pair['isBuyAllowed'] and pair['isSellAllowed']]
    margin_symbol_s = [usdt_pair['symbol'] for usdt_pair in usdt_pair_s]

    return margin_symbol_s


def get_top_gainers_losers(limit=10):
    try:
        active_coin_usdt_tickers = get_active_coin_usdt_tickers()
        sorted_active_coin_usdt_tickers = sorted(active_coin_usdt_tickers, key=lambda ticker: float(ticker['priceChangePercent']))

        top_gainer_tickers = sorted_active_coin_usdt_tickers[::-1][:limit]
        top_loser_tickers = sorted_active_coin_usdt_tickers[:limit]
        top_neutral_tickers = sorted([ticker for ticker in active_coin_usdt_tickers if _symbol_join(ticker['symbol']) not in ['EURUSDT', 'USDPUSDT', 'TUSDUSDT', 'FDUSDUSDT', 'USDCUSDT']], key=lambda ticker: abs(float(ticker['priceChangePercent'])))[
                              :limit]

        return top_gainer_tickers, top_neutral_tickers, top_loser_tickers
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


def get_tradingview_analisys(symbol, interval, print_out=False):
    ta_handler = TA_Handler(
        symbol=symbol,
        screener="crypto",
        exchange="BINANCE",
        interval=interval,
    )

    try:
        analysis = ta_handler.get_analysis()

        if print_out:
            print(f"Summary for {symbol}: {analysis.summary}")

            print("Technical Recommendation:", analysis.summary["RECOMMENDATION"])
            print("Buy Signals:", analysis.summary["BUY"])
            print("Sell Signals:", analysis.summary["SELL"])
            print("Neutral Signals:", analysis.summary["NEUTRAL"])

        return analysis
    except Exception as e:
        print(f"An error occurred: {e}")

        return None


@lru_cache(maxsize=5)
def get_tradingview_recommendation_cached(symbol, interval, last_checked_minute):
    analisys = get_tradingview_analisys(symbol, interval, print_out=False)
    if analisys is None:
        return 'NEUTRAL'

    return analisys.summary["RECOMMENDATION"]


def get_tradingview_recommendations_bulk(symbol_s, interval_s):
    exchange = "BINANCE"
    screener = "crypto"

    symbols = [f"{exchange}:{symbol}" for symbol in symbol_s]

    interval_result_d = {}
    for interval in interval_s:
        try:
            proxies = None
            if USE_PROXY_CLIENT():
                proxies = {
                    "http": "http://127.0.0.1:8888",
                    "https": "http://127.0.0.1:8888",
                }

            # interval_result = get_multiple_analysis(screener, interval, symbols, proxies=proxies)
            interval_result = {}
            batch_size = 500
            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i + batch_size]
                try:
                    res = get_multiple_analysis(
                        screener=screener,
                        interval=interval,
                        symbols=batch,
                        proxies=proxies
                    )
                    interval_result = merge_dicts(interval_result, res)
                except Exception as e:
                    print(f"Batch {i // batch_size} failed: {e}\r\n{str(batch)}")
            interval_result_d[interval] = interval_result
        except json.decoder.JSONDecodeError as ex:
            EXCEPTION_SPLITTED(f"FAILED [TA version {tradingview_ta.__version__}]: {interval}", str(ex))
        finally:
            time.sleep(15)

    return interval_result_d


@lru_cache(maxsize=5)
def build_tradingview_symbol_link(symbol):
    return f"<a href=\"https://www.tradingview.com/chart/?symbol={symbol}\">{symbol}</a>"


@lru_cache(maxsize=5)
def build_binance_symbol_margin_link(symbol):
    return f"<a href=\"https://www.binance.com/en/trade/{symbol}?type=cross\">{symbol}</a>"


def append_tradingview_recommendation(symbol, kiev_now, utc_now, price, symbol_interval_recommendation_s):
    new_row = merge_dicts({
        _UTC_TIMESTAMP: utc_now.replace(microsecond=0),
        _KIEV_TIMESTAMP: kiev_now.replace(microsecond=0),
        'price': price,
    }, merge_dict_s(symbol_interval_recommendation_s[::-1]))

    csv_file = _SYMBOL_TRADINGVIEW_RECOMMENDATION_DF_FILE_PATH(symbol)
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    new_row_df = pd.DataFrame([new_row])
    new_row_df.to_csv(csv_file, mode='a', header=not pd.io.common.file_exists(csv_file), index=False)


def produce_append_tradingview_recommendation_recorder(remove_existing_data=False):

    return append_tradingview_recommendation


def produce_total_account_usdt_equity_presenter(exclude_symbol_s=[], print_out=False):
    dt_s = []
    usdt_equity_s = []

    def present(check_dt):
        total_account_usdt_equity = get_total_account_usdt_equity(exclude_symbol_s, print_out=print_out)

        dt_s.append(check_dt)
        usdt_equity_s.append(total_account_usdt_equity)

        fig = produce_total_account_usdt_equity_figure(dt_s=dt_s, usdt_equity_s=usdt_equity_s)

        return fig

    return present


def featurize_ta_recommendation(df):
    return df


def fill_ohlc(df_ts, df_ohlc, target_col):
    symbol = df_ohlc.iloc[0][_SYMBOL]
    discretization = df_ohlc.iloc[0][_DISCRETIZATION]

    for target_idx, target_row in df_ohlc.iterrows():
        end_dt = target_idx + TIME_DELTA(discretization)
        start_dt = target_idx
        target_val_s = df_ts.loc[start_dt:end_dt][target_col].to_list()
        recommendation_s = []
        for origin_idx, origin_row in enumerate(target_val_s):
            for i in range(origin_idx + 1):
                recommendation_s.append(origin_row)

        if len(recommendation_s) > 0:
            recommendation_counts = Counter(recommendation_s)
            final_recommendation = max(recommendation_counts, key=recommendation_counts.get)
        else:
            final_recommendation = 'NEUTRAL'
            DEBUG(f"{symbol} | {discretization} | {target_idx} | NEUTRAL")

        df_ohlc.loc[target_idx, target_col] = final_recommendation

    return df_ohlc


def fetch_analytics(arg):
    symbol = arg['symbol']
    ohlc_discretization = arg['ohlc_discretization']
    discretization_s = arg['discretization_s']
    interval_s = arg['interval_s']
    close_grad_s = arg['close_grad_s']
    shift_open_close = arg['shift_open_close']
    timeperiod = arg['timeperiod'] if 'timeperiod' in arg and arg['timeperiod'] is not None else TIME_DELTA('1D') * 365

    tar_df = fetch_ta_recommendation_data(symbol, timeperiod)
    mls_df = fetch_margin_loan_state_df(symbol, timeperiod)

    time_feature = _KIEV_TIMESTAMP
    if len(mls_df) > len(tar_df):
        target_feature = 'close'
        target_df = mls_df
    else:
        target_feature = 'price'
        target_df = tar_df

    df_ohlc_d = {}
    for discretization in discretization_s:
        target_col = f'ta_{discretization.lower()}'

        df_ohlc = candelify(tar_df, symbol, discretization, shift_open_close=shift_open_close, target_feature='price')
        df_ohlc_filled = fill_ohlc(tar_df, df_ohlc, target_col)

        df_ohlc_volatility = featurize_ta_recommendation(df_ohlc_filled)
        df_ohlc_d[discretization] = df_ohlc_volatility

    tar_df[_SYMBOL] = symbol
    mls_df[_SYMBOL] = symbol

    close_grad_config_s = [{'feature': 'close', 'window': window} for window in close_grad_s]
    df_ohlc_grad = candelify(target_df, symbol, ohlc_discretization, shift_idx=False, shift_open_close=True, target_feature=target_feature)
    df_ohlc_grad = linear_interpolate(df_ohlc_grad, [])
    df_ohlc_grad = featurize_gradient(df_ohlc_grad, ohlc_discretization, close_grad_config_s)
    df_ohlc_grad, grad_diff_col_s = featurize_gradient_extremums(df_ohlc_grad, feature='close')

    return {
        'symbol': symbol,
        'tar_data': {
            'df': tar_df,
            'df_ohlc_d': df_ohlc_d,
            'interval_s': interval_s,
        },
        'mls_data': {
            'df': mls_df,
        },
        'close_grad': {
            'df_ohlc': df_ohlc_grad,
            'grad_config_s': close_grad_config_s,
            'grad_diff_col_s': grad_diff_col_s,
        }
    }


def fetch_margin_loan_state_df(symbol, timeperiod=TIME_DELTA('1D') * 365):
    file_path = _SYMBOL_MARGIN_LOAN_WATCHER_FILE_PATH(symbol)
    try:
        df_ts = pd.read_csv(file_path, index_col=_UTC_TIMESTAMP, parse_dates=[_UTC_TIMESTAMP, _KIEV_TIMESTAMP], infer_datetime_format=True)

        end_dt = df_ts.iloc[-1].name
        start_dt = end_dt - timeperiod
        df_ts = df_ts[df_ts.index >= start_dt]
        df_ts = df_ts[df_ts.index <= end_dt]

        return df_ts
    except FileNotFoundError as err:
        raise FileNotFoundError(f"NO MARGIN LOAN FILE: {file_path} | {str(err)}")


def fetch_ta_recommendation_data(symbol, timeperiod=TIME_DELTA('1D') * 365):
    file_path = _SYMBOL_TRADINGVIEW_RECOMMENDATION_DF_FILE_PATH(symbol)
    try:
        df_ts = pd.read_csv(file_path, index_col=_UTC_TIMESTAMP, parse_dates=[_UTC_TIMESTAMP, _KIEV_TIMESTAMP], infer_datetime_format=True)

        end_dt = df_ts.iloc[-1].name
        start_dt = end_dt - timeperiod
        df_ts = df_ts[df_ts.index >= start_dt]
        df_ts = df_ts[df_ts.index <= end_dt]

        return df_ts
    except FileNotFoundError as err:
        ERROR(f"\r\nNO TRADINGVIEW RECOMMENDATION FILE: {file_path}")

        return pd.DataFrame()


def fetch_analytics_data_s(include_symbol_s, exclude_symbol_s, interval_s, close_grad_s, shift_open_close=False, timeperiod=None, num_workers=1, print_out=True):
    tickers = get_active_coin_usdt_tickers(include_symbol_s, exclude_symbol_s)
    symbol_s = [ticker['symbol'] for ticker in tickers]
    if len(include_symbol_s) > 0:
        symbol_s = [symbol for symbol in symbol_s if symbol in include_symbol_s]

    arg_s = [{'symbol': symbol, 'discretization_s': [interval.upper() for interval in interval_s], 'interval_s': interval_s, 'close_grad_s': close_grad_s, 'shift_open_close': shift_open_close, 'timeperiod': timeperiod} for symbol in symbol_s]
    analytics_data_s = func_multi_process(fetch_analytics, arg_s, num_workers, print_result_full=len(arg_s) <= 5, print_out=print_out)

    return analytics_data_s


def append__ticker__unwrap(args):
    symbol = args['symbol']
    price = args['last_price']
    u_now = args['u_now']
    k_now = args['k_now']
    interval_recommendation_d = args['interval_recommendation_d']
    no_recommendation_symbol_s = args['no_recommendation_symbol_s']

    interval_recommend_s = [{'interval': interval, 'recommendation': recommendation[f'BINANCE:{symbol}']} for interval, recommendation in interval_recommendation_d.items()]
    if any([interval_recommend['recommendation'] is None for interval_recommend in interval_recommend_s]):
        none_recommendations_present = " | ".join([f"{interval_recommend['interval']}: None" for interval_recommend in interval_recommend_s if interval_recommend['recommendation'] is None])
        no_recommendation_symbol_s.append({'symbol': symbol, 'none_recommendations_present': none_recommendations_present})

    symbol_interval_recommendation_s = [{f"ta_{interval_recommend['interval']}": interval_recommend['recommendation'].summary['RECOMMENDATION'] if interval_recommend['recommendation'] is not None else 'NEUTRAL'} for interval_recommend in
                                        interval_recommend_s]
    append_tradingview_recommendation(symbol, k_now, u_now, price, symbol_interval_recommendation_s)

    return f"{symbol} | {price}"


def append_ta_recommendation_parallel(tickers, interval_recommendation_d, u_now, k_now, num_workers=1):
    manager = multiprocessing.Manager()
    no_recommendation_symbol_s = manager.list()

    arg_s = [{
        'symbol': ticker['symbol'],
        'last_price': ticker['lastPrice'],
        'u_now': u_now,
        'k_now': k_now,
        'interval_recommendation_d': interval_recommendation_d,
        'no_recommendation_symbol_s': no_recommendation_symbol_s,
    } for ticker in tickers]

    func_multi_process(append__ticker__unwrap, arg_s, num_workers=num_workers, print_result_full=False)

    return no_recommendation_symbol_s


def run_tradingview_recommendations_collector(include_symbol_s, exclude_symbol_s, interval_s, block_until_round, remove_existing_data, num_workers):
    # presenter = produce_total_account_usdt_equity_presenter(exclude_symbol_s)
    btc_recommendation_file = _SYMBOL_TRADINGVIEW_RECOMMENDATION_DF_FILE_PATH('BTCUSDT')
    os.makedirs(os.path.dirname(btc_recommendation_file), exist_ok=True)
    if remove_existing_data:
        run_safety_interrupter("###### REMOVE EXISTING DATA ######", 15)

        remove_all_files_from_folder(os.path.dirname(btc_recommendation_file))

    no_last_recommendation_symbol_s = []

    counter = 0
    def iterate():
        nonlocal counter

        if counter > 0:
            printmd(f"**NO RECOMMENDATION SYMBOLS:**")
            for no_recommendation_symbol in no_last_recommendation_symbol_s:
                print(f"{no_recommendation_symbol['symbol']}: {no_recommendation_symbol['none_recommendations_present']}")

        block_until_next(block_until_round, title="TVA")
        clear_output()

        u_now = utc_now()
        k_now = kiev_now()
        time_present = datetime_Y_m_d__h_m_s(k_now)
        printmd(f"**STARTED:** {time_present}")

        measure = produce_measure()

        margin_symbol_s = get_margin_symbol_s()
        tickers = get_active_coin_usdt_tickers(exclude_symbol_s=exclude_symbol_s, include_symbol_s=margin_symbol_s)
        symbol_s = [ticker['symbol'] for ticker in tickers]

        if len(include_symbol_s) > 0:
            symbol_s = [symbol for symbol in symbol_s if symbol in include_symbol_s]

        interval_recommendation_d = get_tradingview_recommendations_bulk(symbol_s, interval_s)
        manager = multiprocessing.Manager()
        no_recommendation_symbol_s = manager.list()

        arg_s = [{
            'symbol': ticker['symbol'],
            'last_price': ticker['lastPrice'],
            'u_now': u_now,
            'k_now': k_now,
            'interval_recommendation_d': interval_recommendation_d,
            'no_recommendation_symbol_s': no_recommendation_symbol_s,
        } for ticker in tickers if ticker['symbol'] in symbol_s]

        func_multi_thread_executor(append__ticker__unwrap, arg_s, num_workers=num_workers, print_result_full=not is_cloud())

        # try:
        #     fig = presenter(k_now)
        #     fig.show()
        # except:
        #     traceback.print_exc()

        no_last_recommendation_symbol_s.clear()
        for no_margin_loan_symbol in no_recommendation_symbol_s:
            no_last_recommendation_symbol_s.append(no_margin_loan_symbol)

        printmd(f"**DURATION:** {measure()}\r\nHANDLED: {symbol_s}")

        counter += 1

    while True:
        tryall_delegate(iterate, tryalls_count=3)


def repay_cross(asset_info, symbol):
    client = produce_binance_client_singleton()

    borrowed = asset_info['borrowed']
    asset_name = asset_info['asset']

    client.repay_margin_loan(asset=asset_name, amount=borrowed, isIsolated='False', symbol=symbol)


def close_cross_margin_position(symbol):
    client = produce_binance_client_singleton()

    price = tryall_delegate(lambda: get_margin_symbol_ticker_current_price(symbol), 'GET PRICE', tryalls_count=5)

    symbol_info = get_symbol_info_file_cached(symbol)
    price_tick_size, quantity_step_size, min_stable_allowed = get_tick_step_info_spot(hashabledict(symbol_info))

    stable_asset_info, coin_asset_info = get_margin_assets_info(symbol_info)

    coin_net = float(coin_asset_info['netAsset'])
    coin_free = float(coin_asset_info['free'])
    coin_borrowed = float(coin_asset_info['borrowed'])
    stable_free = float(stable_asset_info['free'])
    stable_borrowed = float(stable_asset_info['borrowed'])

    if 0 < coin_borrowed <= coin_free:
        repay_cross(coin_asset_info, symbol)

    if 0 < stable_borrowed <= stable_free:
        repay_cross(stable_asset_info, symbol)

    if coin_free * price <= 1 and is_close_to_zero(stable_borrowed, 0.1) and is_close_to_zero(coin_borrowed * price, 0.1):
        return

    changed = False
    quantity_round_order = get_quantity_round_order(symbol)

    coin_stable_net = abs(floor(coin_net, quantity_round_order) * price)
    if coin_net < 0:
        coin_buy = (coin_stable_net if coin_stable_net > min_stable_allowed else min_stable_allowed) / price
        coin_to_buy = floor(coin_buy * 1.02, get_round_order(quantity_step_size))
        if coin_to_buy * price < min_stable_allowed:
            coin_to_buy += 1

        client.create_margin_order(
            symbol=symbol,
            side=SIDE_BUY,
            type=ORDER_TYPE_MARKET,
            quantity=coin_to_buy,
            isIsolated='False'
        )

        changed = True

    if coin_net > 0 and coin_stable_net > min_stable_allowed:
        coin_to_sell = floor(coin_net, get_round_order(quantity_step_size))
        client.create_margin_order(
            symbol=symbol,
            side=SIDE_SELL,
            type=ORDER_TYPE_MARKET,
            quantity=coin_to_sell,
            isIsolated='False'
        )

        changed = True

    time.sleep(1)

    stable_asset_info, coin_asset_info = get_margin_assets_info(symbol_info)

    coin_borrowed = float(coin_asset_info['borrowed'])
    stable_borrowed = float(stable_asset_info['borrowed'])

    if coin_borrowed > 0:
        repay_cross(coin_asset_info, symbol)
        changed = True

    if stable_borrowed > 0:
        repay_cross(stable_asset_info, symbol)
        changed = True

    if changed:
        time.sleep(1)

        stable_asset_info, coin_asset_info = get_margin_assets_info(symbol_info)

        CONSOLE_SPLITTED(f"{coin_asset_info} | {stable_asset_info}")


def binance_generate_x_mbx_apikey():
    API_KEY = BINANCE_API_KEY
    API_SECRET = BINANCE_API_SECRET

    timestamp = int(time.time() * 1000)

    params = {
        'timestamp': timestamp
    }

    query_string = '&'.join([f"{key}={value}" for key, value in params.items()])
    signature = hmac.new(
        API_SECRET.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

    params['signature'] = signature

    return timestamp, signature, API_KEY


def get_symbol_futures_isolated_leverage(client, symbol):
    futures_account = client.futures_account()
    symbol_positions = [sym for sym in futures_account['positions'] if sym['symbol'] == symbol]
    leverage = symbol_positions[0]['leverage']

    return int(leverage)


def stable_balance_futures(client, stable_asset):
    while True:
        try:
            balance = client.futures_account_balance()
            stable_balance = float([bal['availableBalance'] for bal in balance if bal['asset'] == stable_asset][0])

            return stable_balance
        except Exception as e:
            EXCEPTION("FUTURES ACCOUNT ERROR:", e)
            time.sleep(1)
            pass


def check_futures_order_status(client, order, condition=lambda order_status: order_status == "FILLED"):
    symbol = order['symbol']
    oid = order['orderId']
    order_check = client.futures_get_order(symbol=symbol, orderId=oid, isIsolated=True)
    if order_check is None:
        return None

    order_status = order_check['status']

    return condition(order_status)


def check_futures_algo_order_status(client, order, condition=lambda order_status: order_status == "FINISHED"):
    symbol = order['symbol']
    oid = order['algoId']
    order_check = client.futures_get_algo_order(symbol=symbol, algoId=oid, isIsolated=True)
    if order_check is None:
        return None

    order_status = order_check['algoStatus']

    return condition(order_status)


def get_isolated_assets(client, symbol, print_out=True, log_s=None):
    asset_info = client.get_isolated_margin_account(symbols=symbol)['assets'][0]

    def get_asset_info(is_stable):
        asset_type = 'quoteAsset' if is_stable else 'baseAsset'
        net = asset_info[asset_type]['netAsset']
        free = asset_info[asset_type]['free']
        locked = asset_info[asset_type]['locked']
        total_borrowed = asset_info[asset_type]['borrowed']

        log = f"- {asset_info[asset_type]['asset']} | net: {_float_5(net)} | free: {_float_5(free)} | borrowed: {_float_5(total_borrowed)} | locked: {_float_5(locked)}"
        if print_out:
            print_log_trades(log)

        if log_s:
            log_s.append(log)

        return float(net), float(free), float(total_borrowed), float(locked)

    return {'stable': lambda: get_asset_info(True), 'coin': lambda: get_asset_info(False), 'fee': lambda: get_fee_asset(print_out)}


def check_symbol_availability(symbol: str):
    client = produce_binance_client_singleton()

    measure = produce_measure()

    result = {
        _BACKTEST: False,
        _MARGIN: False,
        _FUTURES: False,
    }

    # --- Spot & Margin ---
    try:
        exchange_info = client.get_exchange_info()
        for s in exchange_info["symbols"]:
            if s["symbol"] == symbol.upper():
                result[_BACKTEST] = True
                result[_MARGIN] = s.get("isMarginTradingAllowed", False)
                break
    except Exception as e:
        print("Error fetching spot/margin info:", e)

    # --- Futures ---
    try:
        futures_info = client.futures_exchange_info()
        for s in futures_info["symbols"]:
            if s["symbol"] == symbol.upper():
                result[_FUTURES] = s['status'] != 'SETTLING'
                break
    except Exception as e:
        print("Error fetching futures info:", e)

    DEBUG(f"Duration: {measure()}")

    return result


def check_symbol_availability(*symbol_s: str):
    client = produce_binance_client_singleton()

    measure = produce_measure()

    # result_d = {symbol: { _BACKTEST: False, _MARGIN: False, _FUTURES: False } for symbol in symbol_s}
    result_d = {symbol: { _MARGIN: False, _FUTURES: False } for symbol in symbol_s}

    # --- Spot & Margin ---
    try:
        exchange_info = client.get_exchange_info()
        for s in exchange_info["symbols"]:
            if s["symbol"] in result_d:
                # result_d[s["symbol"]][_BACKTEST] = True
                result_d[s["symbol"]][_MARGIN] = s.get("isMarginTradingAllowed", False)
    except Exception as e:
        print("Error fetching spot/margin info:", e)

    # --- Futures ---
    try:
        futures_info = client.futures_exchange_info()
        for s in futures_info["symbols"]:
            if s["symbol"] in result_d:
                result_d[s["symbol"]][_FUTURES] = s['status'] != 'SETTLING'
    except Exception as e:
        print("Error fetching futures info:", e)

    DEBUG(f"Duration: {measure()}")

    return result_d


def setup_futures_isolate_symbol(_sym_join, ext_log_s=None):
    with log_context(CONSOLE_SPLITTED) as log_s:
        log_s = log_s if ext_log_s is None else ext_log_s

        try:
            client = produce_binance_client_singleton()

            client.futures_change_leverage(symbol=_sym_join, leverage=LEVERAGE())  # Change MARGIN RATIO
            client.futures_change_margin_type(symbol=_sym_join, marginType='ISOLATED')  # Change MARGIN TYPE

            log_s.append(f"CHANGED [MARGIN TYPE: ISOLATED & LEVERAGE: {LEVERAGE()}] ##")
        except binance.exceptions.BinanceAPIException as binanceEx:
            if binanceEx.code == -4046:
                log_s.append("## NO NEED TO CHANGE [MARGIN TYPE & LEVERAGE] ##")
            else:
                log_s.append(str(binanceEx))
                log_s.append(EXCEPTION_SPLITTED)

                raise


if __name__ == "__main__":
    pass