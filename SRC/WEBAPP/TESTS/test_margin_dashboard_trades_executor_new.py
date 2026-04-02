import asyncio
import faulthandler
import os
import random
import threading
import time
import uuid
from asyncio import Queue
from functools import lru_cache
from itertools import groupby
from types import MethodType
from unittest.mock import MagicMock

import numpy as np
from binance import SIDE_BUY, SIDE_SELL

from SRC.CORE._CONSTANTS import _DASHBOARD_SEGMENT, _DASHBOARD_SEGMENT_AUTOTRADING, _MARKET_TYPE, _SYMBOL_JOIN, _LONG, _SYMBOL_DASH, _BINANCE_FEE, BINANCE_MAKER_COMISSION, BINANCE_TAKER_COMISSION, _AUTOTRADING_REGIME, \
    _PROD, _MOCK, _DASHBOARD_SEGMENT_BACKTESTING, _BACKTEST, SIGNAL_LONG_IN, SIGNAL_SHORT_IN, SIGNAL_IGNORE, _NET_FOLDER, _MAX_ORDERS_COUNT, _SHORT, _TRADES_FILE_PATH, _BALANCE_FILE_PATH, _DATETIME_PRICES_FILE_PATH, \
    _DATETIME_PRICE_FILE_PATH, _AUTOMATION_TYPE, _AUTOTRADING, _TRANSFER_CROSS_ISOLATED, _STOP_ON_DROP_DOWN_RATIO, BINANCE_COMISSION, _PART, _symbol, _autotrading_regime, _market_type, _presentation_type, _FUTURES, _DASHBOARD
from SRC.CORE._CONSTANTS import _LIMIT_MAKER, _STOP_LOSS_LIMIT, _DISCRETIZATION
from SRC.CORE.debug_utils import EXCEPTION_SPLITTED
from SRC.CORE.debug_utils import SET_SYMBOL, CONSOLE_SPLITTED, get_public_ip, get_local_ip, SET_CONSOLE_LOGLEVEL, SET_BINANCE_PROD, SET_NOTICE_LOGLEVEL, log_mock_calls
from SRC.CORE.utils import _float_5, datetime_Y_m_d__h_m_s, write_json
from SRC.LIBRARIES.binance_helpers import produce_binance_client_singleton_wrapper, get_margin_symbol_ticker_current_price, get_margin_isolated_total_net_USDT
from SRC.LIBRARIES.new_utils import __uuid4_12 as uuid4_12, get_datetime_price as _get_datetime_price, get_candle as _get_candle, print_log_trades
from SRC.LIBRARIES.new_utils import calculate_weighted_average_price, printmd_populated_char_n_times, check_env_true
from SRC.LIBRARIES.new_utils import parse_net_folder_hashed, floor, tryall_delegate, set_price, parse_string_variables, produce_net_folder, create_folder_file, write_file, _symbol_dash
from SRC.LIBRARIES.time_utils import TIME_DELTA
from SRC.LIBRARIES.time_utils import utc_now
from SRC.WEBAPP.libs.BinanceMarginIsolatedTrader import trade_utils_producer


@lru_cache(maxsize=None)
def produce_mock_margin_isolated_binance_client_singleton():
    mock_client = MagicMock()

    def init(self):
        symbol_slash = os.environ['SYMBOL_SLASH']
        coin_asset = symbol_slash.split("/")[0]
        stable_asset = symbol_slash.split("/")[1]
        fee_asset = 'BNB'
        symbol = f'{coin_asset}{stable_asset}'

        self.symbol = symbol
        self.fee_asset = fee_asset
        self.stable_asset = stable_asset
        self.coin_asset = coin_asset
        self.fee = BINANCE_COMISSION()
        self.maker_fee = BINANCE_MAKER_COMISSION() #MAKER > LIMIT: EXIT TAKE PROFIT
        self.taker_fee = BINANCE_TAKER_COMISSION() #TAKER > MARKET: ENTER POSITION | EXIT STOP LOSS

        self.margin_ratio = 5

        self.coin_free = 0.0
        self.coin_locked = 0.0
        self.coin_borrowed = 0.0

        self.stable_free = 0.0
        self.stable_locked = 0.0
        self.stable_borrowed = 0.0

        self.fee_availabe = 0

        self.oco_orders = []
        self.market_orders = []

        self.counter = 0

        self._recalc_net()

    def _recalc_net(self):
        self.coin_net = self.coin_free + self.coin_locked - self.coin_borrowed
        self.stable_net = self.stable_free + self.stable_locked - self.stable_borrowed

    def transfer__SPOT__ISOLATED_MARGIN(self, asset, amount, toSymbol):
        if asset == self.coin_asset:
            self.coin_free += amount

            self._recalc_net()
            return
        if asset == self.stable_asset:
            self.stable_free += amount

            self._recalc_net()
            return

    def transfer__ISOLATED_MARGIN__SPOT(self, asset, amount, fromSymbol):
        if asset == self.coin_asset:
            self.coin_free -= amount

            self._recalc_net()
            return
        if asset == self.stable_asset:
            self.stable_free -= amount
            self.stable_free = self.stable_net + self.stable_borrowed - self.stable_locked

            self._recalc_net()
            return

    def get_exchange_info(self):
        client = produce_binance_client_singleton_wrapper()
        exchange_info = client.get_exchange_info()

        return exchange_info

    def futures_exchange_info(self):
        client = produce_binance_client_singleton_wrapper()
        exchange_info = client.futures_exchange_info()

        return exchange_info

    def get_symbol_ticker(self, symbol):
        datetime_price = _get_datetime_price()
        price = datetime_price['price']

        return {'price': price}

    def get_margin_account(self):
        return {
            "userAssets": [{
                "asset": f"{self.fee_asset}",
                "netAsset": f"{self.fee_availabe}"
            }]
        }

    def get_isolated_margin_account(self, **params):
        return {
            "assets": [{
                "baseAsset": {
                    "asset": f"{self.coin_asset}",
                    "netAsset": f"{self.coin_net}",
                    "free": f"{self.coin_free}",
                    "borrowed": f"{self.coin_borrowed}",
                    "locked": f"{self.coin_locked}",
                },
                "quoteAsset": {
                    "asset": f"{self.stable_asset}",
                    "netAsset": f"{self.stable_net}",
                    "free": f"{self.stable_free}",
                    "borrowed": f"{self.stable_borrowed}",
                    "locked": f"{self.stable_locked}",
                },
                "symbol": f"{self.symbol}",
                "marginRatio": f"{self.margin_ratio}",
            }]
        }

    def get_max_margin_loan(self, asset, isolatedSymbol, isIsolated=True):
        datetime_price = _get_datetime_price()
        price = datetime_price['price']

        total_net_USDT = self.get_total_net_USDT()
        if asset == self.coin_asset:
            total_coin_locked = (self.stable_locked / price) + self.coin_locked
            total_net_COIN = total_net_USDT / price
            coin_borrowable = total_net_COIN * (self.margin_ratio - 1)
            coin_margin_loan = coin_borrowable - total_coin_locked

            return {'amount': coin_margin_loan}
        else:
            total_stable_locked = (self.coin_locked * price) + self.stable_locked
            stable_borrowable = total_net_USDT * (self.margin_ratio - 1)
            stable_margin_loan = stable_borrowable - total_stable_locked

            return {'amount': stable_margin_loan}

    def get_open_margin_orders(self, symbol, isIsolated=True):
        self.oco_orders.sort(key=lambda x: x['timestamp'])
        grouped_data = {key: list(group) for key, group in groupby(self.oco_orders, key=lambda x: x['transaction_id'])}

        open_orders = []
        for transaction_id, group in grouped_data.items():
            take_profit = group[0]
            stop_loss = group[1]

            if take_profit['status'] == 'NEW' and stop_loss['status'] == 'NEW':
                open_orders.append(take_profit)
                open_orders.append(stop_loss)

        return open_orders

    def get_margin_order(self, symbol, orderId, isIsolated):
        for oco_order in self.oco_orders:
            if oco_order['orderId'] == orderId:
                return oco_order

    def create_margin_order(self, symbol, side, type, quantity, isIsolated, sideEffectType="NO_SIDE_EFFECT", transaction_id="TRANSACTION_ID"):
        datetime_price = _get_datetime_price()
        price = datetime_price['price']
        order_id = str(uuid.uuid4())[:8]
        self___taker_fee = self.taker_fee
        self___maker_fee = self.maker_fee

        if price is None:
            raise ValueError("price required in mock")

        executed_qty = quantity
        quote_qty = quantity * price

        marginBuyBorrowAsset = None
        marginBuyBorrowAmount = 0.0

        marginSellBorrowAsset = None
        marginSellBorrowAmount = 0.0

        # =========================
        # STEP 1: BORROW
        # =========================
        if side == "BUY":
            deficit = quote_qty - self.stable_free
            if deficit > 0:
                if sideEffectType in ("MARGIN_BUY", "AUTO_REPAY"):
                    self.stable_borrowed += deficit
                    self.stable_free += deficit

                    marginBuyBorrowAsset = self.stable_asset
                    marginBuyBorrowAmount = deficit
                else:
                    raise Exception("Insufficient balance")

        if side == "SELL":
            deficit = executed_qty - self.coin_free
            if deficit > 0:
                if sideEffectType in ("MARGIN_BUY", "AUTO_REPAY"):
                    self.coin_borrowed += deficit
                    self.coin_free += deficit

                    marginSellBorrowAsset = self.coin_asset
                    marginSellBorrowAmount = deficit
                else:
                    raise Exception("Insufficient balance")

        # =========================
        # STEP 2: EXECUTION
        # =========================
        if side == "BUY":
            self.stable_free -= quote_qty
            self.coin_free += executed_qty

        if side == "SELL":
            self.coin_free -= executed_qty
            self.stable_free += quote_qty

        # =========================
        # STEP 3: COMMISSION
        # =========================
        if side == "BUY":
            coin_commission = executed_qty * self___taker_fee

            self.coin_free -= coin_commission

            commission = coin_commission
            commission_asset = self.coin_asset

        if side == "SELL":
            stable_commission = quote_qty * self___taker_fee

            self.stable_free -= stable_commission

            commission = stable_commission
            commission_asset = self.stable_asset

        # =========================
        # STEP 4: AUTO REPAY
        # =========================
        if sideEffectType == "AUTO_REPAY":
            if side == "BUY":
                repay = min(self.coin_free, self.coin_borrowed)
                self.coin_free -= repay
                self.coin_borrowed -= repay

            if side == "SELL":
                repay = min(self.stable_free, self.stable_borrowed)
                self.stable_free -= repay
                self.stable_borrowed -= repay

        self._recalc_net()

        # =========================
        # FILLS (Binance format)
        # =========================
        fills = [{
            "price": f"{price}",
            "qty": f"{executed_qty}",
            "commission": f"{commission}",
            "commissionAsset": f"{commission_asset}",
        }]

        order_result = {
            "symbol": self.symbol,
            "orderId": f"{order_id}",
            "transaction_id": f"{transaction_id}",
            "side": side,
            "status": "FILLED",
            "executedQty": executed_qty,
            "price": price,
            "sideEffectType": sideEffectType,
            "fills": fills,
        }

        # only attach if used
        if marginBuyBorrowAsset:
            order_result["marginBuyBorrowAsset"] = marginBuyBorrowAsset
            order_result["marginBuyBorrowAmount"] = str(marginBuyBorrowAmount)

        if marginSellBorrowAsset:
            order_result["marginSellBorrowAsset"] = marginSellBorrowAsset
            order_result["marginSellBorrowAmount"] = str(marginSellBorrowAmount)

        self.market_orders.append(order_result)

        return order_result

    def create_margin_oco_order(
            self,
            symbol,
            side,
            quantity,
            price,
            stopPrice,
            stopLimitPrice,
            stopLimitTimeInForce,
            isIsolated,
            sideEffectType,
            transaction_id
    ):
        take_profit_order_id = str(uuid.uuid4())[:8]
        stop_loss_order_id = str(uuid.uuid4())[:8]

        # =========================
        # BORROW (if needed)
        # =========================
        marginBuyBorrowAsset = None
        marginBuyBorrowAmount = 0.0

        marginSellBorrowAsset = None
        marginSellBorrowAmount = 0.0

        place_oco_result = {
            'transaction_id': transaction_id,
            'orders': [
                {'symbol': symbol, 'orderId': take_profit_order_id, 'type': _LIMIT_MAKER},
                {'symbol': symbol, 'orderId': stop_loss_order_id, 'type': _STOP_LOSS_LIMIT},
            ],
        }

        if side == SIDE_SELL:
            deficit = quantity - self.coin_free

            if deficit > 0:
                if sideEffectType in ("MARGIN_BUY", "AUTO_REPAY"):
                    self.coin_borrowed += deficit
                    self.coin_free += deficit

                    marginSellBorrowAsset = self.coin_asset
                    marginSellBorrowAmount = deficit
                else:
                    raise Exception("Not enough base asset")

            place_oco_result['coin_locked'] = quantity
            # if np.isclose(quantity, 1.292):
            #     pass

        if side == SIDE_BUY:
            required = quantity * stopLimitPrice  # worst case lock
            deficit = required - self.stable_free

            if deficit > 0:
                if sideEffectType in ("MARGIN_BUY", "AUTO_REPAY"):
                    self.stable_borrowed += deficit
                    self.stable_free += deficit

                    marginBuyBorrowAsset = self.stable_asset
                    marginBuyBorrowAmount = deficit
                else:
                    raise Exception("Not enough quote asset")

            place_oco_result['stable_locked'] = required

        # =========================
        # LOCK FUNDS
        # =========================
        if side == SIDE_SELL:
            coin_locked = quantity

            self.coin_free -= coin_locked
            self.coin_locked += coin_locked

        if side == SIDE_BUY:
            stable_locked = quantity * stopLimitPrice

            self.stable_free -= stable_locked
            self.stable_locked += stable_locked

        # =========================
        # CREATE ORDERS
        # =========================

        take_profit_order = {
            'symbol': symbol,
            'orderId': take_profit_order_id,
            'transaction_id': transaction_id,
            'price': price,
            'side': side,
            'type': _LIMIT_MAKER,
            'status': 'NEW',
            'quantity': quantity,
            'origQty': quantity,
            'timestamp': utc_now()
        }

        stop_loss_order = {
            'symbol': symbol,
            'orderId': stop_loss_order_id,
            'transaction_id': transaction_id,
            'price': stopLimitPrice,
            'side': side,
            'type': _STOP_LOSS_LIMIT,
            'status': 'NEW',
            'stopPrice': stopPrice,
            'stopLimitPrice': stopLimitPrice,
            'quantity': quantity,
            'origQty': quantity,
            'timestamp': utc_now()
        }

        place_oco_result['orderReports'] = [
            take_profit_order,
            stop_loss_order
        ]

        self.oco_orders.append(take_profit_order)
        self.oco_orders.append(stop_loss_order)

        self._recalc_net()

        # =========================
        # RESPONSE
        # =========================

        # attach borrow info ONLY if used
        if marginBuyBorrowAsset:
            place_oco_result["marginBuyBorrowAsset"] = marginBuyBorrowAsset
            place_oco_result["marginBuyBorrowAmount"] = str(marginBuyBorrowAmount)

        if marginSellBorrowAsset:
            place_oco_result["marginSellBorrowAsset"] = marginSellBorrowAsset
            place_oco_result["marginSellBorrowAmount"] = str(marginSellBorrowAmount)

        return place_oco_result

    def execute_oco_order_backtesting(self, place_oco_result):
        candle = _get_candle()

        filled_orders_count = 0

        if candle is None:
            return filled_orders_count

        take_profit, stop_loss = self.get_oco_orders(self, place_oco_result)

        if take_profit['status'] == 'FILLED' or stop_loss['status'] == 'FILLED':
            return filled_orders_count

        take_stop_order = self.counter % 2 == 0

        self.counter += 1

        if place_oco_result['transaction_type'] == _LONG:
            if take_stop_order:
                if candle['high'] >= take_profit['price']:
                    take_profit['status'] = 'FILLED'
                    stop_loss['status'] = 'EXPIRED'
                    quantity = take_profit['quantity']

                    self.execute_oco_order(SIDE_SELL, quantity, take_profit['price'], place_oco_result, profit_or_loss=True)
                    filled_orders_count += 1

                    return filled_orders_count

                if candle['low'] <= stop_loss['stopLimitPrice']:
                    stop_loss['status'] = 'FILLED'
                    take_profit['status'] = 'EXPIRED'
                    quantity = stop_loss['quantity']

                    self.execute_oco_order(SIDE_SELL, quantity, stop_loss['stopLimitPrice'], place_oco_result, profit_or_loss=False)
                    filled_orders_count += 1

                    return filled_orders_count
            else:
                if candle['low'] <= stop_loss['stopLimitPrice']:
                    stop_loss['status'] = 'FILLED'
                    take_profit['status'] = 'EXPIRED'
                    quantity = stop_loss['quantity']

                    self.execute_oco_order(SIDE_SELL, quantity, stop_loss['stopLimitPrice'], place_oco_result, profit_or_loss=False)
                    filled_orders_count += 1

                    return filled_orders_count

                if candle['high'] >= take_profit['price']:
                    take_profit['status'] = 'FILLED'
                    stop_loss['status'] = 'EXPIRED'
                    quantity = take_profit['quantity']

                    self.execute_oco_order(SIDE_SELL, quantity, take_profit['price'], place_oco_result, profit_or_loss=True)
                    filled_orders_count += 1

                    return filled_orders_count

        if place_oco_result['transaction_type'] == _SHORT:
            if take_stop_order:
                if candle['low'] <= take_profit['price']:
                    take_profit['status'] = 'FILLED'
                    stop_loss['status'] = 'EXPIRED'
                    quantity = take_profit['quantity']

                    self.execute_oco_order(SIDE_BUY, quantity, take_profit['price'], place_oco_result, profit_or_loss=True)
                    filled_orders_count += 1

                    return filled_orders_count

                if candle['high'] >= stop_loss['stopLimitPrice']:
                    stop_loss['status'] = 'FILLED'
                    take_profit['status'] = 'EXPIRED'
                    quantity = stop_loss['quantity']

                    self.execute_oco_order(SIDE_BUY, quantity, stop_loss['stopLimitPrice'], place_oco_result, profit_or_loss=False)
                    filled_orders_count += 1

                    return filled_orders_count
            else:
                if candle['high'] >= stop_loss['stopLimitPrice']:
                    stop_loss['status'] = 'FILLED'
                    take_profit['status'] = 'EXPIRED'
                    quantity = stop_loss['quantity']

                    self.execute_oco_order(SIDE_BUY, quantity, stop_loss['stopLimitPrice'], place_oco_result, profit_or_loss=False)
                    filled_orders_count += 1

                    return filled_orders_count

                if candle['low'] <= take_profit['price']:
                    take_profit['status'] = 'FILLED'
                    stop_loss['status'] = 'EXPIRED'
                    quantity = take_profit['quantity']

                    self.execute_oco_order(SIDE_BUY, quantity, take_profit['price'], place_oco_result, profit_or_loss=True)
                    filled_orders_count += 1

                    return filled_orders_count

            if filled_orders_count == 0:
                enter_price = place_oco_result['datetime_price']['price']
                enter_close_time = place_oco_result['datetime_price']['date_time']

                current_price = candle['close']
                current_close_time = candle['close_time']

                transaction_life_time = current_close_time - enter_close_time

                if transaction_life_time > TIME_DELTA(discretization=os.environ[_DISCRETIZATION]) * 32:
                    if check_env_true('LOG_TO_CONSOLE'):
                        printmd_populated_char_n_times(char="~", times=10, title=f"OUT OF LIFE TIME: {str(place_oco_result['transaction_id'])}", decorate="***")

                    quantity = take_profit['quantity']
                    if place_oco_result['transaction_type'] == _SHORT:
                        profit_or_loss = current_price < enter_price
                        self.execute_oco_order(SIDE_BUY, quantity, current_price, place_oco_result, profit_or_loss=profit_or_loss)
                    else:
                        profit_or_loss = current_price > enter_price
                        self.execute_oco_order(SIDE_SELL, quantity, current_price, place_oco_result, profit_or_loss=profit_or_loss)

                    if profit_or_loss:
                        take_profit['status'] = 'FILLED'
                        stop_loss['status'] = 'EXPIRED'
                    else:
                        stop_loss['status'] = 'FILLED'
                        take_profit['status'] = 'EXPIRED'

                    filled_orders_count += 1

        return filled_orders_count

    def execute_oco_order(self, side, quantity, price, place_oco_result, profit_or_loss):
        try:
            market_order = [
                o for o in self.market_orders
                if str(o['transaction_id']) == str(place_oco_result['transaction_id'])
            ][0]

            fee = self.maker_fee if profit_or_loss else self.taker_fee

            # =========================
            # BUY (closing short)
            # =========================
            if side == SIDE_BUY:
                stable_locked = place_oco_result['exit_oco_order']['stable_locked']

                # 1. UNLOCK
                self.stable_locked -= stable_locked
                self.stable_free += stable_locked

                # 2. EXECUTE
                quote_spent = quantity * price
                self.stable_free -= quote_spent
                self.coin_free += quantity

                # 3. COMMISSION (in base)
                coin_commission = quantity * fee
                self.coin_free -= coin_commission

                # 4. AUTO REPAY (base)
                repay = min(self.coin_free, self.coin_borrowed)
                self.coin_free -= repay
                self.coin_borrowed -= repay

            # =========================
            # SELL (closing long)
            # =========================
            elif side == SIDE_SELL:
                coin_locked = place_oco_result['exit_oco_order']['coin_locked']

                # 1. UNLOCK
                self.coin_locked -= coin_locked
                self.coin_free += coin_locked

                # 2. EXECUTE
                self.coin_free -= quantity
                quote_received = quantity * price
                self.stable_free += quote_received

                # 3. COMMISSION (in quote)
                stable_commission = quote_received * fee
                self.stable_free -= stable_commission

                # 4. AUTO REPAY (quote)
                repay = min(self.stable_free, self.stable_borrowed)
                self.stable_free -= repay
                self.stable_borrowed -= repay

            else:
                raise ValueError("Invalid side")

            # =========================
            # FINAL RECALC
            # =========================
            self._recalc_net()

            return {
                "status": "FILLED",
                "side": side,
                "price": price,
                "executedQty": quantity,
            }

        except Exception as ex:
            EXCEPTION_SPLITTED(
                f"side: {side} | quantity: {quantity} | price: {price} | place_oco_result: {place_oco_result}"
            )

    def repay_margin_loan(self, **params):
        print(f"~~~~~~~ NOT IMPLEMENTED ~~~~~~~")

    def cancel_margin_order(self, **params):
        symbol = params['symbol']
        is_isolated = params['isIsolated']
        order_id = params['orderId']

        for order in [o for o in self.oco_orders if o['orderId'] == order_id]:
            order['status'] = 'CANCELED'

    def _opened_transactions_count(self):
        from collections import defaultdict

        transaction_d = defaultdict(list)

        for item in self.oco_orders:
            transaction_d[item["transaction_id"]].append(item['status'])

        opened_transactions_count = sum(1 for v in transaction_d.values() if all(status == 'NEW' for status in v))

        return opened_transactions_count

    mock_client.init.side_effect = log_mock_calls(MethodType(init, mock_client), name='BINANCE MOCK', log_level='DEBUG')
    mock_client._recalc_net.side_effect = log_mock_calls(MethodType(_recalc_net, mock_client), name='BINANCE MOCK', log_level='DEBUG')
    mock_client._opened_transactions_count.side_effect = log_mock_calls(MethodType(_opened_transactions_count, mock_client), name='BINANCE MOCK', log_level='DEBUG')
    mock_client.transfer__SPOT__ISOLATED_MARGIN.side_effect = log_mock_calls(MethodType(transfer__SPOT__ISOLATED_MARGIN, mock_client), name='BINANCE MOCK', log_level='DEBUG')
    mock_client.transfer__ISOLATED_MARGIN__SPOT.side_effect = log_mock_calls(MethodType(transfer__ISOLATED_MARGIN__SPOT, mock_client), name='BINANCE MOCK', log_level='DEBUG')
    mock_client.get_exchange_info.side_effect = log_mock_calls(MethodType(get_exchange_info, mock_client), name='BINANCE MOCK', log_level='DEBUG')
    mock_client.futures_exchange_info.side_effect = log_mock_calls(MethodType(futures_exchange_info, mock_client), name='BINANCE MOCK', log_level='DEBUG')
    mock_client.get_symbol_ticker.side_effect = log_mock_calls(MethodType(get_symbol_ticker, mock_client), name='BINANCE MOCK', log_level='DEBUG')
    mock_client.get_margin_account.side_effect = log_mock_calls(MethodType(get_margin_account, mock_client), name='BINANCE MOCK', log_level='DEBUG')
    mock_client.get_isolated_margin_account.side_effect = log_mock_calls(MethodType(get_isolated_margin_account, mock_client), name='BINANCE MOCK', log_level='DEBUG')
    mock_client.get_max_margin_loan.side_effect = log_mock_calls(MethodType(get_max_margin_loan, mock_client), name='BINANCE MOCK', log_level='DEBUG')
    mock_client.get_open_margin_orders.side_effect = log_mock_calls(MethodType(get_open_margin_orders, mock_client), name='BINANCE MOCK', log_level='DEBUG')
    mock_client.get_margin_order.side_effect = log_mock_calls(MethodType(get_margin_order, mock_client), name='BINANCE MOCK', log_level='DEBUG')
    mock_client.create_margin_order.side_effect = log_mock_calls(MethodType(create_margin_order, mock_client), name='BINANCE MOCK', log_level='NOTICE')
    mock_client.create_margin_oco_order.side_effect = log_mock_calls(MethodType(create_margin_oco_order, mock_client), name='BINANCE MOCK', log_level='NOTICE')
    mock_client.execute_oco_order_backtesting.side_effect = log_mock_calls(MethodType(execute_oco_order_backtesting, mock_client), name='BINANCE MOCK', log_level='DEBUG')
    mock_client.execute_oco_order.side_effect = log_mock_calls(MethodType(execute_oco_order, mock_client), name='BINANCE MOCK', log_level='NOTICE')
    mock_client.repay_margin_loan.side_effect = log_mock_calls(MethodType(repay_margin_loan, mock_client), name='BINANCE MOCK', log_level='DEBUG')
    mock_client.cancel_margin_order.side_effect = log_mock_calls(MethodType(cancel_margin_order, mock_client), name='BINANCE MOCK', log_level='NOTICE')

    return mock_client