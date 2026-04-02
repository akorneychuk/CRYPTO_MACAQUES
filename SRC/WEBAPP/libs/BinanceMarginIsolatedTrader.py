import os
import os
import sys
import time
import traceback
from datetime import datetime
from threading import Thread
from unittest.mock import MagicMock

import binance.exceptions
from binance.enums import SIDE_BUY, ORDER_TYPE_MARKET, SIDE_SELL, TIME_IN_FORCE_GTC
from filelock import FileLock

import SRC.LIBRARIES.binance_helpers as binance_helpers
from SRC.CORE._CONFIGS import get_config
from SRC.CORE._CONSTANTS import _BALANCE_FILE_PATH, NO_CHANGE_RATIO, RUN_TO_END, PART_OFFSET, _STOP_ON_JUMP_UP_RATIO, _STOP_ON_DROP_DOWN_RATIO, _MAX_ORDERS_COUNT, _SYMBOL_SLASH, \
	_LIMIT_MAKER, _STOP_LOSS_LIMIT, USE_PROXY_CLIENT, TRANSACTION_RETRY_COUNT, BINANCE_TAKER_COMISSION, BINANCE_MAKER_COMISSION, _AUTOMATION_TYPE, _TRANSFER_CROSS_ISOLATED, _LONG, \
	_SHORT, _AUTOTRADING_REGIME, _MARGIN, _WRITE_IGNORE_BALANCES, _PROD, _MOCK, _TRADES_ERRORS_FILE_PATH, _OCO_ORDER_CHECKER_LOCK_FILE_PATH, LEVERAGE, IGNORE_SEPARATOR, POSITION_SEPARATOR, ALERT_SEPARATOR, EXCEPTION_SEPARATOR, _symbol, \
	_autotrading_regime, _market_type, _presentation_type, _DASHBOARD, _PART
from SRC.CORE.debug_utils import ERROR as ERR, CONSOLE_SPLITTED, log_context, EXCEPTION_SPLITTED, ERROR_SPLITTED, is_autotrading, is_backtesting, produce_formatters, DEBUG_SPLITTED, is_running_under_pycharm_debugger, NOTICE_SPLITTED
from SRC.CORE.utils import _float_n, write_json, datetime_h_m_s__d_m_Y, get_item_from_list_dict, read_json_safe, _float_5
from SRC.LIBRARIES.binance_helpers import get_isolated_assets
from SRC.LIBRARIES.binance_helpers import get_tick_step_info_spot, get_exchange_info_time_cached, get_margin_symbol_ticker_current_price
from SRC.LIBRARIES.new_utils import floor, is_close_to_zero, top_up_net_folder, get_round_order, get_pretty_datetime_price, is_no_trades_timeout, \
	is_over_timeout, check_env_true, _symbol_join, get_datetime_price, tryall_delegate, env_string, timed_cache, append_file, lock_with_file, filter_dict, calculate_weighted_average_price, env_int
from SRC.LIBRARIES.new_utils import print_log_trades as _print_log_trades
from SRC.LIBRARIES.new_utils import printmd_log_trades as _printmd_log_trades
from SRC.LIBRARIES.time_utils import TIME_DELTA, round_down_to_nearest_step, kiev_now, as_kiev_tz
from SRC.WEBAPP.libs.exceptions import ExitAutomationError, AutotradingSessionNotStartedError

print_log_trades = lambda *msg_log_s: _print_log_trades(*msg_log_s, to_console=check_env_true('LOG_TO_CONSOLE'), to_file=True)
printmd_log_trades = lambda *msg_log_s: _printmd_log_trades(*msg_log_s, to_console=check_env_true('LOG_TO_CONSOLE'), to_file=True)

if USE_PROXY_CLIENT():
	pass

_bi, _b, nl_ = produce_formatters(mode=None)


def trade_utils_producer(client, start_autotrading_data, ext_log_s=None):
	with log_context(CONSOLE_SPLITTED) as log_s:
		log_s = log_s if ext_log_s is None else ext_log_s

		symbol_slash = os.environ[_SYMBOL_SLASH]
		autotrading_regime = env_string(_AUTOTRADING_REGIME, _MOCK)
		automation_type = env_string(_AUTOMATION_TYPE)

		coin_asset = symbol_slash.split("/")[0]
		stable_asset = symbol_slash.split("/")[1]
		fee_asset = 'BNB'
		symbol = f'{coin_asset}{stable_asset}'

		stop_loss_stop_ratio = 1.0005

		run_to_end = RUN_TO_END()

		if is_backtesting():
			stop_on_jump_up_ratio = float(os.environ[_STOP_ON_JUMP_UP_RATIO])

		stop_on_drop_down_ratio = float(os.environ[_STOP_ON_DROP_DOWN_RATIO])

		part = env_int(_PART, None)
		max_positions_count = env_int(_MAX_ORDERS_COUNT, 3)

		if part:
			part_details = f"int({part} / {max_positions_count + PART_OFFSET()} |  [part / (max_positions_count max_positions_count + PART_OFFSET())]"
			margin_isolated_global_part = int(part / (max_positions_count + PART_OFFSET()))
		else:
			part_details = "CONFIG [margin_isolated_global_part]"
			margin_isolated_global_part = get_config("automation_dashboard_app.autotrade.margin_isolated_global_part")

		tickers = binance_helpers.get_active_coin_usdt_tickers()
		spot_balance = 10_000
		initial_balance = 1000

		if is_autotrading():
			if autotrading_regime == _PROD:
				dept_getter = get_isolated_assets(client, symbol, print_out=False)
				stable_net, stable_free, stable_borrowed, stable_locked = dept_getter['stable']()
				if stable_net > initial_balance:
					initial_balance = stable_net
					initial_balance_details = f"EXISTING"
				else:
					initial_balance_details = f"CALCULATED"
			else:
				total_net_dict_s = read_json_safe(_BALANCE_FILE_PATH(), [])
				if len(total_net_dict_s) > 0:
					initial_balance = round(total_net_dict_s[-1]['balance'], 2)
					initial_balance_details = f"EXISTING"
				else:
					initial_balance_details = f"CALCULATED"
		else:
			initial_balance = 1000
			initial_balance_details = "HARDCODED"

		if isinstance(client, MagicMock):
			client.init()

		sym_join = _symbol_join(symbol)
		try:
			exchange_info = get_exchange_info_time_cached()
			symbol_info = binance_helpers.get_exchange_symbol_info(symbol, exchange_info)
			price_tick_size, quantity_step_size, min_stable_allowed = get_tick_step_info_spot(symbol_info)
		except:
			exchange_info = get_exchange_info_time_cached()
			pair = [k for k in exchange_info['symbols'] if k['symbol'] == symbol or k['symbol'] == sym_join][0]
			price_tick_size, quantity_step_size, min_stable_allowed = get_tick_step_info_spot(pair)

		price_round_order = get_round_order(price_tick_size)
		quantity_round_order = get_round_order(quantity_step_size)
		leverage = LEVERAGE()

		price_tick_size_str = "{:.15f}".format(price_tick_size).rstrip('0').rstrip('.') if '.' in "{:.15f}".format(price_tick_size) else "{:.15f}".format(price_tick_size)

		if is_autotrading() and env_string(_AUTOTRADING_REGIME) == _PROD:
			fee_info = f"!! BINANCE PRODUCTION FEE SETUP !!"
		else:
			fee_info = f"TAKER FEE: {round(BINANCE_TAKER_COMISSION(), 5)} ({round(BINANCE_TAKER_COMISSION() * 100, 5)}%) | MAKER FEE: {round(BINANCE_MAKER_COMISSION(), 5)} ({round(BINANCE_MAKER_COMISSION() * 100, 5)}%)"

		log_s.extend([
			f"TIME: {datetime_h_m_s__d_m_Y(datetime.now())}",
			fee_info,
			f'PRICE TICK: {price_tick_size_str} | QUANTITY STEP: {quantity_step_size} | MIN STABLE ALLOWED: {min_stable_allowed}',
			f'LEVERAGE: {leverage} | MAX POSITIONS: {max_positions_count} | PART OFFSET: {PART_OFFSET()}',
			f'SPOT BALANCE: {_float_5(spot_balance)}',
			f'INITIAL BALANCE: {initial_balance} [{initial_balance_details}] | GLOBAL SPOT PART: {margin_isolated_global_part} [{part_details}]'
		])

		max_position_stable = (initial_balance / (max_positions_count + PART_OFFSET()))
		if is_autotrading() and max_position_stable < min_stable_allowed:
			session_identifier = get_unique_session_identifier(start_autotrading_data)
			raise AutotradingSessionNotStartedError(
				f'SIGNAL CONSUMER: {session_identifier}',
				start_autotrading_data,
				f"EXIT AUTOTRADING | {max_position_stable} [MAX POSITION STABLE] < {min_stable_allowed} [MIN STABLE ALLOWED]")

		def get_fee_asset(print_out=True):
			acct = client.get_margin_account()
			fee_asset_info = [k for k in acct['userAssets'] if k['asset'] == 'BNB'][0]

			net = fee_asset_info['netAsset']

			if print_out:
				print_log_trades(f'FEE {"BNB"} | net: {net}')

			return float(net)

		def order_status_filled(client, order, condition=lambda order_status: order_status == "FILLED"):
			symbol = order['symbol']
			oid = order['orderId']
			order_check = client.get_margin_order(symbol=symbol, orderId=oid, isIsolated=True)
			order_status = order_check['status']

			return condition(order_status)

		def get_oco_orders(client, place_oco_result):
			oco_orders = [client.get_margin_order(symbol=order['symbol'], orderId=order['orderId'], isIsolated=True) for order in place_oco_result['exit_oco_order']['orders']]

			take_profit_order = next(filter(lambda order: order['type'] == _LIMIT_MAKER, oco_orders))
			stop_loss_order = next(filter(lambda order: order['type'] == _STOP_LOSS_LIMIT, oco_orders))

			return take_profit_order, stop_loss_order

		def get_max_borrow_asset(asset):
			max_borrow = float(client.get_max_margin_loan(asset=asset, isolatedSymbol=symbol, isIsolated=True)['amount'])
			max_borrow_safe = max_borrow * 2 / 3

			return max_borrow_safe

		def cancel_all_active_oco_orders(symbol):
			open_orders = client.get_open_margin_orders(symbol=symbol, isIsolated='True')
			for open_order in open_orders:
				try:
					client.cancel_margin_order(symbol=symbol, isIsolated=True, orderId=open_order['orderId'])
				except Exception as ex:
					pass

		def borrow_coin(quantity):
			client.create_margin_loan(asset=coin_asset, amount=str(quantity), isIsolated='True', symbol=symbol)

		def repay_coin(quantity):
			client.repay_margin_loan(asset=coin_asset, amount=str(quantity), isIsolated='True', symbol=symbol)

		def borrow_stable(quantity):
			client.create_margin_loan(asset=stable_asset, amount=str(quantity), isIsolated='True', symbol=symbol)

		def repay_stable(quantity):
			client.repay_margin_loan(asset=stable_asset, amount=str(quantity), isIsolated='True', symbol=symbol)

		def transfer_spot__isolated_margin(quantity, asset=stable_asset):
			try:
				#https://developers.binance.com/docs/wallet/asset/user-universal-transfer
				client.transfer__SPOT__ISOLATED_MARGIN(asset=asset, amount=quantity, toSymbol=symbol)
				print_log_trades(f'Transferred SPOT > ISOLATED MARGIN | asset = {asset} | toSymbol = {symbol} | amount = {quantity}')
			except:
				traceback.print_exc()
				sys.stdout.flush()
				raise Exception()

		def transfer_isolated_margin__spot(quantity, asset='USDT'):
			try:
				#https://developers.binance.com/docs/wallet/asset/user-universal-transfer
				client.transfer__ISOLATED_MARGIN__SPOT(asset=asset, amount=quantity, fromSymbol=symbol)
				print_log_trades(f'Transferred ISOLATED > SPOT | asset = {asset} | fromSymbol = {symbol} | amount = {quantity}')
			except:
				traceback.print_exc()
				sys.stdout.flush()
				raise Exception()

		def get_side_positions_count(side=None):
			open_orders = client.get_open_margin_orders(symbol=symbol)
			if side is None:
				return len(open_orders)

			side_oco_oders = list(filter(lambda order: order['side'] == side, open_orders))
			side_positions_count = int(len(side_oco_oders) / 2)

			return side_positions_count

		def get_total_net_USDT():
			datetime_price = get_datetime_price()
			price = datetime_price['price']
			total_net_USDT = binance_helpers.get_margin_isolated_total_net_USDT(client, symbol, price)

			return total_net_USDT

		def present_total_net_USDT(total_net_USDT):
			return f'TOTAL {"USDT"} | balance: ${_float_n(total_net_USDT, 5)}'

		def get_present_total_net_USDT():
			total_net_USDT = get_total_net_USDT()
			total_net_USDT_present = present_total_net_USDT(total_net_USDT)

			return total_net_USDT_present

		def calculate_write_balances_present_net_total(entry_datetime_price, exit_datetime_price, transaction_id, correlation_id, transaction_type=None, transaction_result=None):
			with FileLock(_BALANCE_FILE_PATH().replace('json', 'lock'), timeout=5):
				total_net = get_total_net_USDT()
				total_net_present = present_total_net_USDT(total_net)
				total_net_dict_s = read_json_safe(_BALANCE_FILE_PATH(), [])

				entry_transaction_dt_utc = round_down_to_nearest_step(entry_datetime_price['date_time'], TIME_DELTA('1S'))
				exit_transaction_dt_utc = round_down_to_nearest_step(exit_datetime_price['date_time'], TIME_DELTA('1S'))

				if is_backtesting():
					if len(total_net_dict_s) > 0 and (total_net_dict_s[-1]['balance'] == total_net or ('date_time' in total_net_dict_s[-1] and total_net_dict_s[-1]['date_time'] == exit_transaction_dt_utc)):
						return total_net_present

				transaction_type = transaction_type if transaction_type is not None else transaction_id
				transaction_result = transaction_result if transaction_result is not None else transaction_id

				total_net_dict_s.append({
					'entry_date_time': entry_transaction_dt_utc,
					'exit_date_time': exit_transaction_dt_utc,
					'date_time': exit_transaction_dt_utc,
					'transaction_id': str(transaction_id),
					'correlation_id': str(correlation_id),
					'transaction_type': transaction_type,
					'transaction_result': transaction_result,
					'balance': total_net
				})
				write_json(total_net_dict_s, _BALANCE_FILE_PATH())

				return total_net_present

		def align_balances(full_print=False, ext_log_s=None):
			with log_context(CONSOLE_SPLITTED) as log_s:
				log_s = log_s if ext_log_s is None else ext_log_s

				price = get_margin_symbol_ticker_current_price(symbol)

				log_s.append(f"ALIGN BALANCES")
				log_s.append(f"CURRENT STATE:")

				dept_getter = get_isolated_assets(client, symbol, print_out=full_print or True, log_s=log_s)
				stable_net, stable_free, stable_borrowed, stable_locked = dept_getter['stable']()
				coin_net, coin_free, coin_borrowed, coin_locked = dept_getter['coin']()

				cancel_all_active_oco_orders(symbol=symbol)
				if 0 < coin_borrowed <= coin_free:
					repay_coin(coin_borrowed)
					dept_getter = get_isolated_assets(client, symbol, print_out=full_print or True, log_s=log_s)
					coin_net, coin_free, coin_borrowed, coin_locked = dept_getter['coin']()

				if 0 < stable_borrowed <= stable_free:
					repay_stable(stable_borrowed)
					dept_getter = get_isolated_assets(client, symbol, print_out=full_print or True, log_s=log_s)
					stable_net, stable_free, stable_borrowed, stable_locked = dept_getter['stable']()

				if coin_free * price <= 1 and is_close_to_zero(stable_borrowed, 0.1) and is_close_to_zero(coin_borrowed * price, 0.1):
					return

				changed = False
				coin_stable_net = abs(floor(coin_net, get_round_order(quantity_step_size)) * price)
				if coin_net < 0:
					coin_buy = (coin_stable_net if coin_stable_net > min_stable_allowed else min_stable_allowed) / price
					coin_to_buy = floor(coin_buy * 1.005, get_round_order(quantity_step_size))
					client.create_margin_order(
						symbol=symbol,
						side=SIDE_BUY,
						type=ORDER_TYPE_MARKET,
						quantity=coin_to_buy,
						isIsolated='True',
						sideEffectType="AUTO_REPAY"
					)

					changed = True

				if coin_net > 0 and coin_stable_net > min_stable_allowed:
					coin_to_sell = floor(coin_net, get_round_order(quantity_step_size))
					client.create_margin_order(
						symbol=symbol,
						side=SIDE_SELL,
						type=ORDER_TYPE_MARKET,
						quantity=coin_to_sell,
						isIsolated='True',
						sideEffectType="AUTO_REPAY"
					)

					changed = True

				time.sleep(1)

				log_s.append(f"REVERSED STATE:")

				dept_getter = get_isolated_assets(client, symbol, print_out=full_print or False, log_s=log_s)
				stable_net, stable_free, stable_borrowed, stable_locked = dept_getter['stable']()
				coin_net, coin_free, coin_borrowed, coin_locked = dept_getter['coin']()

				if coin_borrowed > 0:
					repay_coin(coin_borrowed)
					changed = True

				if stable_borrowed > 0:
					repay_stable(stable_borrowed)
					changed = True

				if changed:
					print_log_trades('---------------------------------------------------------------------------')

					time.sleep(1)

					log_s.append(f"FINAL STATE:")

					dept_getter = get_isolated_assets(client, symbol, print_out=full_print or True, log_s=log_s)
					stable_net, stable_free, stable_borrowed, stable_locked = dept_getter['stable']()
					coin_net, coin_free, coin_borrowed, coin_locked = dept_getter['coin']()

		def calculate_initial_balance(date_time_utc):
			last_dt_UTC = round_down_to_nearest_step(date_time_utc, TIME_DELTA('5S'))
			balance_s = read_json_safe(_BALANCE_FILE_PATH(), [])
			if len(balance_s) > 0:
				last_balance_UTC = balance_s[-1]['date_time']
				datetime_price = {'date_time': last_dt_UTC}

				if last_dt_UTC > last_balance_UTC:
					net_total_present = calculate_write_balances_present_net_total(entry_datetime_price=datetime_price, exit_datetime_price=datetime_price, transaction_id='UNKNOWN', correlation_id='UNKNOWN', transaction_type='INITIAL')
					print_log_trades(net_total_present)
			else:
				datetime_price={'date_time': last_dt_UTC}
				net_total_present = calculate_write_balances_present_net_total(entry_datetime_price=datetime_price, exit_datetime_price=datetime_price, transaction_id='UNKNOWN', correlation_id='UNKNOWN', transaction_type='INITIAL')
				print_log_trades(net_total_present)

		def finalize():
			try:
				dept_getter = get_isolated_assets(client, symbol, print_out=False)
				stable_net, stable_free, stable_borrowed, stable_locked = dept_getter['stable']()
				transfer_isolated_margin__spot(stable_net, 'USDT')
			except Exception:
				ERR(f'NO ASSET in ISOLATED: {symbol}')

		def initialize(started_dt_utc, initial_balance=5, ext_log_s=None):
			try:
				dept_getter = get_isolated_assets(client, symbol, print_out=False)
				stable_net, stable_free, stable_borrowed, stable_locked = dept_getter['stable']()
			except IndexError:
				transfer_spot__isolated_margin(1)
				dept_getter = get_isolated_assets(client, symbol, print_out=False)
				stable_net, stable_free, stable_borrowed, stable_locked = dept_getter['stable']()

			if check_env_true(_TRANSFER_CROSS_ISOLATED):
				if stable_net < initial_balance:
					transfer_amount = initial_balance - stable_net
					transfer_spot__isolated_margin(transfer_amount)

			if is_autotrading():
				align_balances(ext_log_s=ext_log_s)

			calculate_initial_balance(started_dt_utc)

		if isinstance(client, MagicMock):
			client.get_total_net_USDT = get_total_net_USDT
			client.get_oco_orders = get_oco_orders

		def calc_oco_sell_quantity(borrow_buy_result, oco_sell_deals_count, coin_net):
			actual_earned_coins = sum([float(fill['qty']) - float(fill['commission']) for fill in borrow_buy_result['fills']])
			round_order = get_round_order(quantity_step_size)
			if oco_sell_deals_count == 0 and coin_net > 0:
				oco_sell_quantity = round(floor(coin_net, round_order), round_order)
			else:
				oco_sell_quantity = round(floor(actual_earned_coins, round_order), round_order)

			return oco_sell_quantity

		def calc_oco_buy_quantity(borrow_sell_result):
			round_order = get_round_order(quantity_step_size)
			oco_buy_quantity = round(floor(sum([float(fill['qty']) for fill in borrow_sell_result['fills']]), round_order), round_order)

			return oco_buy_quantity

		def _try_oco_executed_print(log_s, oco_result):
			correlation_id = oco_result['correlation_id']
			transaction_id = oco_result['transaction_id']
			transaction_type = oco_result['transaction_type']
			transaction_result = oco_result['transaction_result']
			entry_order = oco_result['entry_order']
			exit_order = oco_result['exit_order'] if 'exit_order' in oco_result else None
			exit_oco_order = oco_result['exit_oco_order']
			entry_datetime_price = oco_result['entry_datetime_price']
			exit_datetime_price = oco_result['exit_datetime_price']

			quantity = exit_oco_order['quantity']
			exit_side = exit_oco_order['side']
			exit_order_id = exit_order['orderId'] if exit_order else oco_result['exit_order_id']
			exit_price = oco_result['exit_price']
			entry_price = calculate_weighted_average_price(entry_order['fills'])

			logger = (CONSOLE_SPLITTED if is_autotrading() else DEBUG_SPLITTED) if len(log_s) == 0 else lambda *msg: log_s.extend(msg)
			datetime_price = get_datetime_price()
			net_total_present = get_present_total_net_USDT()

			log_separator = f"{POSITION_SEPARATOR} transaction_id: {transaction_id} || correlation_id: {correlation_id} {POSITION_SEPARATOR}"
			log_separator_thin = '---------------------------------------------------------------------------'

			logs = [
				f"{_bi(transaction_result)}|{_bi(transaction_type)}: [{symbol} | {autotrading_regime} | {_MARGIN} | transaction_id: {transaction_id} || correlation_id: {correlation_id}]",
				f"QTY: {quantity} | {get_pretty_datetime_price(datetime_price)} | exit_order_id: {exit_order_id}",
				f"PRICE: {datetime_price['price']} | DATETIME: {as_kiev_tz(datetime_price['date_time'])} | {net_total_present}"
			]

			print_log_trades(log_separator)
			print_log_trades(*logs)
			print_log_trades(log_separator_thin)

			state_getter = get_isolated_assets(client, symbol, print_out=True)
			stable_net, stable_free, stable_borrowed, stable_locked = state_getter['stable']()
			coin_net, coin_free, coin_borrowed, coin_locked = state_getter['coin']()

			print_log_trades(log_separator_thin)
			print_log_trades(log_separator)

			net_total_present = calculate_write_balances_present_net_total(entry_datetime_price, exit_datetime_price, transaction_id, correlation_id, transaction_type, transaction_result)
			print_log_trades(net_total_present)
			top_up_net_folder()

			logger(*logs)

		def oco_checker(place_oco_result):
			if 'transaction_result' in place_oco_result:
				return True

			transaction_id = place_oco_result['transaction_id']
			correlation_id = place_oco_result['correlation_id']

			datetime_price = place_oco_result['datetime_price']
			datetime_price_now = get_datetime_price()

			transaction_type = place_oco_result['transaction_type']

			take_profit_order, stop_loss_order = get_oco_orders(client, place_oco_result)

			is_take_profit_order_canceled = take_profit_order['status'] == 'CANCELED'
			is_stop_loss_order_canceled = stop_loss_order['status'] == 'CANCELED'

			if is_take_profit_order_canceled and is_stop_loss_order_canceled:
				transaction_result = "FORCE CLOSED POSITION"

				quantity = take_profit_order['origQty']
				price = datetime_price_now['price']
				order_id = 'WEB_ORDER_ID'

				place_oco_result['exit_price'] = price
				place_oco_result['exit_quantity'] = quantity
				place_oco_result['exit_order_id'] = order_id
				place_oco_result['exit_datetime_price'] = datetime_price
				place_oco_result['transaction_result'] = transaction_result

				_try_oco_executed_print([], place_oco_result)

				return True

			is_take_profit_order_filled = take_profit_order['status'] == 'FILLED'
			is_stop_loss_order_filled = stop_loss_order['status'] == 'FILLED'

			if is_take_profit_order_filled or is_stop_loss_order_filled:
				transaction_result = "PROFIT" if is_take_profit_order_filled else "LOSS"

				exit_order = take_profit_order if is_take_profit_order_filled else stop_loss_order
				quantity = exit_order['origQty']
				price = exit_order['price']
				order_id = exit_order['orderId']

				place_oco_result['exit_price'] = price
				place_oco_result['exit_quantity'] = quantity
				place_oco_result['exit_order'] = exit_order
				place_oco_result['exit_datetime_price'] = datetime_price
				place_oco_result['transaction_result'] = transaction_result

				_try_oco_executed_print([], place_oco_result)

				return True

			return False

		@lock_with_file(lockfile=_OCO_ORDER_CHECKER_LOCK_FILE_PATH(), timeout=15 if not is_running_under_pycharm_debugger() else 300)
		def oco_checker_autotrading(place_oco_result):
			return oco_checker(place_oco_result)

		def no_action(datetime_price, transaction_id, correlation_id, transaction_type, transaction_result):
			logger = CONSOLE_SPLITTED if is_autotrading() else DEBUG_SPLITTED

			if is_autotrading() and transaction_type != 'IGNORE' or check_env_true(_WRITE_IGNORE_BALANCES, False):
				transaction_result_present = transaction_result if isinstance(transaction_result, str) else "|".join(transaction_result)
				net_total_present = calculate_write_balances_present_net_total(datetime_price, datetime_price, transaction_id, correlation_id, transaction_type, transaction_result_present)
			else:
				net_total_present = get_present_total_net_USDT()

			separator = IGNORE_SEPARATOR
			if transaction_type == 'WARNING':
				separator = ALERT_SEPARATOR
			if transaction_type == 'ERROR':
				separator = EXCEPTION_SEPARATOR

			log_separator = f'{separator} [{kiev_now().isoformat()}] transaction_id: {transaction_id} | correlation_id: {correlation_id} {separator}'
			logs = [
				f"{transaction_type}: [{symbol} | {autotrading_regime} | {_MARGIN} | transaction_id: {transaction_id} | correlation_id: {correlation_id}]:",
				f"PRICE: {datetime_price['price']} | DATETIME: {as_kiev_tz(datetime_price['date_time'])} | {net_total_present}"
			]

			if transaction_type == 'IGNORE' and transaction_result == 'IGNORE':
				pass
			else:
				if isinstance(transaction_result, list):
					logs.extend([f"CAUSE: {result}" for result in transaction_result])
				else:
					logs.append(f"CAUSE: {transaction_result}")

			print_log_trades(*[log_separator, *logs, log_separator])
			logger(*logs)

		def IGNORE(datetime_price, transaction_id, correlation_id, cause_present='IGNORE'):
			no_action(datetime_price, transaction_id, correlation_id, 'IGNORE', cause_present)

		def ERROR(datetime_price, transaction_id, correlation_id, error_present):
			no_action(datetime_price, transaction_id, correlation_id, 'ERROR', error_present)

		def WARNING(datetime_price, transaction_id, correlation_id, error_present):
			no_action(datetime_price, transaction_id, correlation_id, 'WARNING', error_present)

		def _try_market_action_print(log_s, side, qty, price, transaction_id, correlation_id, order_id_title='XXXXXXXX', error_title=''):
			stable_used = qty * price

			logs = [
				f"{error_title}ENTERED [{side}]: [{symbol} | {autotrading_regime} | {_MARGIN} | transaction_id: {transaction_id} || correlation_id: {correlation_id}]:",
				f"ENTRY PRICE: {_float_n(price, 5)} | STABLE: {_float_n(stable_used, 5)} | QTY: {qty}"
			]

			print_log_trades(*logs)
			log_s.extend(logs)

		def _try_oco_action_print(log_s, side, qty, price, take_profit_price, stop_loss_price, take_profit_ratio, stop_loss_ratio, transaction_id, correlation_id, order_id_title='XXXXXXXX', error_title=''):
			stable_used = qty * price

			logs = [
				f"{error_title}OCO [{side}]: [{symbol} | {autotrading_regime} | {_MARGIN} | transaction_id: {transaction_id} || correlation_id: {correlation_id}]:",
				f"PROFIT PRICE: {_float_5(take_profit_price)} | LOSS PRICE: {_float_5(stop_loss_price)} | TPR: {_float_5(take_profit_ratio)} | SLR: {_float_5(stop_loss_ratio)}"
			]

			print_log_trades(*logs)
			log_s.extend(logs)

		def _record_position_error(log_s, position_side, transaction_id, correlation_id, exception):
			append_file(_TRADES_ERRORS_FILE_PATH(), f"{transaction_id}|{correlation_id}|{str(exception)}")

			log_separator_thin = '---------------------------------------------------------------------------'

			logs = [
				f'!!!! ERROR POSITION [{position_side}]: correlation_id: {transaction_id} || correlation_id: {correlation_id} EXCEPTION BELOW !!!!'
			]

			state_getter = get_isolated_assets(client, symbol, print_out=True, log_s=logs)
			stable_net, stable_free, stable_borrowed, stable_locked = state_getter['stable']()
			coin_net, coin_free, coin_borrowed, coin_locked = state_getter['coin']()

			logs.append(traceback.format_exc())
			logs.append(EXCEPTION_SPLITTED)

			print_log_trades(*[log_separator_thin, *logs, log_separator_thin])
			log_s.extend(logs)

		def LONG(entry_precheck, take_profit_ratio, stop_loss_ratio, transaction_id, correlation_id):
			logger = CONSOLE_SPLITTED if is_autotrading() else DEBUG_SPLITTED

			can_execute_order = entry_precheck['can_execute_order']
			part = entry_precheck['part']
			datetime_price = entry_precheck['datetime_price']
			max_stable_borrowable = entry_precheck['max_stable_borrowable']
			oco_sell_deals_count = entry_precheck['oco_sell_deals_count']
			oco_buy_deals_count = entry_precheck['oco_buy_deals_count']
			cause_s = entry_precheck['cause_s']

			x = 1 / part

			with log_context(logger) as log_s:
				part_pesent = f'1/{round(1 / x)}' if x < 1 else ''
				position_side_log_title = f'{_b("LONG IN")}: {part_pesent}x{leverage} | {get_pretty_datetime_price(datetime_price)} || {get_present_total_net_USDT()}'

				log_s.append(position_side_log_title)
				printmd_log_trades(f"{POSITION_SEPARATOR} transaction_id: {transaction_id} || correlation_id: {correlation_id} {POSITION_SEPARATOR}")
				printmd_log_trades(position_side_log_title)

				state_getter = get_isolated_assets(client, symbol, print_out=True)

				stable_net, stable_free, stable_borrowed, stable_locked = state_getter['stable']()
				coin_net, coin_free, coin_borrowed, coin_locked = state_getter['coin']()

				print_log_trades('---------------------------------------------------------------------------')

				side = SIDE_BUY
				buy_price = datetime_price['price']
				max_stable_borrowable = tryall_delegate(lambda: get_max_borrow_asset(stable_asset), tryalls_count=3)
				stable_not_available = coin_locked * buy_price
				stable_use = (max_stable_borrowable - stable_not_available) * x
				buy_quantity = floor(stable_use / buy_price, quantity_round_order)
				borrow_buy_result = None

				try:
					def wrap_margin_buy(tryall=1):
						try:
							borrow_buy_result = client.create_margin_order(
								symbol=symbol,
								side=side,
								type=ORDER_TYPE_MARKET,
								quantity=buy_quantity,
								isIsolated='True',
								sideEffectType="MARGIN_BUY",
								transaction_id=transaction_id
							)
							order_id_title = borrow_buy_result['orderId']
							_try_market_action_print(log_s, side, buy_quantity, buy_price, transaction_id, correlation_id, order_id_title=order_id_title)

							return borrow_buy_result, stable_net, stable_free, stable_borrowed, stable_locked, coin_net, coin_free, coin_borrowed, coin_locked
						except binance.exceptions.BinanceAPIException as binanceEx:
							if binanceEx.code != -3044 or tryall > TRANSACTION_RETRY_COUNT():
								_try_market_action_print(log_s, side, buy_quantity, buy_price, transaction_id, correlation_id, error_title=f'TRIED [{tryall}] > ')
								raise

							time.sleep(1)
							return wrap_margin_buy(tryall + 1)

					borrow_buy_result, stable_net, stable_free, stable_borrowed, stable_locked, coin_net, coin_free, coin_borrowed, coin_locked = wrap_margin_buy()

					while True:
						state_getter = get_isolated_assets(client, symbol, print_out=True)
						_stable_net, _stable_free, _stable_borrowed, _stable_locked = state_getter['stable']()
						_coin_net, _coin_free, _coin_borrowed, _coin_locked = state_getter['coin']()
						if (coin_net == _coin_net or coin_free == _coin_free) and (stable_net == _stable_net or stable_free == _stable_free):
							print_log_trades('----------------------------- RETRY ---------------------------------------')
							time.sleep(0.1)
							continue
						else:
							stable_net, stable_free, stable_borrowed, stable_locked = _stable_net, _stable_free, _stable_borrowed, _stable_locked
							coin_net, coin_free, coin_borrowed, coin_locked = _coin_net, _coin_free, _coin_borrowed, _coin_locked
							print_log_trades('---------------------------------------------------------------------------')
							break

					def wrap_margin_oco_sell(tryall=1):
						side = SIDE_SELL
						sell_price = get_datetime_price()['price']
						sell_quantity = calc_oco_sell_quantity(borrow_buy_result, oco_sell_deals_count, _coin_net)

						take_profit_price = round(sell_price * take_profit_ratio, price_round_order)
						stop_price = floor(sell_price / stop_loss_ratio, price_round_order)
						stop_loss_price = floor(stop_price / stop_loss_stop_ratio, price_round_order)

						try:
							sell_repay_result = client.create_margin_oco_order(
								symbol=symbol,
								side=side,
								quantity=sell_quantity,
								price=take_profit_price,
								stopPrice=stop_price,
								stopLimitPrice=stop_loss_price,
								isIsolated='True',
								sideEffectType="AUTO_REPAY",
								stopLimitTimeInForce=TIME_IN_FORCE_GTC,
								transaction_id=transaction_id
							)
							profit_order = get_item_from_list_dict(sell_repay_result['orderReports'], 'type', _LIMIT_MAKER)
							loss_order = get_item_from_list_dict(sell_repay_result['orderReports'], 'type', _STOP_LOSS_LIMIT)
							order_id_title = f"{profit_order['orderId']} | {loss_order['orderId']}"
							_try_oco_action_print(log_s, side, sell_quantity, sell_price, take_profit_price, stop_loss_price, take_profit_ratio, stop_loss_ratio, transaction_id, correlation_id, order_id_title=order_id_title)

							state_getter = get_isolated_assets(client, symbol, print_out=True)
							coin_net, coin_free, coin_borrowed, coin_locked = state_getter['coin']()

							sell_repay_result['side'] = side
							sell_repay_result['quantity'] = sell_quantity

							return sell_repay_result, coin_net, coin_free, coin_borrowed, coin_locked
						except binance.exceptions.BinanceAPIException as binanceEx:
							if binanceEx.code != -3044 or tryall > TRANSACTION_RETRY_COUNT():
								_try_oco_action_print(log_s, side, sell_quantity, sell_price, take_profit_price, stop_loss_price, take_profit_ratio, stop_loss_ratio, transaction_id, correlation_id, error_title=f'ERROR > TRIED [{tryall}] > ')

								log_s.append(f"!!!!! LONG OCO PLACE ERROR !!!!!")
								state_getter = get_isolated_assets(client, symbol, print_out=True, log_s=log_s)
								stable_net, stable_free, stable_borrowed, stable_locked = state_getter['stable']()
								coin_net, coin_free, coin_borrowed, coin_locked = state_getter['coin']()

								raise

							time.sleep(1)
							return wrap_margin_oco_sell(tryall + 1)

					sell_repay_result, coin_net, coin_free, coin_borrowed, coin_locked = wrap_margin_oco_sell()

					print_log_trades(f"{POSITION_SEPARATOR} transaction_id: {transaction_id} || correlation_id: {correlation_id} {POSITION_SEPARATOR}")

					oco_result = {
						'transaction_type': _LONG,
						'transaction_id': transaction_id,
						'correlation_id': correlation_id,
						'entry_order': borrow_buy_result,
						'exit_oco_order': sell_repay_result,
						'datetime_price': datetime_price,
						'entry_datetime_price': datetime_price,
					}

					return oco_result
				except Exception as ex:
					_record_position_error(log_s, _LONG, transaction_id, correlation_id, ex)

					print_log_trades(f"{POSITION_SEPARATOR} transaction_id: {transaction_id} || correlation_id: {correlation_id} {POSITION_SEPARATOR}")

					if borrow_buy_result:
						sell_repay_result = client.create_margin_order(
							symbol=symbol,
							side=SIDE_SELL,
							type=ORDER_TYPE_MARKET,
							quantity=buy_quantity,
							isIsolated='True',
							sideEffectType="AUTO_REPAY",
							transaction_id=transaction_id
						)

					raise

		def SHORT(entry_precheck, take_profit_ratio, stop_loss_ratio, transaction_id, correlation_id):
			logger = CONSOLE_SPLITTED if is_autotrading() else DEBUG_SPLITTED

			can_execute_order = entry_precheck['can_execute_order']
			part = entry_precheck['part']
			datetime_price = entry_precheck['datetime_price']
			max_stable_borrowable = entry_precheck['max_stable_borrowable']
			oco_sell_deals_count = entry_precheck['oco_sell_deals_count']
			oco_buy_deals_count = entry_precheck['oco_buy_deals_count']
			cause_s = entry_precheck['cause_s']

			x = 1 / part

			with log_context(logger) as log_s:
				part_pesent = f'1/{round(1 / x)}' if x < 1 else ''
				position_side_log_title = f'{_b("SHORT IN")}: {part_pesent}x{leverage} | {get_pretty_datetime_price(datetime_price)} || {get_present_total_net_USDT()}'

				log_s.append(position_side_log_title)
				printmd_log_trades(f"{POSITION_SEPARATOR} transaction_id: {transaction_id} || correlation_id: {correlation_id} {POSITION_SEPARATOR}")
				printmd_log_trades(position_side_log_title)

				state_getter = get_isolated_assets(client, symbol, print_out=True)

				stable_net, stable_free, stable_borrowed, stable_locked = state_getter['stable']()
				coin_net, coin_free, coin_borrowed, coin_locked = state_getter['coin']()

				print_log_trades('---------------------------------------------------------------------------')

				side = SIDE_SELL
				sell_price = datetime_price['price']
				max_coin_borrowable = tryall_delegate(lambda: get_max_borrow_asset(coin_asset), tryalls_count=3)
				coin_not_available = stable_locked / sell_price
				coin_use = (max_coin_borrowable - coin_not_available) * x
				sell_quantity = floor(coin_use, quantity_round_order)
				borrow_sell_result = None

				print_log_trades(f'STATE | coin borrowable: {_float_n(max_coin_borrowable, 5)} | stable borrowable: {_float_n(max_coin_borrowable * sell_price, 5)}')

				try:
					def wrap_margin_sell(tryall=1):
						try:
							borrow_sell_result = client.create_margin_order(
								symbol=symbol,
								side=side,
								type=ORDER_TYPE_MARKET,
								quantity=sell_quantity,
								isIsolated='True',
								sideEffectType="MARGIN_BUY",
								transaction_id=transaction_id
							)
							order_id_title = borrow_sell_result['orderId']
							_try_market_action_print(log_s, side, sell_quantity, sell_price, transaction_id, correlation_id, order_id_title=order_id_title)

							return borrow_sell_result, stable_net, stable_free, stable_borrowed, stable_locked, coin_net, coin_free, coin_borrowed, coin_locked
						except binance.exceptions.BinanceAPIException as binanceEx:
							if binanceEx.code != -3044 or tryall > TRANSACTION_RETRY_COUNT():
								_try_market_action_print(log_s, side, sell_quantity, sell_price, transaction_id, correlation_id, error_title=f'TRIED [{tryall}] > ')

								raise

							time.sleep(1)
							return wrap_margin_sell(tryall + 1)

					borrow_sell_result, stable_net, stable_free, stable_borrowed, stable_locked, coin_net, coin_free, coin_borrowed, coin_locked = wrap_margin_sell()

					while True:
						state_getter = get_isolated_assets(client, symbol, print_out=True)
						_stable_net, _stable_free, _stable_borrowed, _stable_locked = state_getter['stable']()
						_coin_net, _coin_free, _coin_borrowed, _coin_locked = state_getter['coin']()
						if (coin_net == _coin_net or coin_free == _coin_free) and (stable_net == _stable_net or stable_free == _stable_free):
							print_log_trades('----------------------------- RETRY ---------------------------------------')
							time.sleep(0.1)
							continue
						else:
							stable_net, stable_free, stable_borrowed, stable_locked = _stable_net, _stable_free, _stable_borrowed, _stable_locked
							coin_net, coin_free, coin_borrowed, coin_locked = _coin_net, _coin_free, _coin_borrowed, _coin_locked
							print_log_trades('---------------------------------------------------------------------------')
							break

					def wrap_margin_oco_buy(tryall=1):
						side = SIDE_BUY
						buy_price = get_datetime_price()['price']
						buy_quantity = calc_oco_buy_quantity(borrow_sell_result)

						take_profit_price = floor(buy_price / take_profit_ratio, price_round_order)
						stop_price = round(buy_price * stop_loss_ratio, price_round_order)
						stop_loss_price = round(stop_price * stop_loss_stop_ratio, price_round_order)

						try:
							buy_repay_result = client.create_margin_oco_order(
								symbol=symbol,
								side=side,
								stopLimitTimeInForce=TIME_IN_FORCE_GTC,
								quantity=buy_quantity,
								price=take_profit_price,
								stopPrice=stop_price,
								stopLimitPrice=stop_loss_price,
								isIsolated='True',
								sideEffectType="AUTO_REPAY",
								transaction_id=transaction_id
							)
							profit_order = get_item_from_list_dict(buy_repay_result['orderReports'], 'type', _LIMIT_MAKER)
							loss_order = get_item_from_list_dict(buy_repay_result['orderReports'], 'type', _STOP_LOSS_LIMIT)
							order_id_title = f"{profit_order['orderId']} | {loss_order['orderId']}"
							_try_oco_action_print(log_s, side, buy_quantity, buy_price, take_profit_price, stop_loss_price, take_profit_ratio, stop_loss_ratio, transaction_id, correlation_id, order_id_title=order_id_title)

							state_getter = get_isolated_assets(client, symbol, print_out=True)
							stable_net, stable_free, stable_borrowed, stable_locked = state_getter['stable']()

							buy_repay_result['side'] = side
							buy_repay_result['quantity'] = buy_quantity

							return buy_repay_result, stable_net, stable_free, stable_borrowed, stable_locked
						except binance.exceptions.BinanceAPIException as binanceEx:
							if binanceEx.code != -3044 or tryall > TRANSACTION_RETRY_COUNT():
								_try_oco_action_print(log_s, side, buy_quantity, buy_price, take_profit_price, stop_loss_price, take_profit_ratio, stop_loss_ratio, transaction_id, correlation_id, error_title=f'ERROR > TRIED [{tryall}] > ')

								state_getter = get_isolated_assets(client, symbol, print_out=True)
								stable_net, stable_free, stable_borrowed, stable_locked = state_getter['stable']()
								coin_net, coin_free, coin_borrowed, coin_locked = state_getter['coin']()
								log_s.append(f"!!!!! SHORT OCO PLACE ERROR !!!!!")
								log_s.append(f"stable_net: {stable_net} | stable_free: {stable_free} | stable_borrowed: {stable_borrowed} | stable_locked: {stable_locked}")
								log_s.append(f"coin_net: {coin_net} | coin_free: {coin_free} | coin_borrowed: {coin_borrowed} | coin_locked: {coin_locked}")

								raise

							time.sleep(1)
							return wrap_margin_oco_buy(tryall+1)

					buy_repay_result, stable_net, stable_free, stable_borrowed, stable_locked = wrap_margin_oco_buy()

					print_log_trades(f"{POSITION_SEPARATOR} transaction_id: {transaction_id} || correlation_id: {correlation_id} {POSITION_SEPARATOR}")

					oco_result = {
						'transaction_type': _SHORT,
						'transaction_id': transaction_id,
						'correlation_id': correlation_id,
						'entry_order': borrow_sell_result,
						'exit_oco_order': buy_repay_result,
						'datetime_price': datetime_price,
						'entry_datetime_price': datetime_price,
					}

					return oco_result
				except Exception as ex:
					print_log_trades(f"{POSITION_SEPARATOR} transaction_id: {transaction_id} || correlation_id: {correlation_id} {POSITION_SEPARATOR}")

					if borrow_sell_result:
						buy_repay_result = client.create_margin_order(
							symbol=symbol,
							side=SIDE_BUY,
							type=ORDER_TYPE_MARKET,
							quantity=sell_quantity,
							isIsolated='True',
							sideEffectType="AUTO_REPAY",
							transaction_id=transaction_id
						)

					raise

		position_side_action_d = {
			_LONG: LONG,
			_SHORT: SHORT
		}

		def cancel_order_safe(order_id):
			order = client.get_margin_order(symbol=symbol, isIsolated=True, orderId=order_id)
			if order['status'] != 'CANCELED':
				client.cancel_margin_order(symbol=symbol, isIsolated=True, orderId=order_id)

		@lock_with_file(lockfile=_OCO_ORDER_CHECKER_LOCK_FILE_PATH(), timeout=15 if not is_running_under_pycharm_debugger() else 300)
		def close_all_positions(oco_result_s):
			with log_context(CONSOLE_SPLITTED) as log_s:
				log_s.append(f"CLOSING ALL POSITIONS [{sym_join}]:")

				transaction_grouped_order_d = binance_helpers.get_margin_opened_transaction_grouped_order_d(client, sym_join)

				for transaction_id, order_d in transaction_grouped_order_d.items():
					log_s.append(f"  CANCEL OCO ORDERS [{transaction_id}]:")
					for type, order in filter_dict(order_d, ['side']).items():
						log_s.append(f"     {type} | TRIGGER PRICE: {order['price']}")

				datetime_price = get_datetime_price()
				exit_price = datetime_price['price']

				active_oco_result_s = [oco_result for oco_result in oco_result_s if 'transaction_result' not in oco_result]
				for oco_result in active_oco_result_s:
					correlation_id = oco_result['correlation_id']
					transaction_id = oco_result['transaction_id']
					transaction_type = oco_result['transaction_type']
					entry_order = oco_result['entry_order']
					exit_oco_order = oco_result['exit_oco_order']
					entry_price = calculate_weighted_average_price(entry_order['fills'])
					exit_quantity = exit_oco_order['quantity']
					exit_side = exit_oco_order['side']
					exit_take_profit_order_id = exit_oco_order['orders'][0]['orderId']
					exit_stop_loss_order_id = exit_oco_order['orders'][1]['orderId']

					if transaction_type == _LONG:
						if exit_price > entry_price:
							transaction_result = 'PROFIT'
						else:
							transaction_result = 'LOSS'
					elif transaction_type == _SHORT:
						if exit_price < entry_price:
							transaction_result = 'PROFIT'
						else:
							transaction_result = 'LOSS'
					else:
						raise AssertionError(f"NO TRANSACTION TYPE ALLOWED: {transaction_type}")

					tryall_delegate(lambda: cancel_order_safe(exit_take_profit_order_id), label=f'Cancel OCO TAKE PROFIT order ERROR')
					tryall_delegate(lambda: cancel_order_safe(exit_stop_loss_order_id), label=f'Cancel OCO STOP LOSS order ERROR')

					time.sleep(1)

					exit_order = tryall_delegate(lambda: client.create_margin_order(
						symbol=symbol,
						side=exit_side,
						type=ORDER_TYPE_MARKET,
						quantity=exit_quantity,
						isIsolated='True',
						sideEffectType="AUTO_REPAY"
					), label=f'{transaction_type} > Close transaction on ERROR')

					oco_result['exit_price'] = exit_price
					oco_result['exit_quantity'] = exit_quantity
					oco_result['exit_order'] = exit_order
					oco_result['exit_datetime_price'] = datetime_price
					oco_result['transaction_result'] = transaction_result

					_try_oco_executed_print(log_s, oco_result)

				align_balances(full_print=True, ext_log_s=log_s)

				log_s.append(f"CANCELLED ALL OCO ORDERS")

		class BaseExecutor:
			def __init__(self, stop_queue, hash):
				self.stop_queue = stop_queue
				self.hash = hash

			def close_all_positions(self):
				pass

			def initialize(self, started_dt_utc, symbol, ext_log_s=None):
				initialize(started_dt_utc, initial_balance, ext_log_s=ext_log_s)

			def execute_oco_orders(self):
				pass

			def ignore(self, transaction_id='TRANSACTION_ID', correlation_id='CORRELATION_ID'):
				datetime_price = get_datetime_price()
				IGNORE(datetime_price, transaction_id, correlation_id)

			def get_total_net_USDT(self):
				total_net_USDT = get_total_net_USDT()

				return total_net_USDT

			def get_order_entry_precheck(self, side, transaction_id, correlation_id):
				_hash = self.hash
				datetime_price = get_datetime_price()

				long_positions_count = get_side_positions_count(SIDE_BUY)
				short_positions_count = get_side_positions_count(SIDE_SELL)
				positions_remains = max_positions_count - (long_positions_count + short_positions_count)
				part = positions_remains + PART_OFFSET()
				total_stable_balance = round(get_total_net_USDT(), 5)

				logs = []
				log_seprator = f'{ALERT_SEPARATOR} [{kiev_now().isoformat()}] transaction_id: {transaction_id} | correlation_id: {correlation_id} {ALERT_SEPARATOR}'

				min_balance = initial_balance * stop_on_drop_down_ratio
				if total_stable_balance < min_balance:
					error_msg = f"[{_hash}] EXIT | DROPPED BELOW | MIN: {min_balance} | BALANCE: ${total_stable_balance}"
					logs.extend([log_seprator, error_msg, log_seprator])
					print_log_trades(*logs)
					NOTICE_SPLITTED(*logs)

					raise ExitAutomationError(error_msg)

				if is_backtesting():
					if is_no_trades_timeout():
						error_msg = f"[{_hash}] EXIT | NO TRADES TIMEOUT | BALANCE: ${total_stable_balance}"
						logs.extend([log_seprator, error_msg, log_seprator])
						print_log_trades(*logs)
						NOTICE_SPLITTED(*logs)

						raise ExitAutomationError(error_msg)

					if not run_to_end:
						max_balance = initial_balance * stop_on_jump_up_ratio
						if total_stable_balance > max_balance:
							error_msg = f"[{_hash}] EXIT | JUMPED ABOVE | MAX: {max_balance} | BALANCE: ${total_stable_balance}"
							logs.extend([log_seprator, error_msg, log_seprator])
							print_log_trades(*logs)
							NOTICE_SPLITTED(*logs)

							raise ExitAutomationError(error_msg)

						if is_over_timeout(self.created_dt) and abs(initial_balance - total_stable_balance) / initial_balance <= NO_CHANGE_RATIO():
							error_msg = f"[{_hash}] EXIT | NO PRICE CHANGE | BALANCE: ${total_stable_balance}"
							logs.extend([log_seprator, error_msg, log_seprator])
							print_log_trades(*logs)
							NOTICE_SPLITTED(*logs)

							raise ExitAutomationError(error_msg)

				if side == _LONG:
					max_stable_available = tryall_delegate(lambda: get_max_borrow_asset(stable_asset), tryalls_count=3)
				elif side == _SHORT:
					max_coin_borrowable = tryall_delegate(lambda: get_max_borrow_asset(coin_asset), tryalls_count=3)
					max_stable_available = max_coin_borrowable * datetime_price['price']
				else:
					raise RuntimeError(f"!!! SIDE: {side} NOT ALLOWED !!!")

				can_execute_order_condition_s = []
				cause_s = []

				stable_used = max_stable_available / part
				if stable_used >= min_stable_allowed:
					can_execute_order_condition_s.append(True)
				else:
					can_execute_order_condition_s.append(False)
					ignore_cause = f"MIN NOTIONAL CONSTRAINT: {stable_used} [STABLE USED] < {min_stable_allowed} [STABLE MIN]"
					cause_s.append(ignore_cause)

				if positions_remains > 0:
					can_execute_order_condition_s.append(True)
				else:
					can_execute_order_condition_s.append(False)
					oco_orders_count = f"LONG`s: {long_positions_count} | SHORT`s: {short_positions_count}"
					ignore_cause = f'NO ORDERS LEFT: {positions_remains} of {max_positions_count} || {oco_orders_count}'
					cause_s.append(ignore_cause)

				can_execute_order = all(can_execute_order_condition_s)

				return {
					'can_execute_order': can_execute_order,
					'cause_s': cause_s,
					'part': part,
					'total_stable_balance': total_stable_balance,
					'max_stable_borrowable': max_stable_available,
					'stable_used': stable_used,
					'oco_buy_deals_count': long_positions_count,
					'oco_sell_deals_count': short_positions_count,
					'datetime_price': datetime_price,
				}

			def can_execute_long_order(self, transaction_id, correlation_id):
				entry_precheck = self.get_order_entry_precheck(_LONG, transaction_id, correlation_id)

				return entry_precheck['can_execute_order']

			def can_execute_short_order(self, transaction_id, correlation_id):
				entry_precheck = self.get_order_entry_precheck(_SHORT, transaction_id, correlation_id)

				return entry_precheck['can_execute_order']

			def finalize(self):
				finalize()

		class ExecutorBacktestingSync(BaseExecutor):
			def __init__(self, stop_queue, hash):
				super().__init__(stop_queue, hash)
				self.oco_result_s = []
				self.created_dt = datetime.now()

			def execute_long_order(self, take_profit_ratio, stop_loss_ratio, transaction_id='TRANSACTION_ID', correlation_id='CORRELATION_ID'):
				entry_precheck = self.get_order_entry_precheck(_LONG, transaction_id, correlation_id)
				if entry_precheck['can_execute_order']:
					place_oco_result = LONG(entry_precheck, take_profit_ratio, stop_loss_ratio, transaction_id, correlation_id)
					self.oco_result_s.append(place_oco_result)
				else:
					WARNING(entry_precheck['datetime_price'], transaction_id, correlation_id, entry_precheck['cause_s'])

			def execute_short_order(self, take_profit_ratio, stop_loss_ratio, transaction_id='TRANSACTION_ID', correlation_id='CORRELATION_ID'):
				entry_precheck = self.get_order_entry_precheck(_SHORT, transaction_id, correlation_id)
				if entry_precheck['can_execute_order']:
					place_oco_result = SHORT(entry_precheck, take_profit_ratio, stop_loss_ratio, transaction_id, correlation_id)
					self.oco_result_s.append(place_oco_result)
				else:
					WARNING(entry_precheck['datetime_price'], transaction_id, correlation_id, entry_precheck['cause_s'])

			def execute_oco_orders(self):
				filtered_oco_result_s = [o for o in self.oco_result_s if 'transaction_result' not in o]
				sorted_filtered_oco_result_s = filtered_oco_result_s[::-1]
				for oco_result in sorted_filtered_oco_result_s:
					filled_orders_count = client.execute_oco_order_backtesting(oco_result)
					if filled_orders_count > 0:
						oco_checker(oco_result)

			def exit_position(self, transaction_id, correlation_id):
				filtered_oco_result_s = [o for o in self.oco_result_s if 'transaction_result' not in o]
				sorted_filtered_oco_result_s = filtered_oco_result_s[::-1]
				client.exit_position_backtesting(sorted_filtered_oco_result_s, transaction_id, correlation_id)


		return {
			'executors_producer': lambda stop_queue, hash: [ExecutorBacktestingSync(stop_queue, hash)],
			'get_total_net_USDT': get_total_net_USDT,
			'get_present_total_net_USDT': get_present_total_net_USDT,
		}