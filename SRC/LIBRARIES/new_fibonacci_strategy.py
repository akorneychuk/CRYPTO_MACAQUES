
import math
from datetime import datetime, timedelta
from uuid import uuid4

import pandas as pd
from dateutil import parser

from SRC.CORE._CONSTANTS import _UTC_TIMESTAMP, _KIEV_TIMESTAMP, project_root_dir
from SRC.LIBRARIES.new_data_utils import fetch
from SRC.LIBRARIES.new_plot_utils import display_df_full
from SRC.LIBRARIES.time_utils import TIME_DELTA, as_utc_tz


def is_target_candle(df_window): #TODO: Window has 1000 candles, last candle is tagert or not > True or False
    pass


def is_target_candle_killme(last_closed_min_row, discretization, df_window): #Window has 1000 candles, last candle is tagert or not > True or False
    each_ns = 7
    is_target_candle = (last_closed_min_row[_UTC_TIMESTAMP].to_pydatetime() - as_utc_tz(datetime(1970, 1, 1))) % (TIME_DELTA(discretization) * each_ns) == timedelta(0)

    return is_target_candle


def calc_pnl_qty(side, entry_price, exit_price, quantity, entry_fee, exit_fee):
    if side == 'LONG':
        gross_pnl = quantity * (exit_price - entry_price)
    elif side == 'SHORT':
        gross_pnl = quantity * (entry_price - exit_price)
    else:
        raise RuntimeError("Invalid side. Must be 'LONG' or 'SHORT'.")

    total_fee = quantity * entry_price * entry_fee + quantity * exit_price * exit_fee
    net_pnl = gross_pnl - total_fee

    return {
        'gross_pnl': gross_pnl,
        'total_fee': total_fee,
        'net_pnl': net_pnl,
    }


class TargetCandle:
    def __init__(self, strategy, min_candle, target_candle):
        dismiss_multiplier = 1.005

        low = target_candle['low']
        high = target_candle['high']

        sl_level = strategy.sl_level
        levels = strategy.levels
        level_weights = strategy.level_weights

        log_start = math.log(low)
        log_end = math.log(high)
        log_range = log_end - log_start
        all_levels = [0] + levels + [sl_level]
        all_level_prices = {level: math.exp(log_start + log_range * level) for level in all_levels}
        entry_level_setup_s = [{
            'level': level,
            'price': all_level_prices[level],
            'weight': level_weights[level],
            'transaction_id': uuid4().hex[:12]
        } for level in levels]
        general_stop_loss_price = all_level_prices[sl_level]

        self.min_candle_s = [min_candle]
        self.entry_level_setup_s = entry_level_setup_s
        self.general_stop_loss_price = general_stop_loss_price
        self.correlation_id = uuid4().hex[:12]
        self.all_level_prices = all_level_prices
        self.all_levels = all_levels
        self.entered_level_s = []
        self.dismiss_price = min_candle['close'] * dismiss_multiplier
        self.is_dismissed = False

    def is_active(self):
        is_running = len(self.entered_level_s) == 0 or any(transaction['status'] == 'ENTERED' for transaction in self.entered_level_s)
        is_active = is_running and not self.is_dismissed

        return is_active

    def produce_signal(self, last_closed_min_row):
        correlation_id = self.correlation_id

        min_candle = last_closed_min_row.to_dict()
        min_candle_high = min_candle['high']
        min_candle_low = min_candle['low']

        if self.min_candle_s[-1][_UTC_TIMESTAMP] < min_candle[_UTC_TIMESTAMP]:
            self.min_candle_s.append(min_candle)

        ignore_signal = {
            'correlation_id': correlation_id,
            'transaction_id': uuid4().hex[:12],
            'signal': 'IGNORE',
        }

        if min_candle_high >= self.dismiss_price and len(self.entered_level_s) == 0:
            self.is_dismissed = True

            return [ignore_signal]

        for entry_level_setup in self.entry_level_setup_s:
            transaction_id = entry_level_setup['transaction_id']
            entry_level = entry_level_setup['level']
            entry_weight = entry_level_setup['weight']
            entry_part = 1 / entry_weight
            entry_price = entry_level_setup['price']
            entry_idx = self.all_levels.index(entry_level)

            entered_level_transaction_id_s = [entered_level['transaction_id'] for entered_level in self.entered_level_s if entered_level['status'] == 'ENTERED']
            if transaction_id not in entered_level_transaction_id_s and min_candle_low <= entry_price:
                general_stop_loss_price = self.general_stop_loss_price

                exit_idx = entry_idx - 1
                exit_level = self.all_levels[exit_idx]
                exit_price = self.all_level_prices[exit_level]

                take_profit_ratio = round(exit_price / entry_price, 4)
                # stop_loss_ratio = round(entry_price / general_stop_loss_price, 4)
                # profit_loss_ratio = round((take_profit_ratio - 1) / (entry_price / general_stop_loss_price - 1), 4)

                self.entered_level_s.append({
                    'transaction_id': transaction_id,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'entry_level': entry_level,
                    'exit_level': exit_level,
                    'status': 'ENTERED',
                })

                return [{
                    'correlation_id': correlation_id,
                    'transaction_id': transaction_id,
                    'signal': 'LONG',
                    'take_profit_ratio': take_profit_ratio,
                    # 'stop_loss_ratio': stop_loss_ratio,
                    # 'profit_loss_ratio': profit_loss_ratio,
                    'stop_loss_price': general_stop_loss_price,
                    'part': entry_part,
                }]

        if len(self.entered_level_s) > 0:
            last_entered_level = self.entered_level_s[-1]
            last_entered_level_exit_price = last_entered_level['exit_price']
            if min_candle_high >= last_entered_level_exit_price:
                close_signal_s = []
                for entered_level in self.entered_level_s:
                    if entered_level['transaction_id'] == last_entered_level['transaction_id']:
                        continue

                    transaction_id = entered_level['transaction_id']
                    entered_level['status'] = 'EXITED'
                    close_signal_s.append({
                        'correlation_id': correlation_id,
                        'transaction_id': transaction_id,
                        'signal': 'CLOSE',
                    })
                if len(close_signal_s) > 0:
                    return close_signal_s
                else:
                    return [ignore_signal]

            if min_candle_low <= self.general_stop_loss_price:
                for entered_level in self.entered_level_s:
                    entered_level['status'] = 'EXITED'

        return [ignore_signal]

    def calc_stats(self):
        pass #Calculate here statistics


class Strategy:
    def __init__(self, discretization, candle_depo):
        self.discretization = discretization
        self.candle_depo = candle_depo
        self.target_candle_s = []
        self.base_level = -0.618
        self.position_weights = [0.2, 0.3, 0.5]
        self.levels = [round(self.base_level - (i - 1) * 1.0, 4) for i in range(1, len(self.position_weights) + 1)]
        self.level_weights = {level: self.position_weights[idx] for idx, level in enumerate(self.levels)}
        self.sl_level = self.levels[-1] - 1.0

    def produce_target_candle_if_match(self, last_closed_min_row, window):
        # is_target_candle = is_target_candle(window)
        is_target_candle = is_target_candle_killme(last_closed_min_row, self.discretization, window)

        if is_target_candle:
            min_canlde = last_closed_min_row.to_dict()
            target_candle = window.iloc[-1].to_dict()

            target_candle = TargetCandle(self, min_canlde, target_candle)
            self.target_candle_s.append(target_candle)

    def produce_signal_s(self, last_closed_min_row, target_window):
        is_target_discretization = (last_closed_min_row[_UTC_TIMESTAMP].to_pydatetime() - as_utc_tz(datetime(1970, 1, 1))) % TIME_DELTA(self.discretization) == timedelta(0)
        if is_target_discretization:
            self.produce_target_candle_if_match(last_closed_min_row, target_window)

        signal_s = []
        active_target_candle_s = [target_candle for target_candle in self.target_candle_s if target_candle.is_active()]
        for target_candle in active_target_candle_s:
            signals = target_candle.produce_signal(last_closed_min_row)
            signal_s.extend(signals)

        return signal_s


def run_signal_producer():
    symbol = 'BTCUSDT'
    market_type = 'FUTURES'
    discretization = '15M'
    discretization_min = '1M'
    start_dt_str = '2025-10-01'
    candle_depo = 1000
    entry_fee = 0.0004
    exit_fee = 0.0004
    side = 'LONG'

    target_df_cache_path = f'{project_root_dir()}/killme__target_df__{symbol}__{market_type}__{discretization}__{start_dt_str}.csv'
    min_df_cache_path = f'{project_root_dir()}/killme__min_df__{symbol}__{market_type}__{discretization_min}__{start_dt_str}.csv'
    try:
        # TODO: DON'T FORGET TO CLEAR CACHED DF`s ONCE CHANGED start_dt
        target_df = pd.read_csv(target_df_cache_path, index_col='timestamp', parse_dates=[_UTC_TIMESTAMP, _KIEV_TIMESTAMP], infer_datetime_format=True)
        min_df = pd.read_csv(min_df_cache_path, index_col='timestamp', parse_dates=[_UTC_TIMESTAMP, _KIEV_TIMESTAMP], infer_datetime_format=True)
    except Exception as ex:
        target_df = fetch(market_type, symbol, discretization, parser.parse(start_dt_str))
        min_df = fetch(market_type, symbol, discretization_min, parser.parse(start_dt_str))

        target_df.to_csv(target_df_cache_path)
        min_df.to_csv(min_df_cache_path)

    strategy = Strategy(discretization, candle_depo)

    transaction_s = []

    target_window_size = 1000
    min_window_size = int(TIME_DELTA(discretization) / TIME_DELTA(discretization_min)) * target_window_size
    for i in range(min_window_size, len(min_df) + 1):
        min_window = min_df.iloc[i - min_window_size:i]
        last_closed_min_row = min_window.iloc[-1]
        min_candle = last_closed_min_row.to_dict()
        min_candle_timestamp = min_candle[_UTC_TIMESTAMP]
        target_window = target_df[target_df[_UTC_TIMESTAMP].isin(min_window[_UTC_TIMESTAMP])]

        signal_s = strategy.produce_signal_s(last_closed_min_row, target_window)

        for signal in signal_s:
            if signal['signal'] == 'LONG':
                entry_price = min_candle['close']
                take_profit_ratio = signal['take_profit_ratio']
                stop_loss_ratio = signal['stop_loss_ratio']
                part = signal['part']

                profit_price = entry_price * take_profit_ratio
                loss_price = entry_price / stop_loss_ratio

                quantity = round(candle_depo / part / entry_price, 6)

                transaction_s.append({
                    'idx': min_candle_timestamp,
                    'correlation_id': signal['correlation_id'],
                    'transaction_id': signal['transaction_id'],
                    'entry_price': entry_price,
                    'profit_price': profit_price,
                    'loss_price': loss_price,
                    'quantity': quantity,
                    'entry_timestamp': min_candle_timestamp,
                    'status': 'ENTERED',
                })
            elif signal['signal'] == 'CLOSE':
                transaction = [transaction for transaction in transaction_s if transaction['transaction_id'] == signal['transaction_id']][0]

                quantity = transaction['quantity']
                entry_price = transaction['entry_price']
                exit_price = min_candle['close']

                pnl = calc_pnl_qty(side, entry_price, exit_price, quantity, entry_fee, exit_fee)
                transaction['gross_pnl'] = pnl['gross_pnl']
                transaction['total_fee'] = pnl['total_fee']
                transaction['net_pnl'] = pnl['net_pnl']
                transaction['exit_timestamp'] = min_candle_timestamp,
                transaction['status'] = 'EXITED'

        entered_transaction_s = [transaction for transaction in transaction_s if transaction['status'] == 'ENTERED']
        for transaction in entered_transaction_s:
            if min_candle['high'] >= transaction['profit_price']:
                exit_price = transaction['profit_price']
                status = 'PROFIT'
            elif min_candle['low'] <= transaction['loss_price']:
                exit_price = transaction['loss_price']
                status = 'LOSS'
            else:
                continue

            entry_price = transaction['entry_price']
            quantity = transaction['quantity']

            pnl = calc_pnl_qty(side, entry_price, exit_price, quantity, entry_fee, exit_fee)
            transaction['gross_pnl'] = pnl['gross_pnl']
            transaction['total_fee'] = pnl['total_fee']
            transaction['net_pnl'] = pnl['net_pnl']
            transaction['exit_timestamp'] = min_candle_timestamp,
            transaction['status'] = status

        print(f"Produced strategy signals: {signal_s}")

    transaction_df = pd.DataFrame(transaction_s).set_index('idx')
    display_df_full(transaction_df.head(30))


if __name__ == "__main__":
    run_signal_producer()