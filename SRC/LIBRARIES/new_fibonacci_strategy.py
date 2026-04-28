import os
import math
from datetime import datetime, timedelta
from uuid import uuid4

import pandas as pd
from SRC.CORE.debug_utils import CONSOLE_SPLITTED
from dateutil import parser

from SRC.CORE._CONSTANTS import _UTC_TIMESTAMP, _KIEV_TIMESTAMP, project_root_dir, BINANCE_TAKER_COMISSION, BINANCE_MAKER_COMISSION
from SRC.LIBRARIES.new_data_utils import fetch
from SRC.LIBRARIES.new_plot_utils import display_df_full
from SRC.LIBRARIES.new_utils import create_folder_file
from SRC.LIBRARIES.time_utils import TIME_DELTA, as_utc_tz, deterministic_int
from SRC.LIBRARIES import new_utils as nu


def is_last_candle_target(df, window=10, measure_percentile=0.4, use_candle_size_instead_of_shadow=True, filter_by_measure_range=False, measure_lower_mult=0.5, measure_upper_mult=2.0, use_mrc=True, use_mrc_r2=True, use_mrc_s2=True):
    """
    Определяет, является ли последняя свеча в DataFrame целевой.

    Параметры:
    df : pandas.DataFrame с колонками 'open', 'high', 'low', 'close', 'volume'
          Должен содержать минимум window+1 свечей (рекомендуется 1000)
    window : период для скользящих расчётов (10)
    measure_percentile : порог перцентиля (0.8 = 80-й перцентиль)
    use_candle_size_instead_of_shadow : если True, используем свечной размах (high-low); иначе максимальную тень
    filter_by_measure_range : если True, дополнительно фильтруем по диапазону относительно скользящего среднего
    measure_lower_mult, measure_upper_mult : множители для диапазона
    use_mrc : использовать ли фильтр MRC
    use_mrc_r2, use_mrc_s2 : учитывать ли касания R2 и/или S2

    Возвращает:
    bool : True, если последняя свеча удовлетворяет всем критериям
    """
    if len(df) < window + 1:
        return False

    # Копируем, чтобы не портить исходный DataFrame
    df = df.copy()

    # Если нужно, добавляем MRC индикаторы
    if use_mrc:
        df = nu.add_mrc_indicators(df, length=200)  # используем длину 200 как в mrc_calculate

    # Рассчитываем объёмный перцентиль (по предыдущим свечам)
    df['volume_percentile'] = df['volume'].shift(1).rolling(window=window, min_periods=1).quantile(measure_percentile)

    # Рассчитываем меру (размах свечи или максимальная тень)
    if use_candle_size_instead_of_shadow:
        df['measure'] = df['high'] - df['low']
    else:
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        df['measure'] = df[['upper_shadow', 'lower_shadow']].max(axis=1)

    # Перцентиль меры по предыдущим свечам
    df['measure_percentile'] = df['measure'].shift(1).rolling(window=window, min_periods=1).quantile(measure_percentile)

    # Скользящее среднее меры для фильтрации по диапазону
    if filter_by_measure_range:
        df['measure_avg'] = df['measure'].shift(1).rolling(window=window, min_periods=1).mean()

    # Берём последнюю свечу
    last = df.iloc[-1]
    idx = len(df) - 1  # индекс последней строки

    # os.system('say Hello! I am the Nippel System. Who are you? Sergei Ilon Mask or Andrey Pidaras?')

    # Условие: достаточно ли данных для расчёта (idx >= window)
    if idx < window:
        return False

    # Касание уровней MRC (если используется)
    if use_mrc:
        if 'upband2' not in last or 'loband2' not in last:
            return False
        r2 = last['upband2']
        s2 = last['loband2']
        high = last['high']
        low = last['low']

        touches_r2 = (high >= r2 and low <= r2) or (low > r2)
        touches_s2 = (high >= s2 and low <= s2) or (high < s2)
        touches_level = (use_mrc_r2 and touches_r2) or (use_mrc_s2 and touches_s2)
        if not touches_level:
            return False

    # Объём выше перцентиля
    volume_ok = last['volume'] > df['volume_percentile'].iloc[idx]
    if not volume_ok:
        return False

    # Мера выше перцентиля
    measure_ok = last['measure'] > df['measure_percentile'].iloc[idx]
    if not measure_ok:
        return False

    # Дополнительная фильтрация по диапазону
    if filter_by_measure_range:
        avg = df['measure_avg'].iloc[idx]
        if not pd.isna(avg):
            low_bound = avg * measure_lower_mult
            high_bound = avg * measure_upper_mult
            if not (low_bound <= last['measure'] <= high_bound):
                return False

    return True


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
    def __init__(self, strategy, trigger_candle, target_candle):
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

        self.trigger_candle_s = [trigger_candle]
        self.entry_level_setup_s = entry_level_setup_s
        self.general_stop_loss_price = general_stop_loss_price
        self.correlation_id = uuid4().hex[:12]
        self.all_level_prices = all_level_prices
        self.all_levels = all_levels
        self.entered_level_s = []
        self.dismiss_price = trigger_candle['close'] * dismiss_multiplier
        self.is_dismissed = False

    def is_dismissed(self):
        return self.is_dismissed

    def is_inprogress(self):
        return len(self.entered_level_s) > 0

    def is_active(self):
        is_running = len(self.entered_level_s) == 0 or any(transaction['status'] == 'ENTERED' for transaction in self.entered_level_s)
        is_active = is_running and not self.is_dismissed

        return is_active

    def produce_signal(self, trigger_candle):
        correlation_id = self.correlation_id

        trigger_candle_idx = trigger_candle[_UTC_TIMESTAMP]
        trigger_candle_close = trigger_candle['close']
        trigger_candle_high = trigger_candle['high']
        trigger_candle_low = trigger_candle['low']

        if self.trigger_candle_s[-1][_UTC_TIMESTAMP] < trigger_candle[_UTC_TIMESTAMP]:
            self.trigger_candle_s.append(trigger_candle)

        ignore_signal = {
            'idx': trigger_candle_idx,
            'correlation_id': correlation_id,
            'transaction_id': uuid4().hex[:12],
            'signal': 'IGNORE',
        }

        if trigger_candle_high >= self.dismiss_price and len(self.entered_level_s) == 0:
            self.is_dismissed = True

            return [ignore_signal]

        for entry_level_setup in self.entry_level_setup_s:
            transaction_id = entry_level_setup['transaction_id']
            entry_level = entry_level_setup['level']
            entry_weight = entry_level_setup['weight']
            entry_price = entry_level_setup['price']
            entry_part = 1 / entry_weight
            entry_idx = self.all_levels.index(entry_level)

            entered_level_transaction_id_s = [entered_level['transaction_id'] for entered_level in self.entered_level_s if entered_level['status'] == 'ENTERED']
            if transaction_id not in entered_level_transaction_id_s and trigger_candle_low <= entry_price:
                general_stop_loss_price = self.general_stop_loss_price

                exit_idx = entry_idx - 1
                exit_level = self.all_levels[exit_idx]
                exit_price = self.all_level_prices[exit_level]

                take_profit_ratio = round(exit_price / entry_price, 4)
                stop_loss_ratio = round(entry_price / general_stop_loss_price, 4)
                profit_loss_ratio = round((take_profit_ratio - 1) / (entry_price / general_stop_loss_price - 1), 4)

                self.entered_level_s.append({
                    'transaction_id': transaction_id,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'entry_level': entry_level,
                    'exit_level': exit_level,
                    'status': 'ENTERED',
                })

                return [{
                    'idx': trigger_candle_idx,
                    'correlation_id': correlation_id,
                    'transaction_id': transaction_id,
                    'signal': 'LONG',
                    'actual_entry_price': entry_price,
                    'take_profit_ratio': take_profit_ratio,
                    'stop_loss_ratio': stop_loss_ratio,
                    'profit_loss_ratio': profit_loss_ratio,
                    'stop_loss_price': general_stop_loss_price,
                    'part': entry_part,
                }]

        if len(self.entered_level_s) > 0:
            last_entered_level = self.entered_level_s[-1]
            last_entered_level_exit_price = last_entered_level['exit_price']
            if trigger_candle_high >= last_entered_level_exit_price:
                close_signal_s = []
                for entered_level in self.entered_level_s:
                    if entered_level['transaction_id'] == last_entered_level['transaction_id']:
                        entered_level['status'] = 'EXITED'
                        continue

                    transaction_id = entered_level['transaction_id']
                    entered_level['status'] = 'EXITED'
                    close_signal_s.append({
                        'idx': trigger_candle_idx,
                        'correlation_id': correlation_id,
                        'transaction_id': transaction_id,
                        'signal': 'CLOSE',
                        'actual_exit_price': last_entered_level_exit_price,
                    })

                if len(close_signal_s) > 0:
                    return close_signal_s
                else:
                    return [ignore_signal]

            if trigger_candle_low <= self.general_stop_loss_price:
                for entered_level in self.entered_level_s:
                    entered_level['status'] = 'EXITED'

        return [ignore_signal]


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

    def produce_target_candle_if_match(self, trigger_candle, window):
        is_target_candle = is_last_candle_target(window)

        if is_target_candle:
            target_candle = window.iloc[-1].to_dict()

            target_candle = TargetCandle(self, trigger_candle, target_candle)
            self.target_candle_s.append(target_candle)

    def produce_signal_s(self, trigger_candle, target_window):
        is_target_discretization = (trigger_candle[_UTC_TIMESTAMP].to_pydatetime() - as_utc_tz(datetime(1970, 1, 1))) % TIME_DELTA(self.discretization) == timedelta(0)
        if is_target_discretization:
            self.produce_target_candle_if_match(trigger_candle, target_window)

        signal_s = []
        active_target_candle_s = [target_candle for target_candle in self.target_candle_s if target_candle.is_active()]
        for target_candle in active_target_candle_s:
            signals = target_candle.produce_signal(trigger_candle)
            signal_s.extend(signals)

        return signal_s


def run_signal_producer(init_data):
    symbol = init_data['symbol']
    market_type = init_data['market_type']
    discretization = init_data['discretization']
    discretization_min = init_data['discretization_min']
    start_dt_str = init_data['start_dt_str']
    candle_depo = init_data['candle_depo']
    entry_fee = init_data['entry_fee']
    exit_fee = init_data['exit_fee']
    side = 'LONG'

    out_evaluation_folder_full_path = f'{project_root_dir()}/OUT/EVALUATION/FIBONACCI_STRATEGY'

    target_df_cache_path = f'{out_evaluation_folder_full_path}/CACHE/{symbol}__{market_type}__{discretization}__{start_dt_str}.csv'
    min_df_cache_path = f'{out_evaluation_folder_full_path}/CACHE/{symbol}__{market_type}__{discretization}__{start_dt_str}.csv'
    result_df_path = f'{out_evaluation_folder_full_path}/{symbol}__{market_type}__{discretization}__{start_dt_str}.csv'

    create_folder_file(target_df_cache_path)
    create_folder_file(min_df_cache_path)
    create_folder_file(result_df_path)

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
    closed_transactions_count = 0
    for i in range(min_window_size, len(min_df) + 1):
        min_window = min_df.iloc[i - min_window_size:i]
        trigger_candle = min_window.iloc[-1].to_dict()
        trigger_candle_close = trigger_candle['close']
        trigger_candle_high = trigger_candle['high']
        trigger_candle_low = trigger_candle['low']
        trigger_candle_idx = trigger_candle[_UTC_TIMESTAMP]

        target_window = target_df[target_df[_UTC_TIMESTAMP].isin(min_window[_UTC_TIMESTAMP])]

        signal_s = strategy.produce_signal_s(trigger_candle, target_window)

        for signal in signal_s:
            if signal['signal'] == 'LONG':
                actual_entry_price = signal['actual_entry_price']

                take_profit_ratio = signal['take_profit_ratio']
                stop_loss_ratio = signal['stop_loss_ratio']
                part = signal['part']

                profit_price = actual_entry_price * take_profit_ratio
                loss_price = actual_entry_price / stop_loss_ratio

                quantity = round(candle_depo / actual_entry_price / part, 6)

                transaction_s.append({
                    'idx': trigger_candle_idx,
                    'correlation_id': signal['correlation_id'],
                    'transaction_id': signal['transaction_id'],
                    'entry_price': actual_entry_price,
                    'profit_price': profit_price,
                    'loss_price': loss_price,
                    'quantity': quantity,
                    'entry_timestamp': trigger_candle_idx,
                    'status': 'OPENED',
                })
            elif signal['signal'] == 'CLOSE':
                transaction = [transaction for transaction in transaction_s if transaction['transaction_id'] == signal['transaction_id']][0]

                quantity = transaction['quantity']
                entry_price = transaction['entry_price']
                actual_exit_price = signal['actual_exit_price']

                pnl = calc_pnl_qty(side, entry_price, actual_exit_price, quantity, entry_fee, exit_fee)
                transaction['gross_pnl'] = pnl['gross_pnl']
                transaction['total_fee'] = pnl['total_fee']
                transaction['net_pnl'] = pnl['net_pnl']
                transaction['exit_timestamp'] = trigger_candle_idx
                transaction['status'] = 'INTERRUPTED'
                transaction['result'] = 'LOSS'

        opened_transaction_s = [transaction for transaction in transaction_s if transaction['status'] == 'OPENED']
        for transaction in opened_transaction_s:
            profit_price = transaction['profit_price']
            loss_price = transaction['loss_price']

            if deterministic_int(trigger_candle_idx) % 2 == 0:
                if trigger_candle_high >= profit_price:
                    exit_price = profit_price
                    result = 'PROFIT'
                elif trigger_candle_low <= loss_price:
                    exit_price = loss_price
                    result = 'LOSS'
                else:
                    continue
            else:
                if trigger_candle_low <= loss_price:
                    exit_price = loss_price
                    result = 'LOSS'
                elif trigger_candle_high >= profit_price:
                    exit_price = profit_price
                    result = 'PROFIT'
                else:
                    continue

            entry_price = transaction['entry_price']
            quantity = transaction['quantity']

            pnl = calc_pnl_qty(side, entry_price, exit_price, quantity, entry_fee, exit_fee)
            transaction['gross_pnl'] = pnl['gross_pnl']
            transaction['total_fee'] = pnl['total_fee']
            transaction['net_pnl'] = pnl['net_pnl']
            transaction['exit_timestamp'] = trigger_candle_idx
            transaction['status'] = 'CLOSED'
            transaction['result'] = result

        logs = [f'{str(trigger_candle_idx)}']
        # if len(signal_s) > 0:
        #     logs.append(f"PRODUCED SIGNALS: {list(reversed(signal_s))}")

        # opened_transaction_s = [transaction for transaction in transaction_s if transaction['status'] == 'OPENED']
        # if len(opened_transaction_s) > 0:
        #     logs.append(f'OPENED TRANSACTION: {list(reversed(opened_transaction_s))}')

        closed_transaction_s = [transaction for transaction in transaction_s if transaction['status'] != 'OPENED']
        if len(closed_transaction_s) > 0:
            logs.append(f'CLOSED TRANSACTION:')
            for closed_transaction in closed_transaction_s[-10:]:
                logs.append(f"  {closed_transaction}")

        if len(logs) > 1 and len(closed_transaction_s) > closed_transactions_count:
            CONSOLE_SPLITTED(*logs)

        closed_transactions_count = len(closed_transaction_s)

    target_candles_alive_count = len([tc for tc in strategy.target_candle_s if tc.is_active()])
    target_candles_inprogress_count = len([tc for tc in strategy.target_candle_s if tc.is_inprogress()])
    target_candles_dismissed_count = len([tc for tc in strategy.target_candle_s if tc.is_dismissed])
    target_candles_info_str = f'TARGET CANDLES TOTAL: {len(strategy.target_candle_s)} | ALIVE: {target_candles_alive_count} | IN PROGRESS: {target_candles_inprogress_count} | DISMISSED: {target_candles_dismissed_count}'

    if len(transaction_s) > 0:
        transactions_df = pd.DataFrame(transaction_s).set_index('idx')
        transactions_df.to_csv(result_df_path)

        transactions_df = pd.read_csv(result_df_path, index_col='idx', parse_dates=['entry_timestamp', 'exit_timestamp'], infer_datetime_format=True)
        total_gross_pnl = transactions_df['gross_pnl'].sum()
        total_fee = transactions_df['total_fee'].sum()
        total_net_pnl = transactions_df['net_pnl'].sum()

        print(f"CLOSED TRANSACTIONS COUNT: {len(transaction_s)} | {target_candles_info_str}")
        print(f"GROSS PNL: {total_gross_pnl:.2f} | TOTAL FEE: {total_fee:.2f} | NET PNL: {total_net_pnl:.2f}")
        display_df_full(transactions_df.head(30))
    else:
        print(f"NO CLOSED TRANSACTIONS | {target_candles_info_str}")


if __name__ == "__main__":
    os.environ['BINANCE_FEE'] = '0.036 - 0.036'

    init_data = {}
    init_data['symbol'] = 'ZECUSDT'
    init_data['symbol'] = 'PEOPLEUSDT'
    init_data['symbol'] = 'ZKCUSDT'
    init_data['symbol'] = 'EPICUSDT'
    init_data['symbol'] = 'CYBERUSDT'
    init_data['symbol'] = 'PUMPUSDT'
    init_data['market_type'] = 'FUTURES'
    init_data['discretization'] = '1H'
    init_data['discretization_min'] = '1M'
    init_data['start_dt_str'] = '2025-09-01'
    init_data['candle_depo'] = 1000
    init_data['entry_fee'] = BINANCE_TAKER_COMISSION()
    init_data['exit_fee'] = BINANCE_TAKER_COMISSION()

    run_signal_producer(init_data)