import asyncio
import decimal
import json
import math
import os
import random
import time
from _decimal import Decimal
from datetime import datetime, timedelta
from functools import reduce, lru_cache
from itertools import tee
from json import JSONDecodeError
from json import JSONDecodeError as JSONDecodeError2
from json.decoder import JSONDecodeError as JSONDecodeError1
from threading import Thread

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import trapz
from pandas import Timestamp
from sklearn.metrics import roc_curve, auc
from sklearn.neighbors import KernelDensity

from SRC.CORE.ApiClient import ApiClient
from SRC.CORE._CONSTANTS import CLASSES, CPU_COUNT, STATE_OUT, STATE_IN, TZ, INTERVAL, PARTITIONING_MAP, TRADE_FOLDER_PATH, UTC_TZ, KIEV_TZ, project_root_dir
from SRC.CORE._CONSTANTS import FEATURE_DIFF, MEAN_GRAD, START_COL, END_COL, ROLLING_GRAD_WINDOW_FREQs, UNBALANCED_CENTER_RATIO, \
    FORCE_FEATURIZE, LOG_START, POWER_DEGREE, NON_LINEARITY_TOP_DEFAULT, LOG_BASE, FEATURES, SYMBOL_PROCESS_KEY, STABLE_COIN_KEY, BALANCE_KEY, \
    ALT_COIN_KEY, ACTION_NO
from SRC.CORE.debug_utils import printmd, log_module
from SRC.CORE.debug_utils import printmd_low, run_medium, print_memory, produce_measure_medium, printmd_high, display_high, printmd_medium, display_medium, run_high, \
    print_action_title_description__low, is_cloud

pd.options.mode.chained_assignment = None


def queue_replace_first(queue):
    if len(queue) > 1:
        last_element = queue.pop()
        queue.insert(0, last_element)


def shift_queue(queue):
    if len(queue) > 1:
        first_element = queue.pop(0)
        queue.append(first_element)

    return queue


def flatten(l):
    return [item for sublist in l for item in sublist]


def get_next_value(lst, index):
    return lst[index % len(lst)]


def linear(a, b, x_t):
    linear = np.array(list(map(lambda x: a + x * b, x_t)))

    return linear


def build_days_segments(X_s, Y_s, D_s, size, slide):
    day_x_s = []
    day_y_s = []
    day_d_s = []

    end = len(X_s)
    start = end - size - 1
    while start >= 0:
        cur_x = X_s[start:end]
        cur_y = Y_s[start:end]
        cur_d = D_s[start:end]

        day_x_s.append(cur_x)
        day_y_s.append(cur_y)
        day_d_s.append(cur_d)

        start = start - slide
        end = end - slide

    return np.array(day_x_s[::-1]), np.array(day_y_s[::-1]), np.array(day_d_s[::-1])


def build_price_change_multiplier_sequence(day_segments):
    day_x_s = day_segments[0]
    day_y_s = day_segments[1]
    multiplier_s = []

    for i in range(len(day_x_s) - 1, -1, -1):
        price_segment = day_y_s[i]
        price_n = price_segment[len(price_segment) - 1]
        price_n_minus_1 = price_segment[0]
        if price_n > price_n_minus_1:
            multiplier = price_n / price_n_minus_1
        elif price_n < price_n_minus_1:
            multiplier = - (price_n_minus_1 / price_n)
        else:
            multiplier = 0

        multiplier_s.append(multiplier)

    return np.array(multiplier_s[::-1])


def build_price_change_multiplier(data_frame, size, stride):
    indexes_x = data_frame.index.to_numpy()
    prices_x = data_frame['Price'].to_numpy()
    dates_x = data_frame['Date'].to_numpy()
    segments = build_days_segments(indexes_x, prices_x, dates_x, size, stride)
    multiplier_sequence = build_price_change_multiplier_sequence(segments)

    return segments, multiplier_sequence


def build_derivate_coords(derivate_s, day_segment_s):
    day_x_s = day_segment_s
    derivative_coords = []
    for i in range(len(derivate_s)):
        derivate = derivate_s[i]
        day_segment = day_x_s[i]
        derivative_coord = build_derivative_coord(derivate, day_segment[0], day_segment[len(day_segment) - 1])
        derivative_coords.append(derivative_coord)

    return np.array(derivative_coords)


def build_axis_x_date_range(data_frame, window):
    xxx = data_frame.index.to_numpy()
    x_ticks_range = (np.arange(max(xxx) + 1, min(xxx), -window) - 1)[::-1]
    d_ticks_range = np.array(list(map(lambda i: data_frame.loc[i]['Date'].strftime("%d/%m/%Y"), x_ticks_range)))

    return x_ticks_range, d_ticks_range


def build_normal_distribution_measure_probability_distribution_range(y_s, data_min, data_max, dits_ranges):
    x_d = np.linspace(data_min, data_max, 1000)

    kde = KernelDensity(bandwidth=0.1, kernel='gaussian')
    kde.fit(y_s[:, None])

    logprob = kde.score_samples(x_d[:, None])

    density = np.exp(logprob) / 4.945
    plt.fill_between(x_d, density, alpha=0.5)
    plt.plot(y_s, np.full_like(y_s, -0.01), '|k', markeredgewidth=1)
    plt.show()

    probabilities = []
    for r in dits_ranges:
        start = r[0]
        end = r[1]
        x_d_r = density[start:end]
        prob = trapz(x_d_r, dx=0.1)

        x_r = x_d[start:end]
        _from_abs = x_r[0]
        _from_perc = round(_from_abs * 100, 2)
        _to_abs = x_r[len(x_r) - 1]
        _to_perc = round(_to_abs * 100, 2)

        probabilities.append(((_from_abs, _to_abs), prob))
        plt.axvline(x=_from_abs, color='b', label=f'x={_from_abs}%')
        plt.axvline(x=_to_abs, color='b', label=f'x={_to_abs}')
        _from_abs_pres = '{:.4f}'.format(_from_abs)
        _to_abs_pres = '{:.4f}'.format(_to_abs)
        if _from_abs >= 0:
            plt.text(_from_abs + 0.01, 0.6, f'x={_from_abs_pres}', rotation=90)
        else:
            plt.text(_from_abs - 0.035, 0.6, f'x={_from_abs_pres}', rotation=90)
        if _to_abs >= 0:
            plt.text(_to_abs + 0.01, 0.6, f'x={_to_abs_pres}', rotation=90)
        else:
            plt.text(_to_abs - 0.035, 0.6, f'x={_to_abs_pres}', rotation=90)

        print(f'Range (%)= {_from_perc} : {_to_perc} > Probability = {prob}')

    return np.array(probabilities[::-1])


def build_log_data(Y_s):
    res = np.log(Y_s)

    return res


def test_stationarity(time_series, window):
    from statsmodels.tsa.stattools import adfuller
    movingAverage = time_series.rolling(window=window).mean()
    movingStandartDeviation = time_series.rolling(window=window).std()

    orig = plt.plot(time_series, color='blue', label='Original')
    mean = plt.plot(movingAverage, color='red', label=f'Rolling mean - {window}')
    std = plt.plot(movingStandartDeviation, color='black', label=f'Rolling standart deviation - {window}')

    plt.legend(loc='best')
    plt.title('Rolling mean and standart deviation')
    plt.show(block=False)

    df_test = adfuller(time_series, autolag='AIC')
    f_output = pd.Series(df_test[0:4], index=['Test Statistics', 'p-value', '#Lags used', 'Umber og observations used'])
    for key, value in df_test[4].items():
        f_output[f'Critica Value {key}'] = value

    print(f_output)


def select_range(data, start_date, end_date):
    mask = (data['Date'] >= start_date) & (data['Date'] <= end_date)
    range_dt = data.loc[mask]

    return range_dt


def normalize_data_set(data):
    data['Price'] = pd.to_numeric(data['Price'].str.replace(',', ''), errors='coerce')
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.iloc[::-1].reset_index(drop=True)
    prices_x = data['Price'].to_numpy()
    # prices_x_log = build_log_data(prices_x)
    # prices_x_log___norm_0_1 = NormalizeData_0_plus_1(prices_x_log)  # For all dataset
    # data['PriceNorm'] = prices_x_log___norm_0_1.tolist()

    return data


def print_prices_extremum(data_range, from_d, to_d):
    dates_range = pd.date_range(start=from_d, end=to_d).to_pydatetime().tolist()
    for date in dates_range:
        from_d = date
        to_d = date + timedelta(days=1)
        sub_data_range = select_range(data_range, from_d, to_d)
        weighter_price_s = sub_data_range['Weighted_Price'].to_numpy()
        weighter_price_mean = np.nanmean(weighter_price_s, axis=0)
        print(f'{from_d}: {weighter_price_mean}')


def _predicate(x, _from, _to):
    res = _from <= x <= _to

    return res


def build_lambda(_from, _to):
    return lambda x: _predicate(x, _from, _to)


def get_num_from_one_hot_label(one_hot_label):
    a = one_hot_label.detach().numpy()
    res = np.where(a == 1)[1]
    num_label = res[0]

    return num_label


def get_class_from_bin(bin, bins_list):
    index = 0
    for start_bin, end_bin in bins_list:
        if start_bin <= bin <= end_bin:
            return index

        index += 1

    if bin < bins_list[0][0]:
        return 0

    if bin > bins_list[-1][1]:
        return len(bins_list) - 1

    raise Exception(f"OUT OF BINS: {bins_list} VALUE:{bin}")


def get_one_hot_from_num_label(num_label, class_count, device):
    import torch

    if torch.is_tensor(num_label):
        num_label = num_label.detach().cpu().data.numpy().astype(int)
    else:
        num_label = num_label.astype(int)

    a = np.array([num_label])
    b = np.zeros((a.size, class_count))
    b[np.arange(a.size), a] = 1

    return torch.Tensor(b).to(device)


def build_permutad_data(_y_s, indexes_range):
    import torch

    forecast_window = 30
    prediction_window = 1

    min_x = min(indexes_range) + forecast_window
    max_x = max(indexes_range) - prediction_window

    ______________range_abs = max_x - min_x - forecast_window - prediction_window
    ______________permutation_abs = torch.randperm(______________range_abs)
    permutation = ______________permutation_abs + min_x + forecast_window

    x_s = []
    y_s = []
    for i in permutation:
        multiplier_x = _y_s[i - forecast_window:i]
        x_s.append(multiplier_x)

        multiplier_y = _y_s[i:i + prediction_window]
        y_s.append(multiplier_y[0])

    x_s = np.array(x_s)
    y_s = np.array(y_s)

    return x_s, y_s


def calculate_auc_roc(actuals, predictions):
    actuals = np.array(actuals)
    predictions = np.array(predictions)
    fpr = {}
    tpr = {}
    thresh = {}
    roc_auc = {}

    for i in range(len(predictions[0])):
        fpr[i], tpr[i], thresh[i] = roc_curve(actuals, predictions[:, i], pos_label=i)
        roc_auc[i] = auc(fpr[i], tpr[i])

    return fpr, tpr, roc_auc


def produce_out_image_file_path(is_train, dir, file_name):
    out_dir = f'OUT'
    sub_dir = "TRAIN" if is_train else "TEST"
    path = f'{out_dir}/{dir}/{sub_dir}'

    import os
    if not os.path.exists(path):
        os.makedirs(path)

    return f'{path}/{file_name}'


def get_row_delta(df):
    row_delta = df.iloc[1].name - df.iloc[0].name

    return row_delta


def featurize_range(df, row_index_val, rows_count, featurize):
    rows_delta = get_row_delta(df) * rows_count
    start_from = df.loc[row_index_val].name - rows_delta

    filtered_df = df[df.index >= start_from]
    filtered_df = featurize(filtered_df, df)
    df[df.index >= row_index_val] = filtered_df[filtered_df.index >= row_index_val]

    return df


def normalize_minus_1__plus_1(data, outer_max=None, outer_min=None):
    max_val = outer_max if outer_max is not None else np.amax(data)
    min_val = outer_min if outer_min is not None else np.amin(data)
    max_abs = max([abs(max_val), abs(min_val)])
    min_abs = -max_abs
    normalized = (2 * ((data - min_abs) / (max_abs - min_abs))) - 1
    _min = np.amin(normalized)
    _max = np.amax(normalized)

    return normalized


def featurize_lambda(df, original_prop, target_prop, _lambda):
    vectorized = np.vectorize(_lambda)
    assigner = {target_prop: lambda r: vectorized(r[original_prop])}
    if target_prop not in df:
        df = df.assign(**assigner)

    null_values = df[target_prop].isnull()
    if null_values.any():
        df[null_values] = df[null_values].assign(**assigner)

    return df


def produce_empty_candle_df():
    return pd.DataFrame(columns=['close_time', 'open', 'high', 'low', 'close'])


def produce_empty_cached_df():
    return pd.DataFrame(columns=[*['timestamp', 'utc_timestamp'], *FEATURES])


def featurize_candles_mins_dashboard(candle_mins_buffer, build_presentation_coordinates=True):
    rolling_grad_window_freqs = ROLLING_GRAD_WINDOW_FREQs

    if len(candle_mins_buffer) == 0:
        return produce_empty_candle_df()

    if len(candle_mins_buffer) == 1:
        candles = list(candle_mins_buffer)
        candles.append(candles[0])
        candle_mins_buffer = candles

    candles_mins_df = pd.DataFrame(candle_mins_buffer)
    candles_mins_df = featurize_price_avg_mean_ratio(candles_mins_df)
    rolling_diff_window_freq = 2
    candles_mins_df['mean_abs_diff'] = candles_mins_df['mean'].rolling(rolling_diff_window_freq, min_periods=rolling_diff_window_freq).apply(calc_mean_abs_diff)
    candles_mins_df[FEATURE_DIFF] = candles_mins_df['mean_abs_diff'].apply(calc_mean_rel_diff)
    converter = lambda h: lambda x: x
    for rolling_grad_window_freq in rolling_grad_window_freqs:
        mean_grad = f'{MEAN_GRAD}_{rolling_grad_window_freq}'
        candles_mins_df[mean_grad] = candles_mins_df[FEATURE_DIFF].rolling(rolling_grad_window_freq, min_periods=rolling_grad_window_freq).apply(calc_grad)

    candles_mins_df = featurize_lambda(candles_mins_df, 'close_time', 'utc_timestamp', converter(0))
    if build_presentation_coordinates:
        for rolling_grad_window_freq in rolling_grad_window_freqs:
            candles_mins_df = do_build_presentation_coordinates(candles_mins_df, rolling_grad_window_freq)

    candles_mins_df.ta.rsi(append=True)
    candles_mins_df.ta.macd(append=True)
    candles_mins_df.ta.atr(14, append=True)

    return candles_mins_df


def featurize_price_avg_mean_ratio(df):
    cols = df[['open', 'high', 'low', 'close']]
    df['std'] = cols.std(axis=1)
    df['mean'] = cols.mean(axis=1)
    df['std_mean_ratio'] = df['std'] / df['mean']

    return df


def featurize(df, cached_df, parameters):
    from datetime import datetime

    pair = parameters['PAIR']
    discretization = parameters['DISCRETIZATION']
    rolling_diff_window_freq = parameters['ROLLING_DIFF_WINDOW_FREQ']
    rolling_grad_window_freqs = parameters['ROLLING_GRAD_WINDOW_FREQs']
    partitioning = PARTITIONING_MAP[parameters['PARTITIONING']]

    df = featurize_lambda(df, 'close_time', 'utc_timestamp', lambda ts: datetime.fromtimestamp(ts / 1000).astimezone(UTC_TZ) + timedelta(milliseconds=1))
    df = featurize_lambda(df, 'close_time', 'kiev_timestamp', lambda ts: datetime.fromtimestamp(ts / 1000).astimezone(TZ()) + timedelta(milliseconds=1))
    df = featurize_price_avg_mean_ratio(df)

    diff_window = f'{rolling_diff_window_freq * INTERVAL()}{partitioning}'
    df['mean_abs_diff'] = df['mean'].rolling(diff_window, min_periods=rolling_diff_window_freq).apply(calc_mean_abs_diff)
    df['mean_rel_diff'] = df['mean_abs_diff'].apply(calc_mean_rel_diff)

    for rolling_grad_window_freq in rolling_grad_window_freqs:
        grad_window = f'{rolling_grad_window_freq * INTERVAL()}{partitioning}'
        mean_grad = f'{MEAN_GRAD}_{rolling_grad_window_freq}'
        df[mean_grad] = df['mean_rel_diff'].rolling(grad_window, min_periods=rolling_grad_window_freq).apply(calc_grad)

    df['pair'] = pair
    df['discretization'] = discretization

    return df


def subtract_time_delta_from_date(date, _time_delta):
    subtracted_date = pd.to_datetime(date) - _time_delta

    return subtracted_date


def get_loc_by_condition_in_range(df, condition, in_range=timedelta(days=1)):
    loc = df.loc[condition]
    utc_timestamp = loc['utc_timestamp'].iloc[int(len(loc) / 2)]
    start_date = subtract_time_delta_from_date(utc_timestamp, in_range)
    end_date = subtract_time_delta_from_date(utc_timestamp, -in_range)
    pair = loc['pair'].to_list()[0]
    pair_df = df.where(df['pair'] == pair)
    df_range = pair_df[pair_df['utc_timestamp'] >= start_date][pair_df['utc_timestamp'] <= end_date]

    return loc, df_range


def split_df_with_overlap(df, nrows, overlap=0):
    def iterate():
        for i in range(0, len(df) - overlap, nrows - overlap):
            yield df.iloc[i: i + nrows]

    result = list(iterate())

    return result


def calculate_series_absolute_diff(x, y):
    if len(x) == 0:
        res = y
    else:
        _x = x[-1]
        res = _x + (_x * y)
    x.append(res)

    return x


def calc_grad(window):
    series = window.to_list()
    result = reduce(calc_series_cumulative_diff, series, [0])
    coef = calc_linear_regression_coefs(np.asarray(range(len(result))), result)
    grad = coef[1]

    return grad


def calc_series_cumulative_diff(x, y):
    if len(x) == 0:
        res = y
    else:
        _x = x[-1]
        res = _x + y
    x.append(res)

    return x


def calc_linear_regression_coefs(x, y):
    # number of observations/points
    n = np.size(x)

    # mean of x and y vector
    m_x = np.mean(x)
    m_y = np.mean(y)

    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y * x) - n * m_y * m_x
    SS_xx = np.sum(x * x) - n * m_x * m_x

    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1 * m_x

    return (b_0, b_1)


def roll_over_columns(df, input_col_name, out_cols_names, rolling_window, generator_func):
    output = pd.DataFrame(columns=out_cols_names)

    def multiple_column_rolling_wrap(_lambda):
        row = 0

        def compose(window, df):
            nonlocal row
            cols, vals = out_cols_names, _lambda(window)
            df.loc[row, cols] = vals
            row += 1

            return 1

        return compose

    df[input_col_name].rolling(rolling_window).apply(multiple_column_rolling_wrap(generator_func), kwargs={'df': output})

    output.index = df.index

    df[out_cols_names] = output[out_cols_names]
    del output

    return df


def calc_relative_diff(_y):
    if _y > 0:
        relative_diff = _y - 1
    elif _y < 0:
        relative_diff = _y + 1
    else:
        relative_diff = 0

    return relative_diff


def calc_mean_abs_diff(window):
    pair = window.to_list()
    if len(pair) == 1:
        return np.nan

    now = pair[1]
    prev = pair[0]

    if np.isclose(now, 0) or np.isclose(prev, 0):
        # return np.nan
        return 1.0

    if now > prev:
        abs_diff = now / prev
    elif now < prev:
        abs_diff = -(prev / now)
    else:
        abs_diff = 0

    return abs_diff


def calc_mean_rel_diff(abs_diff):
    if math.isnan(abs_diff):
        return np.nan

    relative_diff = calc_relative_diff(abs_diff)

    return relative_diff


def calc_mean_diff(window):
    pair = window.to_list()
    if len(pair) == 1:
        return np.nan, np.nan

    now = pair[1]
    prev = pair[0]

    if now > prev:
        abs_diff = now / prev
    elif now < prev:
        abs_diff = -(prev / now)
    else:
        abs_diff = 0

    relative_diff = calc_relative_diff(abs_diff)

    return abs_diff, relative_diff


def permute_segments(segments):  # permuted_list = permute_segments(list)
    import torch

    segments_permuted = []
    for i in torch.randperm(len(segments)):
        segment = segments[i]
        segments_permuted.append(segment)

    return segments_permuted


def do_build_presentation_coordinates(df, window, feature='mean'):
    items = df[f'{feature}_grad_{window}'].items()
    y_s = f'{feature}_grad_ys_{window}'
    y_e = f'{feature}_grad_ye_{window}'

    start = df.index[0]
    grad_window = get_row_delta(df) * window

    df[y_s] = np.nan
    df[y_e] = np.nan

    i = 0
    for indx, val in items:
        if indx < start + grad_window:
            continue
        grad_coord = build_derivative_coord(val, i - window, i)
        df.loc[indx - grad_window, y_s] = grad_coord[0][1]
        df.loc[indx, y_e] = grad_coord[1][1]
        i += 1

    df = df.sort_index()

    return df


def build_gradient_presentation_coordinates(symbol, df, ROLLING_GRAD_WINDOW_FREQs):
    title = f'BUILDING GRADIENTs'
    description = f"symbol = {symbol} | dataframe size = {len(df)} | gradients windows = {ROLLING_GRAD_WINDOW_FREQs}"
    print_action_title_description__low(title, description)

    run_medium(print_memory)

    measure = produce_measure_medium('GRADIENTs BUILT')

    printmd_high("**Original dataframe:**")
    display_high(df)

    for ROLLING_GRAD_WINDOW_FREQ in ROLLING_GRAD_WINDOW_FREQs:
        df = do_build_presentation_coordinates(df, ROLLING_GRAD_WINDOW_FREQ)

    printmd_high("**Extended dataframe:**")
    display_high(df)

    run_medium(print_memory)
    measure()

    return df


def get_csv_files_from_dir(dir, predicate):
    relevant_path = os.path.abspath(dir)
    included_extensions = ['csv']
    file_names = [fn for fn in os.listdir(relevant_path) if any(fn.endswith(ext) and predicate(fn) for ext in included_extensions)]

    return file_names


def num_zeroes_after_floating_point(x):
    if x % 1 == 0:
        return 0
    else:
        return -1 - math.floor(math.log10(x % 1))


def num_digits_after_floating_point(_float):
    my_decimal = Decimal(str(_float))
    res = abs(my_decimal.as_tuple().exponent)

    return res


def num_zeroes_before_floating_point(x):
    _abs = abs(x)

    return int(0 if _abs < 1 else math.log10(_abs * 1.0) + 1)


def process_format_precision_order(_float, precision=6):
    _float = float(_float)

    if pd.isna(_float):
        return _float

    if np.isinf(_float) or np.isneginf(_float):
        return _float

    precision_order = precision

    before = num_zeroes_before_floating_point(_float)
    after_zeros = num_zeroes_after_floating_point(_float)
    # after_digits = num_digits_after_floating_point(_float)

    if before > 0:
        float_precision = precision_order - before if precision_order >= before else 0
    else:
        float_precision = precision_order + after_zeros

    formatted_float = (f'%0.{float_precision}f' % _float).rstrip('0').rstrip('.')

    return formatted_float if formatted_float != '' else '0.0'


def _float_2(_float):
    return process_format_precision_order(_float, 2)


def _float_3(_float):
    return process_format_precision_order(_float, 3)


def _float_4(_float):
    return process_format_precision_order(_float, 4)


def _float_5(_float):
    return process_format_precision_order(_float, 5)


def _float_6(_float):
    return process_format_precision_order(_float, 6)


def _float_7(_float):
    return process_format_precision_order(_float, 7)


def _float_n(_float, precisition=2):
    return process_format_precision_order(_float, precisition)


# DEPRECATED (USELESS)
def process_format_precision_order_6_df(df):
    float_cols = df.select_dtypes(include=[np.float32])
    for float_col in float_cols:
        df[float_col] = df[float_col].apply(process_format_precision_order)

    return df


def fetch_featurize_cache_binance(process_symbol, trade_symbol, kline_size, filename, featurize=None, save=False):
    from SRC.CORE.utils import fetch_featurize_cache
    from SRC.CORE.binance_api import BinanceApiClient

    binance_client = BinanceApiClient(process_symbol, trade_symbol, kline_size)

    return fetch_featurize_cache(binance_client, process_symbol, kline_size, filename, featurize, save)


def fetch_featurize_cache(apiClient: ApiClient, symbol, kline_size, file_path, featurize=None, save=False):
    file_path_real = case_insensitive_path(file_path)
    kline_size = kline_size.lower()
    if os.path.isfile(file_path_real):
        cached_data = pd.read_csv(file_path_real)
    else:
        cached_data = pd.DataFrame()

    start_point, end_point = apiClient.minutes_of_new_data(cached_data)
    printmd_low(f"Downloading all available **{kline_size}** data  for **{symbol}** since: **{start_point}**. Be patient..!")
    klines = apiClient.get_historical_klines(start_point, end_point)
    new_data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])
    new_data['timestamp'] = pd.to_datetime(new_data['timestamp'], unit='ms')

    if len(cached_data) > 0:
        data_df = pd.concat([cached_data, new_data], ignore_index=True)
        cached_data_len = len(cached_data)
    else:
        data_df = new_data
        cached_data_len = 0

    measure = produce_measure_medium("PREPROCESS")

    data_df.set_index('timestamp', inplace=True)
    data_df.index = pd.to_datetime(data_df.index)

    data_df = data_df[~data_df.index.duplicated(keep='last')]

    data_df['open'] = pd.to_numeric(data_df['open'], errors='coerce')
    data_df['high'] = pd.to_numeric(data_df['high'], errors='coerce')
    data_df['low'] = pd.to_numeric(data_df['low'], errors='coerce')
    data_df['close'] = pd.to_numeric(data_df['close'], errors='coerce')
    data_df['volume'] = pd.to_numeric(data_df['volume'], errors='coerce')
    data_df['close_time'] = pd.to_numeric(data_df['close_time'], errors='coerce')
    data_df['quote_av'] = pd.to_numeric(data_df['quote_av'], errors='coerce')
    data_df['trades'] = pd.to_numeric(data_df['trades'], errors='coerce')
    data_df['tb_base_av'] = pd.to_numeric(data_df['tb_base_av'], errors='coerce')
    data_df['tb_quote_av'] = pd.to_numeric(data_df['tb_quote_av'], errors='coerce')

    measure()

    if featurize is not None:
        force = 'Force ' if FORCE_FEATURIZE else ''
        measure = produce_measure_medium(f'{force}FEATURIZE')
        overlap = 1000
        if cached_data_len >= overlap and not FORCE_FEATURIZE:
            if len(new_data) > 0:
                data_df = featurize_range(data_df, new_data.iloc[0]['timestamp'], overlap, featurize=featurize)
        else:
            data_df = featurize(data_df, None)

        measure()

    if save and (len(new_data) > 0 or FORCE_FEATURIZE):
        data_df.to_csv(file_path)
        pass

    if len(new_data) > 0:
        printmd_low(f"**All {symbol} caught up..! (NEW DATA SIZE: {len(new_data)})**")
    else:
        printmd_low(f"**All {symbol} caught up..!**")

    return data_df


def build_derivative_coord(derivate, start, end):
    alpha = derivate * 85
    tan_alpha = math.tan((alpha * math.pi / 180))
    A = B = end - start
    half_A = A / 2
    half_B = B / 2

    if derivate > 0.5:
        a = half_A
        b = abs(a / tan_alpha)
        i_x = start + B / 2 - b
        i_y = - a
        j_x = start + B / 2 + b
        j_y = + a
    elif 0.5 > derivate > 0:
        b = half_B
        a = abs(tan_alpha * b)
        i_x = start
        i_y = -a
        j_x = end
        j_y = +a
    elif 0 > derivate > -0.5:
        b = half_B
        a = abs(tan_alpha * b)
        i_x = start
        i_y = +a
        j_x = end
        j_y = -a
    elif derivate < -0.5:
        a = half_A
        b = abs(a / tan_alpha)
        i_x = start + B / 2 - b
        i_y = + a
        j_x = start + B / 2 + b
        j_y = - a
    else:
        i_x = start
        i_y = 0
        j_x = end
        j_y = 0

    return [(i_x, i_y), (j_x, j_y)]


def produce_grad_fields(ROLLING_GRAD_WINDOW_FREQs):
    feature_fields = list(map(lambda grad_freq: f'{MEAN_GRAD}_{grad_freq}', ROLLING_GRAD_WINDOW_FREQs))

    return feature_fields


nan_counter = 0


def has_nan_df(df):
    global nan_counter

    has_nan = df.isna().any().any()
    if has_nan:
        if nan_counter == 0:
            printmd_medium("***`DateTime NaNs:`***")
            nan_rows = df[df.isnull().any(axis=1)]
            display_medium(nan_rows)

        nan_counter += 1
    else:
        nan_counter = 0

    return has_nan


gap_counter = 0


def has_gap_df(df):
    global gap_counter

    seconds_df = df['utc_timestamp'].diff().iloc[1:].apply(lambda dt: dt.seconds)
    has_gap = len(seconds_df.unique()) > 1
    if has_gap:
        if gap_counter == 0:
            printmd_medium("***`DateTime Gaps:`***")
            printmd_medium(f"Range: {df.iloc[0]['utc_timestamp']} - {df.iloc[-1]['utc_timestamp']}")

        gap_counter += 1
    else:
        gap_counter = 0

    return has_gap


def filter_segments_containing_datetime_gaps(segments):
    segments = list(filter(lambda filtered_segment: not has_nan_df(filtered_segment) and not has_gap_df(filtered_segment), segments))

    return segments


def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def select_range(df, start_dt, end_dt):
    return df[df['timestamp'] >= start_dt][df['timestamp'] <= end_dt]


def get_sorted_features_list_from_df(df):
    sorted_features_list = sorted(df.columns.difference([START_COL, END_COL]), key=lambda x: int(x.split('_')[-1]))

    return sorted_features_list


def get_one_hot_cols(classes):
    return [f'oh_{i}' for i in range(classes)]


@lru_cache(maxsize=None)
def get_one_hot_from_clazz(clazz, clazzes_count):
    if clazz >= clazzes_count or clazz < 0:
        raise Exception(f"Clazz can be in range: 0 - {clazzes_count - 1} | Actual clazz: {clazz}")

    a = np.array([clazz])
    one_hot = np.zeros((a.size, clazzes_count))
    one_hot[np.arange(a.size), a] = 1
    one_hot = np.array(one_hot[0]).astype(int)

    return list(one_hot)


def get_one_hot_from_class_producer(classes):
    cl_oh_mapping = [get_one_hot_from_clazz(clazz, classes) for clazz in range(classes)]

    def one_hot_getter(clazz):
        one_hot = cl_oh_mapping[clazz]

        return one_hot

    return one_hot_getter


def get_class_from_one_hot(one_hot):
    res = np.where(one_hot == 1)[1]
    num_label = res[0]

    return num_label


# @lru_cache(maxsize=None)
def get_oh_clazz_map(clazzes_count):
    cl_oh_map = [get_one_hot_from_clazz(clazz, clazzes_count) for clazz in range(clazzes_count)]
    oh_cl_map = {str(one_hot): index for index, one_hot in enumerate(cl_oh_map)}

    return oh_cl_map


# @lru_cache(maxsize=None)
def get_clazz_oh_map(clazzes_count):
    clazz_oh_map = merge_dict_s([{clazz: get_one_hot_from_clazz(clazz, clazzes_count)} for clazz in range(clazzes_count)])

    return clazz_oh_map


# @lru_cache(maxsize=None)
def get_label_cl_map(labels):
    oh_cl_map = merge_dict_s([{f"{labels[str(key)]}": int(val)} for key, val in enumerate(labels)])

    return oh_cl_map


def merge_dict_s(dict_list):
    result = {}
    for d in dict_list:
        result.update(d)  # This will update result with the key-value pairs of d
    return result


class hashabledict(dict):
    def __hash__(self):
        return hash(str(self))


class hashablelist(list):
    def __hash__(self):
        return hash(str(self))


def upsert_dict(lst, new_dict, key):
    for i, d in enumerate(lst):
        if d.get(key) == new_dict.get(key):
            lst[i] = {**d, **new_dict}  # override existing
            break
    else:
        lst.append(new_dict)  # append if not found

    return lst


def upsert_dicts(main_list, new_list, key):
    # build an index for faster lookup
    index = {d[key]: i for i, d in enumerate(main_list) if key in d}

    for new_dict in new_list:
        k = new_dict.get(key)
        if k in index:
            # override existing dict by merging (new overrides old)
            main_list[index[k]] = {**main_list[index[k]], **new_dict}
        else:
            # append new if not exist
            main_list.append(new_dict)
            index[k] = len(main_list) - 1

    return main_list

def get_class_from_one_hot_producer(classes):
    oh_cl_map = get_oh_clazz_map(classes)

    def get_class(one_hot):
        clazz = oh_cl_map[str(one_hot)]

        return clazz

    return get_class


def evaluate_model(loader, net, criterion, device, class_from_one_hot_producer):
    predictions = []
    actuals = []
    losses = []
    for input_s_, output_oh_s_ in loader:
        input_s = input_s_.to(device)
        act_one_hot = output_oh_s_.float().to(device)

        pred = net(input_s)
        loss = criterion(pred, act_one_hot)
        losses.append(loss.item())

        actuals += output_oh_s_.detach().data.numpy().tolist()
        predictions += pred.detach().cpu().data.numpy().tolist()

    mean_loss = sum(losses) / len(losses)
    actuals = [class_from_one_hot_producer(a) for a in actuals]
    auc_roc = calculate_auc_roc(actuals, predictions)

    return auc_roc, mean_loss


def calc_weights(counts):
    total_count = sum(counts)
    frequencies = [count / total_count for count in counts]
    weights = [1 / frequency for frequency in frequencies]
    weights = [w / min(weights) for w in weights]

    return weights


def calc_symmetric_log_space(classes=CLASSES(), log_base=LOG_BASE(), non_linearity_top=NON_LINEARITY_TOP_DEFAULT, log_start=LOG_START(), space_top=1, unbalanced_center_ratio=UNBALANCED_CENTER_RATIO):
    title = f'**CALCULATE SYMMETRIC LOG SPACE**'
    description = f"`classes={classes}, log_base={log_base}, non_linearity_top={non_linearity_top}, log_start={log_start}, space_top={space_top}, unbalanced_center_ratio={unbalanced_center_ratio}`"
    printmd_medium(f"{title}\r\n\r\n{description}")

    symmetric_log_space = calc_symmetric_space_delegate(classes, lambda: calc_log_space(classes, log_base, non_linearity_top, log_start, space_top), space_top, unbalanced_center_ratio=unbalanced_center_ratio)

    return symmetric_log_space


def calc_symmetric_pow_space(classes=CLASSES(), power_degree=POWER_DEGREE(), non_linearity_top=NON_LINEARITY_TOP_DEFAULT, space_top=1, unbalanced_center_ratio=UNBALANCED_CENTER_RATIO, power_start=0):
    title = f'**CALCULATE SYMMETRIC POWER SPACE**'
    description = f"`classes={classes}, power_degree={power_degree}, non_linearity_top={non_linearity_top}, space_top={space_top}, unbalanced_center_ratio={unbalanced_center_ratio}`"
    printmd_medium(f"{title}\r\n\r\n{description}")

    symmetric_pow_space = calc_symmetric_space_delegate(classes, lambda: calc_power_space(classes, power_degree, non_linearity_top, space_top, power_start=power_start), space_top,
                                                        unbalanced_center_ratio=unbalanced_center_ratio)

    return symmetric_pow_space


def calc_symmetric_lin_space(classes=CLASSES(), space_top=1):
    title = f'**CALCULATE SYMMETRIC LINEAR SPACE**'
    description = f"`classes = {classes} | space top = {space_top}`"
    printmd_medium(f"{title}\r\n\r\n{description}")

    symmetric_lin_space = np.linspace(-space_top, space_top, classes + 1)

    return symmetric_lin_space


def calc_symmetric_space_delegate(classes, space_calculator, space_top=1, unbalanced_center_ratio=UNBALANCED_CENTER_RATIO):
    from SRC.CORE.plot_utils import display_plot, get_symmetric_space_distribution_fig

    if classes <= 1:
        raise Exception("CLASSES count should be higher than 1")

    if is_odd(classes) is not True:
        raise Exception("CLASSES count should be ODD")

    space = space_calculator()
    symmetric_space = calc_symmetric_space(space, space_top=space_top)
    center_first_index = int(len(symmetric_space) / 2)
    center_class_neighbors_wide_ratio = (symmetric_space[center_first_index] * 2) / (symmetric_space[center_first_index + 1] - symmetric_space[center_first_index])
    printmd_medium(f"`Neighbors ratio: {center_class_neighbors_wide_ratio}`")
    printmd_medium(f"`Non linear space: {['{:.5f}'.format(val) for val in symmetric_space]}`")

    if len(symmetric_space) - 1 > classes:
        msg = f"!!Log space doesn't meet to classes counts: symmetric_space len={len(symmetric_space)}, classes count={classes}!!"
        printmd(f"**`{msg}`**")

    if center_class_neighbors_wide_ratio >= 1:
        msg = f"Center class neighbors wide ratio = {center_class_neighbors_wide_ratio}"
        printmd(f"**`{msg}`**")

    fig = get_symmetric_space_distribution_fig(symmetric_space, 'Nonlinearity')

    if center_class_neighbors_wide_ratio >= unbalanced_center_ratio if unbalanced_center_ratio > 0 else False:
        msg = f"Center class should be <= neighbours wide {unbalanced_center_ratio} >> center_class_neighbors_wide_ratio = {center_class_neighbors_wide_ratio}"
        printmd(f"**`{msg}`**")
        display_plot(fig)

        raise Exception(msg)

    # run_high(lambda: display_plot(fig))

    return symmetric_space


def ___calc_space_bins(data, symmetric_space):
    bins, counts, weights, weights_count_product_norm = calc_distribution_histogram(data, symmetric_space)

    return bins, counts, weights, weights_count_product_norm, symmetric_space


def is_odd(num):
    return num % 2 == 1


def calc_symmetric_space(space, space_top=1):
    symmetric_space = list([*(space[::-1] * -1), *space])

    return symmetric_space


def calc_log_space_depr(classes, log_base, log_top, log_start, space_top=1):
    nstep = int((classes + 1) / 2)
    nstep = nstep - 1 if log_top < space_top else nstep
    seq = np.logspace(log_start, np.log(log_top) / np.log(log_base), nstep, base=log_base)
    log_space = np.insert(seq, 0, 0.0)[1:]
    log_space = np.array([*log_space, *[space_top]]) if log_top < space_top else log_space

    return log_space


def calc_log_space(classes, log_base, log_top, log_start, space_top=1):
    nstep = int((classes + 1) / 2)
    nstep = nstep - 1 if log_top < space_top else nstep
    seq = np.logspace(np.log(log_start) / np.log(log_base), np.log(log_top) / np.log(log_base), nstep, base=log_base)
    log_space = np.insert(seq, 0, 0.0)[1:]
    log_space = np.array([*log_space, *[space_top]]) if log_top < space_top else log_space

    return log_space


def calc_power_space_regular(classes, power, power_space_top=NON_LINEARITY_TOP_DEFAULT, space_top=1):
    linear_space = np.linspace(0, np.power(power_space_top, 1 / power), classes)
    power_space = np.power(linear_space, power)
    power_space = np.array([*power_space, *[space_top]]) if power_space_top < space_top else power_space

    return power_space


def calc_power_space(classes, power, power_space_top=NON_LINEARITY_TOP_DEFAULT, space_top=1, power_start=0):
    nstep = int((classes + 1) / 2) + 1
    nstep = nstep - 1 if power_space_top < space_top else nstep
    if power_start > 0:
        linear_space = np.linspace(np.power(power_start, 1 / power), np.power(power_space_top, 1 / power), nstep - 1)
        power_space = np.power(linear_space, power)
        power_space = np.array([*power_space, *[space_top]]) if power_space_top < space_top else power_space
    else:
        linear_space = np.linspace(0, np.power(power_space_top, 1 / power), nstep)
        power_space = np.power(linear_space, power)
        power_space = np.array([*power_space, *[space_top]]) if power_space_top < space_top else power_space
        power_space = power_space[1:]

    return power_space


def calc_distribution_histogram(data, symmetric_space):
    abs_max = abs(max(min(data), max(data), key=abs))
    counts, bins = np.histogram(data, bins=symmetric_space, range=(-abs_max, abs_max))
    weights = calc_weights(counts)
    weights_count_product = [weights[i] * counts[i] for i in range(len(weights))]
    weights_count_product_norm = weights_count_product / np.nanmax(weights_count_product)

    run_high(lambda: print(f"BINS: {bins} \r\n COUNTS: {counts} \r\n WEIGHTS: {weights} \r\n Weights x Counts NORM: {weights_count_product_norm}"))

    return bins, counts, weights, weights_count_product_norm


def are_close(float1, float2, percent_tolerance=1):
    if float1 == 0 and float2 == 0:
        return True

    percent_diff = abs(float1 - float2) / ((abs(float1) + abs(float2)) / 2) * 100

    return percent_diff <= percent_tolerance


def parse_pytz_dt(obj):
    dt_iso_str = obj['utc']
    dt = datetime.fromisoformat(dt_iso_str).replace(tzinfo=None)
    dt_utc = UTC_TZ.localize(dt)

    return dt_utc


def serialize_pytz_dt(dt):
    if type(dt) == Timestamp:
        dt = dt.to_pydatetime()

    if dt.tzinfo.utcoffset(None) != 0:
        dt_utc = dt.astimezone(UTC_TZ)
    else:
        dt_utc = dt

    serialized = {
        "_spec_type": "datetime",
        "utc": dt_utc.isoformat(),
        "kiev": dt_utc.astimezone(KIEV_TZ).isoformat(),
    }

    return serialized


CONVERTERS = {
    'datetime': parse_pytz_dt,
    'decimal': decimal.Decimal,
}


class MyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime,)):
            return serialize_pytz_dt(obj)
        elif isinstance(obj, (decimal.Decimal,)):
            return {"utc": str(obj), "_spec_type": "decimal"}
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, timedelta):
            return {"__timedelta__": True, "total_seconds": obj.total_seconds()}
        else:
            return super().default(obj)


def object_hook(obj):
    if "__timedelta__" in obj:
        return timedelta(seconds=obj["total_seconds"])

    _spec_type = obj.get('_spec_type')
    if not _spec_type:
        return obj

    if _spec_type in CONVERTERS:
        return CONVERTERS[_spec_type](obj)
    else:
        raise Exception('Unknown {}'.format(_spec_type))


json_write_lock_map = {}


def write_json_safe_locked(data, file_path, indent=4):
    from SRC.LIBRARIES.new_utils import lock_with_file

    lock_file_path = file_path.replace(".json", ".lock")
    lock_with_file(lock_file_path)(write_json)(data, file_path, indent)


def write_json(data, file_path, indent=4):
    from SRC.LIBRARIES.new_utils import create_folder_file
    create_folder_file(file_path)

    while True:
        try:
            with open(file_path, "w") as json_file:
                json.dump(data, json_file, indent=indent, cls=MyJSONEncoder)

            return
        except FileNotFoundError:
            raise
        except Exception as ex:
            print(str(ex))
            wait = random.uniform(0, 3)
            time.sleep(wait)


def read_json(file_path):
    with open(file_path, "r") as json_file:
        read_data = json.load(json_file, object_hook=object_hook)

        return read_data


def write_gson(data, file_path):
    import json
    import gzip

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with gzip.open(file_path, "wb") as f:
        f.write(json.dumps(data).encode("utf-8"))


def read_gson(file_path):
    import gzip
    with gzip.open(file_path, "rb") as f:
        read_data = json.loads(f.read().decode("utf-8"))

        return read_data


def read_json_safe_locked(file_path, default=None, lock_timeout_secs=10):
    from SRC.LIBRARIES.new_utils import lock_with_file

    lock_file_path = file_path.replace(".json", ".lock")
    json_result = lock_with_file(lock_file_path, timeout=lock_timeout_secs)(read_json_safe)(file_path, default)

    return json_result


def read_json_safe(file_path, default=None):
    if os.path.exists(file_path):
        try:
            return read_json(file_path)
        except JSONDecodeError:
            return default
    else:
        return default


def read_json_safe_retry(file_path, default=None):
    counter = 0
    while True:
        try:
            json_result = read_json(file_path)

            return json_result
        except (JSONDecodeError1, JSONDecodeError2):
            if counter > 3:
                if default:
                    return default

                raise
            wait = random.randint(3, 7) * 0.1
            time.sleep(wait)
        except FileNotFoundError:
            if counter > 5:
                if default:
                    return default

                raise
            wait = random.randint(10, 15) * 0.1
            time.sleep(wait)
        finally:
            counter += 1


def remove_train_meta():
    from SRC.CORE._FUNCTIONS import PRODUCE_META_FILE_NAME
    file_path = PRODUCE_META_FILE_NAME()  # Replace with the desired file path
    os.remove(file_path)


def wrire_train_meta(data, suffix=None):
    from SRC.CORE._FUNCTIONS import PRODUCE_META_FILE_NAME
    import json

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    json_data = json.dumps(data, cls=NpEncoder)

    file_path = PRODUCE_META_FILE_NAME(suffix)  # Replace with the desired file path
    with open(file_path, "w") as json_file:
        json_file.write(json_data)


def read_train_meta(meta_suffix=None):
    from SRC.CORE._FUNCTIONS import PRODUCE_META_FILE_NAME
    import json

    file_path = PRODUCE_META_FILE_NAME(meta_suffix)  # Replace with the desired file path

    with open(file_path, "r") as json_file:
        json_data = json_file.read()

    deserialized_data = json.loads(json_data)

    return deserialized_data


def save_model(cnn, meta_suffix, model_suffix):
    import torch

    from SRC.CORE._FUNCTIONS import PRODUCE_MODEL_FILE_NAME

    file_path = PRODUCE_MODEL_FILE_NAME(meta_suffix, model_suffix)

    torch.save(cnn.state_dict(), file_path)


def get_saved_model(model, meta_suffix, model_suffix):
    import torch
    from SRC.CORE.debug_utils import get_processing_device
    from SRC.CORE._FUNCTIONS import PRODUCE_MODEL_FILE_NAME

    file_path = PRODUCE_MODEL_FILE_NAME(meta_suffix, model_suffix)

    device, device_name_formatted = get_processing_device()
    model.load_state_dict(torch.load(file_path, map_location=device))
    model.to(device)

    print_action_title_description__low('GET SAVED MODEL', f'file = {file_path}')

    return model


def produce_timedelta_ticks(epoch_exec_time_s, tick_count_max):
    def timedelta_to_seconds(td):
        try:
            return td.total_seconds()
        except Exception as ex:
            pass

    min_secs = int(min(epoch_exec_time_s, key=timedelta_to_seconds).total_seconds()) - 1
    max_secs = int(max(epoch_exec_time_s, key=timedelta_to_seconds).total_seconds()) + 1

    num_intervals = int(max_secs - min_secs) + 2
    time_intervals = [timedelta(seconds=int(min_secs)).total_seconds() + i for i in range(num_intervals)]

    epoch_exec_time_sec_s = list(map(lambda td: td.total_seconds(), epoch_exec_time_s))
    take_each_n = int(len(time_intervals) / tick_count_max) if len(time_intervals) / 20 > 1 else 1

    tickvals = time_intervals[::take_each_n]
    ticktext = list(map(lambda m_s: f"{int(m_s[0]):02}:{int(m_s[1]):02}", map(lambda s: divmod(s, 60), tickvals)))

    return epoch_exec_time_sec_s, tickvals, ticktext


def format_floats(item):
    if isinstance(item, float):
        return "{:.10f}".format(item)
    elif isinstance(item, list):
        return [format_floats(x) for x in item]
    elif isinstance(item, dict):
        return {k: format_floats(v) for k, v in item.items()}
    else:
        return item


def save_trading_result(trade_symbol, target_candles_df, result_df_s, file_suffix=datetime.now()):
    from SRC.CORE._FUNCTIONS import PRODUCE_TRADE_FILE_PATH

    for key, df in result_df_s.items():
        is_simulation = 'sim' in key

        act_class_s = df['act_prev_class_s'].to_list()
        pred_class_s = df['pred_class_s'].to_list()

        state_s = df['state_s'].to_list()
        action_s = df['action_s'].to_list()

        price_s = df['price_s'].to_list()
        close_dt_s = df['close_dt_s'].to_list()

        stable_balance_s = df['stable_balance_s'][df['state_s'] == STATE_OUT].to_list()
        stable_date_s = df['close_dt_s'][df['state_s'] == STATE_OUT].to_list()

        alt_balance_s = df['alt_balance_s'][df['state_s'] == STATE_IN].to_list()
        alt_date_s = df['close_dt_s'][df['state_s'] == STATE_IN].to_list()

        if len(alt_balance_s) == 0 or len(stable_balance_s) == 0:
            continue

        min_date = stable_date_s[0] if stable_date_s[0] < alt_date_s[0] else alt_date_s[0]
        max_date = stable_date_s[-1] if stable_date_s[-1] > alt_date_s[-1] else alt_date_s[-1]

        save_symbol = trade_symbol.replace("/", "-")
        file_path = PRODUCE_TRADE_FILE_PATH(is_simulation, save_symbol, min_date, max_date, file_suffix)

        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return json.JSONEncoder.default(self, obj)

        data = {
            'act_class_s': act_class_s,
            'pred_class_s': pred_class_s,

            'state_s': state_s,
            'action_s': action_s,

            'stable_balance_s': stable_balance_s,
            'stable_date_s': list(map(lambda dt: dt.isoformat(), stable_date_s)),

            'alt_balance_s': alt_balance_s,
            'alt_date_s': list(map(lambda dt: dt.isoformat(), alt_date_s))
        }

        make_dir(TRADE_FOLDER_PATH(is_simulation))
        json_data = json.dumps(data, cls=NpEncoder)
        with open(file_path, "w") as json_file:
            json_file.write(json_data)


def read_trading_result(trade_symbol, min_date, max_date, is_simulation=False):
    from SRC.CORE._FUNCTIONS import PRODUCE_TRADE_FILE_PATH

    save_symbol = trade_symbol.replace("/", "-")
    file_path = PRODUCE_TRADE_FILE_PATH(is_simulation, save_symbol, min_date, max_date)

    with open(file_path, "r") as json_file:
        data = json.load(json_file)

    data["stable_date_s"] = list(map(lambda iso_dt: datetime.fromisoformat(iso_dt), data["stable_date_s"]))
    data["alt_date_s"] = list(map(lambda iso_dt: datetime.fromisoformat(iso_dt), data["alt_date_s"]))

    act_class_s = data["act_prev_class_s"]
    pred_class_s = data["pred_class_s"]
    state_s = data["state_s"]
    action_s = data["action_s"]

    stable_balance_s = data["stable_balance_s"]
    stable_date_s = data["stable_date_s"]

    alt_balance_s = data["alt_balance_s"]
    alt_date_s = data["alt_date_s"]

    return act_class_s, pred_class_s, state_s, action_s, stable_balance_s, stable_date_s, alt_balance_s, alt_date_s


def print_extracted_current_balance(actor_name, balance, action, state, last_price, last_close_dt, fee=None):
    stable_coin_symbol = balance[STABLE_COIN_KEY][SYMBOL_PROCESS_KEY]
    stable_coin_balance = balance[STABLE_COIN_KEY][BALANCE_KEY]
    stable_coin_balance_format = format(stable_coin_balance, ".5f")

    alt_coint_symbol = balance[ALT_COIN_KEY][SYMBOL_PROCESS_KEY]
    alt_coin_balance = balance[ALT_COIN_KEY][BALANCE_KEY]
    alt_coin_balance_format = format(alt_coin_balance, ".5f")

    state_time_present = f"State: **{state}** | Time: **{datetime_h_m_s__d_m_Y(last_close_dt)}**"
    balance_present = f"{stable_coin_symbol}: **{stable_coin_balance_format}** | {alt_coint_symbol}: **{alt_coin_balance_format}**"
    action_price_fee = f"Action: **{action}** | Price: **{_float_6(last_price)}**"
    if fee is not None:
        action_price_fee = f"{action_price_fee} | Fee: **{_float_6(fee)}**"

    title = f"{actor_name} >> {action_price_fee} || {balance_present} || {state_time_present}"

    if action == ACTION_NO:
        title = title.replace("**", "")
        print(title)
        log_module(title)
    else:
        printmd(title)
        title = title.replace("**", "")
        log_module(title)


def make_dir(*path_s):
    for path in path_s:
        import os
        if not os.path.exists(path):
            os.makedirs(path)


def case_insensitive_path(path):
    parts = path.split(os.sep)
    current_level = ""

    for part in parts:
        if not current_level:
            current_level = part
            continue

        for item in os.listdir(current_level):
            if part.lower() == item.lower():
                current_level = os.path.join(current_level, item)
                break
    result = path if path.lower() != current_level.lower() else current_level

    return result


def get_item_from_list_dict(list_dict, key, val):
    items = list(filter(lambda dict: dict[key] == val, list_dict))
    if len(items) == 0:
        return None

    item = items[0]

    return item


def remove_item_from_list_dict(list_dict, key, value):
    items = [d for d in list_dict if d.get(key) != value]

    return items


def run_multi_thread(*_lambda_s, run_sequentially=False):
    if run_sequentially:
        for _lambda in _lambda_s:
            _lambda()
    else:
        thread_s = []
        for _lambda in _lambda_s:
            thread = Thread(target=_lambda)
            thread_s.append(thread)

        for thread in thread_s:
            thread.start()

        for thread in thread_s:
            thread.join()


def run_multi_process(_lambda, arg_s):
    from multiprocessing import Process

    if is_cloud():
        args_chunk_s = list(divide_list_to_chunks(arg_s, CPU_COUNT()))
        printmd(f'**CPUs {CPU_COUNT()} | CHUNKS COUNT = {len(args_chunk_s)}**')

        i = 0
        for args_chunk in args_chunk_s:
            printmd(f'**CPU CHUNK {i + 1} | ARGS COUNT = {len(args_chunk)}**')

            process_s = []
            for arg in args_chunk:
                process_s.append(Process(target=_lambda, args=(arg,)))

            for process in process_s:
                process.start()

            for process in process_s:
                process.join()

            i += 1
    else:
        for arg in arg_s:
            _lambda(arg)


def filter_ordered_duplicates(input_s):
    from SRC.CORE.utils import are_close

    output_s = []
    for input in input_s:
        if len(output_s) > 0 and are_close(output_s[-1], input, percent_tolerance=0.1):
            continue
        output_s.append(input)

    return output_s


def divide_list_to_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def datetime_Y_m_d__h_m_s(dt, tz=TZ()):
    return strftimezone(dt, "%Y-%m-%d %H:%M:%S", tz)


def datetime_m_d__h_m(dt, tz=TZ()):
    return strftimezone(dt, "%m-%d %H:%M", tz)


def datetime_Y_m_d(dt):
    return strftimezone(dt, "%Y-%m-%d", tz=TZ())


def datetime_h_m_s(dt, tz=TZ()):
    return strftimezone(dt, "%H:%M:%S", tz=tz)


def datetime_h_m_s__d_m_Y(dt, tz=TZ()):
    return strftimezone(dt, "%H:%M:%S %d-%m-%Y", tz=tz)


def datetime_h_m_s__d_m(dt, tz=TZ()):
    return strftimezone(dt, "%H:%M:%S %d-%m", tz=tz)


def datetime_h_m__d_m_y(dt, tz=TZ()):
    return strftimezone(dt, "%H:%M %d-%m-%y", tz=tz)


def datetime_h_m__d_m(dt, tz=TZ()):
    return strftimezone(dt, "%H:%M %d-%m", tz=tz)


def datetime_h_m(dt, tz=TZ()):
    return strftimezone(dt, "%H:%M", tz=tz)


def timedelta_h_m_s(time_delta):
    total_seconds = int(time_delta.total_seconds())

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    h_m_s = f"{hours:02}:{minutes:02}:{seconds:02}"

    return h_m_s


def timedelta_days_h_m_s(time_delta):
    if not time_delta:
        return None

    total_seconds = int(time_delta.total_seconds())

    days = total_seconds // (24 * 3600)
    remaining_seconds = total_seconds % (24 * 3600)

    hours = remaining_seconds // 3600
    remaining_seconds %= 3600

    minutes = remaining_seconds // 60
    seconds = remaining_seconds % 60

    days_h_m_s = f"{days:02}::{hours:02}:{minutes:02}:{seconds:02}"

    return days_h_m_s


def strftimezone(dt, pattern, tz=TZ()):
    from SRC.LIBRARIES.time_utils import localize_tz

    try:
        localized_dt = localize_tz(dt, tz)
        result = localized_dt.strftime(pattern)
    except AttributeError:
        try:
            result = (datetime.min + dt).strftime(pattern)
        except Exception as ex:
            print(f"datetime.min = {datetime.min} | dt = {dt}")
            result = 'UKNOWN'

    return result


def run_on_ui_loop(_lambda, sleep_secs=1):
    async def sent_ui():
        await asyncio.sleep(sleep_secs)

        _lambda()

    asyncio.ensure_future(sent_ui(), loop=asyncio.get_running_loop())


def filter_pairs(objects, filter_strings, key):
    filtered_list = []

    for obj in objects:
        for string in filter_strings:
            trade_symbol = obj.get(key, '')
            trade_symbol_parts = trade_symbol.split("/")
            if string == trade_symbol_parts[0] or string == trade_symbol_parts[1] or string == trade_symbol:
                filtered_list.append(obj)
                break

    return filtered_list


def get_shortable_usdt_pairs():
    import requests

    url = 'https://api.binance.com/api/v1/exchangeInfo'

    data = requests.get(url).json()
    shortable_pairs = [pair for pair in data['symbols'] if pair['isMarginTradingAllowed']]
    shortable_usdt_pairs = list(filter(lambda x: 'USDT' in x['symbol'], shortable_pairs))

    return shortable_usdt_pairs


def get_top_usdt_pairs(only_shortable=False):
    import requests

    url = 'https://api.binance.com/api/v1/ticker/24hr'
    data = requests.get(url).json()
    data.sort(key=lambda x: float(x['priceChangePercent']), reverse=True)
    top_pairs = list(filter(lambda x: 'USDT' in x['symbol'], data))
    if only_shortable:
        shortable_pairs_symbols = list(map(lambda x: x['symbol'], get_shortable_usdt_pairs()))
        top_shortable_pairs = list(filter(lambda x: x['symbol'] in shortable_pairs_symbols, top_pairs))

        return top_shortable_pairs

    return top_pairs


def sleep_forever():
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Interrupted by user")