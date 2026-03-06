import itertools
import math
import multiprocessing
import os
import random
import re
import shutil
import sys
import time
import traceback
from collections import OrderedDict, defaultdict
from datetime import datetime, timedelta
from functools import lru_cache
import multiprocessing
import pandas as pd
import binance.exceptions
import numpy as np
import pandas as pd
from IPython.core.display_functions import display
from binance.enums import HistoricalKlinesType
from dateutil import parser
from filelock import FileLock

from SRC.CORE._CONSTANTS import _SYMBOL, _DISCRETIZATION, _FUTURES, _MARGIN, _BROKEN_DF_IDX_FILE_PATH, UTC_TZ, _UTC_TIMESTAMP, _CLOSE_GRAD, _CLOSE_GRAD_NEXT, USE_GPU, _FINE_TUNE_NET, _IGNORE
from SRC.CORE._CONSTANTS import PARTITIONING_MAP, _LONG, _SHORT, _SIGNAL, SIGNAL_IGNORE, SIGNAL_LONG_OUT, SIGNAL_SHORT_OUT, SIGNAL_SHORT_IN, SIGNAL_LONG_IN, CACHED_FILE_PATH, \
    MAX_FEATURE_NAN_START_COUNT, DISCRETIZATIONS_GROUP_FILE_PATH, _KIEV_TIMESTAMP, _TIMESTAMP, GROUP_SEGMENTS_MAX_LENGTH, _SYMBOL, KIEV_TZ, _REGIME, _DASHBOARD_SEGMENT_AUTOTRADING, _SHOULD_VALIDATE_GROUP_CONSTRAINTS, USE_PROXY_CLIENT, WEIGHTS_FILE_PATH, _DASHBOARD_SEGMENT, CONFIGS_FILE_PATH, _OPEN, \
    _HIGH, _LOW, _CLOSE, project_root_dir, WEIGHTS_DIFF_DIST_DATA_MAP_FILE_PATH, __SIGNAL, __INCLUDED, __DIST, __DIFF, __DIFF_CL, \
    __DD_CL, EXCHANGE_INFO_FILE_PATH, EXCHANGE_INFO_FUTURES_FILE_PATH, EXCHANGE_INFO_FUTURES_COIN_FILE_PATH, __TPR, DATA_FOLDER_PATH, _CONFIGS_SUFFIX, _MODEL_SUFFIX, MODEL_FOLDER_PATH, _SYMBOL_SLASH, \
    _PREDICTION_BROKEN_DF_FILE_PATH, _FUTURES
from SRC.CORE._CONSTANTS import TZ
from SRC.CORE._CONSTANTS import UTC_TZ
from SRC.CORE._CONSTANTS import _MARGIN, _BROKEN_DF_IDX_FILE_PATH, STAGE4_DATA_MAP_FILE_PATH
from SRC.CORE._CONSTANTS import _UTC_TIMESTAMP, _DISCRETIZATION
from SRC.CORE.debug_utils import printmd, format_memory, produce_measure, format_df_memory, produce_parent_process_delegate, printmd_HTML, SET_SYMBOL, is_running_under_pycharm
from SRC.CORE.utils import calc_mean_rel_diff, calc_mean_abs_diff, calc_grad, write_json, read_json, datetime_h_m__d_m_y, pairwise, datetime_Y_m_d__h_m_s, read_json_safe, get_oh_clazz_map, get_one_hot_from_clazz, get_label_cl_map, hashabledict, hashablelist
from SRC.CORE.utils import featurize_lambda
from SRC.LIBRARIES.new_utils import print_populated_char_n_times, format_num, check_env_true, get_input_feature_col_s, get_threshold_col_s, find_first_list_item, remove_list_duplicates
from SRC.LIBRARIES.new_utils import string_bool, run_multi_process, tryall_delegate, merge_dicts, calc_circle_segment, slice_list_start_end, func_multi_process, split_list_into_chunks
from SRC.LIBRARIES.time_utils import TIME_DELTA, PARTITIONING, INTERVAL, INTERVAL_PARTITION, as_utc_tz, utc_now, as_kiev_tz

try:
    from shapely.geometry import Polygon, Point
except:
    from shapely import Polygon, Point


if USE_PROXY_CLIENT():
    pass


def run_fetch_cache(pair_s):
    manager = multiprocessing.Manager()
    unfinished_symbol_s = manager.list()

    for pair in pair_s:
        pair['unfinished_symbol_s'] = unfinished_symbol_s
        pair['pairs_count'] = len(pair_s)

    is_parralel_execution = len(pair_s) > 1
    run_multi_process(fetch_cache_all_unwrap, pair_s, is_parralel_execution=is_parralel_execution, finished_title=f"FETCHED CACHED", print_result_full=False)

    printmd(f"**UNFINISHED SYMBOLS [FETCH | CACHE]:** \r\n{unfinished_symbol_s}")

    return [pair for pair in pair_s if pair['symbol'] in unfinished_symbol_s]


def fetch_with_retry(pair_s, max_retries=5, wait_secs=60):
    """
    Repeatedly fetch unfinished pairs until all succeed or retry limit is reached.
    """
    for attempt in range(1, max_retries + 1):
        unfinished_pair_s = run_fetch_cache(pair_s)
        if not unfinished_pair_s:  # all done
            print(f"✅ All pairs fetched successfully on attempt {attempt}.")
            return []

        print(f"⚠️ Attempt {attempt}: still unfinished {len(unfinished_pair_s)} pairs.")
        if attempt < max_retries:
            print(f"⏳ Waiting {wait_secs}s before retrying...")
            current_wait_secs = wait_secs * attempt
            time.sleep(current_wait_secs)
            pair_s = unfinished_pair_s  # retry unfinished
        else:
            print(f"❌ Max retries ({max_retries}) reached. Unfinished pairs remains:\r\n{unfinished_pair_s}")
            return unfinished_pair_s


def fetch_cache_all_unwrap(pair):
    market_type = pair['market_type'] if 'market_type' in pair else HistoricalKlinesType.SPOT
    symbol = pair['symbol']
    discretization_s = pair['discretization_s']
    start_dt = pair['start_dt']
    end_dt = pair['end_dt']
    unfinished_symbol_s = pair['unfinished_symbol_s']
    pairs_count = pair['pairs_count']

    if pairs_count > 5:
        import random
        time.sleep(random.randint(0, 20))

    try:
        df_s = fetch_cache_all(market_type, symbol, discretization_s, start_dt, end_dt, print_out=False)
        df_highest_discr_start_dt = df_s[-1].iloc[0][_KIEV_TIMESTAMP]
        df_lowest_discr_end_dt = df_s[0].iloc[-1][_KIEV_TIMESTAMP]

        return [symbol, discretization_s, f"{datetime_Y_m_d__h_m_s(df_highest_discr_start_dt)} - {datetime_Y_m_d__h_m_s(df_lowest_discr_end_dt)}"]
    except Exception as ex:
        target_exception = ex.__cause__
        if isinstance(target_exception, binance.exceptions.BinanceAPIException):
            if target_exception.code == -1003:
                return f"ERROR: {symbol}-{discretization_s} >> TOO MANY REQUESTS >> RETRY LATER"

        print(f"####EXCEPTION#### {symbol}-{discretization_s} ####EXCEPTION####")
        print(ex)
        print(traceback.format_exc())
        sys.stdout.flush()
        unfinished_symbol_s.append(symbol)

        return None


def _market_type_binance(market_type):
    if not isinstance(market_type, HistoricalKlinesType):
        market_type = HistoricalKlinesType.FUTURES if market_type == _FUTURES else HistoricalKlinesType.SPOT

    return market_type


def _market_type_cryptobot(market_type):
    if isinstance(market_type, HistoricalKlinesType):
        market_type = _FUTURES if market_type == HistoricalKlinesType.FUTURES else _MARGIN

    return market_type


def run_retrieve_featurize_zigzagize_cache(pair_s, print_out=False):
    is_parralel_execution = len(pair_s) > 1

    manager = multiprocessing.Manager()
    unfinished_symbol_s = manager.list()
    for pair in pair_s:
        pair['unfinished_symbol_s'] = unfinished_symbol_s
        pair['print_out'] = print_out or not is_parralel_execution

    run_multi_process(retrieve_featurize_zigzagize_cache_unwrap, pair_s, is_parralel_execution=is_parralel_execution, finished_title=f"RETRIEVED FEATURIZED ZIGZAGIZED CACHED", print_result_full=False)

    printmd(f"**UNFINISHED SYMBOLS [RETRIEVE | FEATURIZE | ZIGZAGIZE | CACHE]:** \r\n{unfinished_symbol_s}")


def run_retrieve_featurize_cache(pair_s, print_out=False):
    is_parralel_execution = len(pair_s) > 1

    manager = multiprocessing.Manager()
    unfinished_symbol_s = manager.list()
    for pair in pair_s:
        pair['unfinished_symbol_s'] = unfinished_symbol_s
        pair['print_out'] = print_out or not is_parralel_execution

    run_multi_process(retrieve_featurize_cache_unwrap, pair_s, is_parralel_execution=is_parralel_execution, finished_title=f"RETRIEVED FEATURIZED CACHED", print_result_full=False)

    printmd(f"**UNFINISHED SYMBOLS [RETRIEVE | FEATURIZE | CACHE]:** \r\n{unfinished_symbol_s}")


def retrieve_featurize_zigzagize_cache_unwrap(pair):
    symbol = pair['symbol']
    discretization_s = pair['discretization_s']
    input_features = pair['input_features']
    threshold_s = pair['threshold_s']
    unfinished_symbol_s = pair['unfinished_symbol_s']
    print_out = pair['print_out']

    try:
        retrieve_featurize_zigzagize_cache_all(symbol, discretization_s, input_features, threshold_s, print_out=print_out)
    except Exception as ex:
        print(f"####EXCEPTION#### {symbol}-{discretization_s} ####EXCEPTION####")
        print(ex)
        sys.stdout.flush()
        print(traceback.format_exc())
        unfinished_symbol_s.append(symbol)

    return [symbol, discretization_s]


def retrieve_featurize_cache_unwrap(pair):
    symbol = pair['symbol']
    discretization_s = pair['discretization_s']
    unfinished_symbol_s = pair['unfinished_symbol_s']
    print_out = pair['print_out']

    try:
        retrieve_featurize_cache_all(symbol, discretization_s, print_out=print_out)
    except Exception as ex:
        with FileLock(f"{project_root_dir()}/locks/retrieve_featurize_cache.lock"):
            print(f"ERROR RETRIEVE > FEATURIZE > CACHE: {symbol} | {discretization_s}")
            traceback.print_exc()
            unfinished_symbol_s.append(symbol)

    return [symbol, discretization_s]


def run_retrieve_zigzagize_cache(pair_s):
    manager = multiprocessing.Manager()
    unfinished_symbol_s = manager.list()
    for pair in pair_s:
        pair['unfinished_symbol_s'] = unfinished_symbol_s

    is_parralel_execution = len(pair_s) > 1
    run_multi_process(retrieve_zigzagize_cache_unwrap, pair_s, is_parralel_execution=is_parralel_execution, finished_title=f"RETRIEVED ZIGZAGIZED CACHED", print_result_full=False)

    printmd(f"**UNFINISHED SYMBOLS [RETRIEVE | ZIGZAGIZE | CACHE]:** \r\n{unfinished_symbol_s}")


def retrieve_zigzagize_cache_unwrap(pair):
    symbol = pair['symbol']
    discretization_s = pair['discretization_s']
    threshold_s = pair['threshold_s']
    unfinished_symbol_s = pair['unfinished_symbol_s']

    try:
        retrieve_zigzagize_cache_all(symbol, discretization_s, threshold_s, print_out=False)
    except Exception as ex:
        print(f"####EXCEPTION#### {symbol}-{discretization_s} ####EXCEPTION####")
        print(ex)
        sys.stdout.flush()
        print(traceback.format_exc())
        unfinished_symbol_s.append(symbol)

    return [symbol, discretization_s]


def fetch_signalize_cache_unwrap(pair):
    symbol = pair['symbol']
    discretization_s = pair['discretization_s']
    start_dt = pair['start_dt']
    end_dt = pair['end_dt']

    fetch_signalize_cache_all(symbol, discretization_s, start_dt, end_dt)

    return [symbol, discretization_s]


def fetch_zigzagize_cache_unwrap(pair):
    symbol = pair['symbol']
    discretization_s = pair['discretization_s']
    threshold_s = pair['threshold_s']
    start_dt = pair['start_dt']
    end_dt = pair['end_dt']
    market_type = pair['market_type']

    fetch_zigzagize_cache_all(market_type, symbol, discretization_s, threshold_s, start_dt, end_dt)

    return [symbol, discretization_s]


def write_discretization_steps_unwrap(arg):
    discretization_s = arg['discretization_s']
    start_dt = arg['start_dt']
    end_dt = arg['end_dt']

    write_discretization_steps(discretization_s=discretization_s, start_dt=start_dt, end_dt=end_dt)


def produce_supertrand_config():
    super_trand_config = {
        'default': {
            'period': 10,
            'multiplier': 3.0
        },
        'discretization_s': {
            "5M": [
                {
                    'period': 10,
                    'multiplier': 2.0
                },
                {
                    'period': 15,
                    'multiplier': 2.0
                },
                {
                    'period': 25,
                    'multiplier': 2.0
                },
                {
                    'period': 10,
                    'multiplier': 5.0
                },
                {
                    'period': 15,
                    'multiplier': 5.0
                },
                {
                    'period': 25,
                    'multiplier': 5.0
                }
            ],
            "15M": [
                {
                    'period': 10,
                    'multiplier': 2.0
                },
                {
                    'period': 15,
                    'multiplier': 2.0
                },
                {
                    'period': 25,
                    'multiplier': 2.0
                },
                {
                    'period': 10,
                    'multiplier': 5.0
                },
                {
                    'period': 15,
                    'multiplier': 5.0
                },
                {
                    'period': 25,
                    'multiplier': 5.0
                }
            ],
            "30M": [
                {
                    'period': 10,
                    'multiplier': 3.0
                }
            ],
            "1H": [
                {
                    'period': 10,
                    'multiplier': 3.0
                }
            ],
            "2H": [
                {
                    'period': 10,
                    'multiplier': 3.0
                }
            ]
        }
    }

    return super_trand_config


def fetch_featurize_group_all(market_type, symbol, discretization_s, segments, end_dt, print_out=False):
    df_s = []
    for discretization in discretization_s:
        start_dt = end_dt - (TIME_DELTA(discretization) * (segments + MAX_FEATURE_NAN_START_COUNT() + 30))

        df = fetch(market_type, symbol, discretization, start_dt, end_dt)
        df = df.iloc[:-1]
        df_s.append(df)

    super_trand_config = produce_supertrand_config()
    df_s = featurize_all(symbol, df_s, super_trand_config=super_trand_config, print_out=print_out)
    df_dropna_tail_s = list(map(lambda df: df.dropna().tail(segments), df_s))

    if not all([len(df_dropna_tail) == segments for df_dropna_tail in df_dropna_tail_s]):
        raise RuntimeError(f"BROKEN SEGMENTS COUNT: {[len(df_dropna_tail) for df_dropna_tail in df_dropna_tail_s]}")

    return df_dropna_tail_s


def fetch_featurize_all(market_type, symbol, discretization_s, start_dt: datetime=None, end_dt: datetime=None, print_out=True):
    df_s = fetch_all(market_type, symbol, discretization_s, start_dt, end_dt, validate=True, print_out=print_out)
    super_trand_config = produce_supertrand_config()
    df_s = featurize_all(symbol, df_s, super_trand_config=super_trand_config)

    return df_s


def fetch_signalize_cache_all(market_type, symbol, discretization_s, start_dt, end_dt):
    df_s = fetch_all(market_type, symbol, discretization_s, start_dt, end_dt)
    df_s = signalize_all(symbol, df_s)

    write_cache_df(df_s)

    return df_s


def fetch_zigzagize_cache_all(market_type, symbol, discretization_s, threshold_s, start_dt, end_dt):
    df_s = fetch_all(market_type, symbol, discretization_s, start_dt, end_dt)
    df_s = zigzagize_all(symbol, df_s, threshold_s)

    write_cache_df(df_s)

    return df_s


def retrieve_zigzagize_cache_all(symbol, discretization_s, threshold_s, print_out=True):
    df_s = retrieve_all(symbol, discretization_s, print_out=print_out)
    df_s = zigzagize_all(symbol, df_s, threshold_s, print_out=print_out)

    write_cache_df(df_s, print_out=print_out)


def fetch_cache_all(market_type, symbol, discretization_s, start_dt, end_dt, print_out=True):
    df_s = fetch_all(market_type, symbol, discretization_s, start_dt, end_dt, validate=False, print_out=print_out)

    write_cache_df(df_s, print_out=print_out)

    return df_s


def retrieve_featurize_cache_all(symbol, discretization_s, print_out=True):
    df_s = retrieve_all(symbol, discretization_s, print_out=print_out)
    df_s = featurize_all(symbol, df_s, print_out=print_out)

    write_cache_df(df_s)


def filter_features_all(symbol, df_s, input_features, threshold_s, print_out=True):
    discretization_s = [df.iloc[0][_DISCRETIZATION] for df in df_s]
    original_columns_s = df_s[0].columns.to_list()
    original_total_memory = format_memory(sum([df.memory_usage(index=True).sum() for df in df_s]))

    signal_feature_s = [f"{_SIGNAL}_{threshold}" for threshold in threshold_s]
    result_feature_s = [_SYMBOL, _DISCRETIZATION, _KIEV_TIMESTAMP, _UTC_TIMESTAMP, _OPEN, _HIGH, _LOW, _CLOSE, *input_features, *signal_feature_s]

    filtered_df_s = []
    for df in df_s:
        df = df[result_feature_s]
        filtered_df_s.append(df)

    result_columns_s = filtered_df_s[0].columns.to_list()
    result_total_memory = format_memory(sum([filtered_df.memory_usage(index=True).sum() for filtered_df in filtered_df_s]))

    if print_out:
        print(f"FILTERED FEATURES: {symbol} | {discretization_s} | ORIGINAL [count: {len(original_columns_s)} | memory:{original_total_memory}] || RESULT [count: {len(result_columns_s)} | memory: {result_total_memory}]")

    return filtered_df_s


def retrieve_featurize_zigzagize_cache_all(symbol, discretization_s, input_features, threshold_s, print_out=True):
    df_s = retrieve_all(symbol, discretization_s, print_out=print_out)
    df_s = featurize_all(symbol, df_s, print_out=print_out)
    df_s = zigzagize_all(symbol, df_s, threshold_s, print_out=print_out)
    df_s = filter_features_all(symbol, df_s, input_features, [threshold_s], print_out=print_out)

    write_cache_df(df_s, print_out=print_out)


def write_discretization_steps(discretization_s, start_dt=parser.parse("2019-01-01T00:00:00Z"), end_dt=parser.parse("2026-01-01T00:00:00Z")):
    df_s = []
    for discretization in discretization_s:
        start_timedelta = pd.Timedelta(microseconds=0)
        end_timedelta = pd.Timedelta(end_dt - start_dt)
        custom_step = pd.Timedelta(TIME_DELTA(discretization))

        timedelta_rng = pd.timedelta_range(start=start_timedelta, end=end_timedelta, freq=custom_step)
        index = start_dt + timedelta_rng
        df = pd.DataFrame(index=index)
        df[_DISCRETIZATION] = discretization
        df_s.append(df)

    write_group_df(df_s)


def fetch_all(market_type, symbol, discretization_s, start_dt: datetime=None, end_dt: datetime=None, print_out=True, validate=True):
    df_s = []
    for discretization in discretization_s:
        df = fetch(market_type, symbol, discretization, start_dt, end_dt, validate=validate)
        df_s.append(df)

    if print_out:
        print(f'FETCHED: {symbol} | {discretization_s}')
        sys.stdout.flush()

    return df_s


def retrieve_all(symbol, discretization_s, start_dt: datetime=None, end_dt: datetime=None, in_out_features=None, drop_na=False, print_out=True):
    try:
        result_df_s = []
        out_discretization = discretization_s[0]
        for discretization in discretization_s:
            is_output_discretization = discretization == out_discretization

            df = retrieve(symbol=symbol, discretization=discretization)
            if in_out_features is not None:
                in_discr_features = list(in_out_features['input'][discretization])
                feature_s = [*in_discr_features, *[in_out_features['output']]] if is_output_discretization else in_discr_features
                feature_s = remove_list_duplicates([*[_SYMBOL, _DISCRETIZATION, _UTC_TIMESTAMP, _KIEV_TIMESTAMP], *feature_s])

                assert all(col in df.columns for col in feature_s), f"NOT ALL FEATURES EXISTS: {feature_s}"

                if drop_na:
                    ignore_drop_na_feature_s = [feature for feature in feature_s if 'tpr_' not in feature and 'signal_' not in feature]
                    df_nan = df.copy()
                    df = df_nan.dropna(subset=ignore_drop_na_feature_s)

                    lenght_diff = len(df_nan) - len(df)
                    assert lenght_diff <= 51 , f"UNEXPECTED NANs DETECTED [{symbol} | {discretization} | {feature_s}] | MORE THAN [51]: {lenght_diff}"
                    assert df.iloc[-1].name >= df_nan.iloc[-1].name , "ERROR > UNEXPECTED [NANs] DETECTED"
                else:
                    df = df.dropna()

                df = df[feature_s]

            if start_dt is not None:
                df = df[df[_UTC_TIMESTAMP] >= start_dt]
            if end_dt is not None:
                df = df[df[_UTC_TIMESTAMP] <= end_dt]

            result_df_s.append(df)

        if print_out:
            print(f'RETRIEVED: {symbol} | {discretization_s}')
            sys.stdout.flush()

        return result_df_s
    except KeyError as err:
        print(f"    ERROR RETRIEVE [{symbol} | {discretization_s}]:"
              f"\r\n        FEATURES ERROR: {str(err)}")

        raise


def zigzagize_all(symbol, discret_df_s, threshold_s, print_out=True):
    try:
        signal_feature_s = [f"{_SIGNAL}_{threshold}" for threshold in threshold_s]
        discretization_s = [df.iloc[0][_DISCRETIZATION] for df in discret_df_s]
        zigzagized_df_s = []
        for df in discret_df_s:
            for threshold in threshold_s:
                try:
                    df = zigzagize(df=df, threshold=threshold)
                except Exception as ex:
                    print(f"!!! FAILED ZIGZAGIZE !!!: {symbol} | {df.iloc[0][_DISCRETIZATION]} | {threshold} | {str(ex)}")
                    sys.stdout.flush()
                    raise

            all_rest_columns = [col for col in df.columns.to_list() if 'signal' not in col]
            df = df[[*all_rest_columns, *signal_feature_s]]

            zigzagized_df_s.append(df)

        if print_out:
            print(f"ZIGZAGIZED: {symbol} | {discretization_s} | {threshold_s}")
            sys.stdout.flush()

        return zigzagized_df_s
    except Exception as ex:
        print(f"!!! FAILED ZIGZAGIZE !!!: {symbol} | {str(ex)}")
        sys.stdout.flush()
        raise


def signalize_all(symbol, discret_df_s):
    signalized_discret_df_s = []
    for df in discret_df_s:
        discretization = df.iloc[0][_DISCRETIZATION]

        signal_config_s = {
            '3M': {'diff_ratio': 1.003, 'max_distance': 23},
            '5M': {'diff_ratio': 1.0035, 'max_distance': 17},
            '15M': {'diff_ratio': 1.005, 'max_distance': 12},
            '30M': {'diff_ratio': 1.007, 'max_distance': 7},
            '1H': {'diff_ratio': 1.01, 'max_distance': 5},
        }

        signal_config = signal_config_s[discretization]
        df = generate_signals(df=df, diff_ratio=signal_config['diff_ratio'], max_distance=signal_config['max_distance'])

        signalized_discret_df_s.append(df)

    print(f"SIGNALIZED: {symbol} | {[df.iloc[0][_DISCRETIZATION] for df in discret_df_s]}")
    sys.stdout.flush()

    return signalized_discret_df_s


def featurize_all(symbol, discret_df_s, super_trand_config=None, print_out=True):
    try:
        featurized_df_s = []
        for df in discret_df_s:
            discretization = df.iloc[0][_DISCRETIZATION]
            try:
                df = featurize(df=df)
                if super_trand_config is not None:
                    super_trand_discretization_config = super_trand_config['default']
                    if 'discretization_s' in super_trand_config and discretization in super_trand_config['discretization_s']:
                        super_trand_discretization_config_s = super_trand_config['discretization_s'][discretization]
                        for super_trand_discretization_config in super_trand_discretization_config_s:
                            period = super_trand_discretization_config['period']
                            multiplier = super_trand_discretization_config['multiplier']
                            df = featurize_supertrend(df, period=period, multiplier=multiplier)
                    else:
                        period = super_trand_discretization_config['period']
                        multiplier = super_trand_discretization_config['multiplier']
                        df = featurize_supertrend(df, period=period, multiplier=multiplier)
                featurized_df_s.append(df)
            except Exception as ex:
                print(f"!!! FAILED FEATURIZE !!!: {symbol} | {discretization} | {str(ex)}")
                display(df)
                sys.stdout.flush()
                raise

        if _REGIME in os.environ and os.environ[_REGIME] == _DASHBOARD_SEGMENT_AUTOTRADING:
            return featurized_df_s

        if print_out:
            print(f"FEATURIZED: {symbol} | {[df.iloc[0][_DISCRETIZATION] for df in discret_df_s]}")
            sys.stdout.flush()

        return featurized_df_s
    except Exception as ex:
        print(f"!!! FAILED FEATURIZE !!!: {symbol} | {str(ex)}")
        sys.stdout.flush()
        raise


def fetch_featurize_all_unwrap(args):
    market_type = args['market_type']
    symbol = args['symbol']
    discretization_s = args['discretization_s']
    start_dt = args.get('start_dt', None)
    end_dt = args.get('end_dt', None)

    try:
        grouped_df_s = fetch_featurize_all(
            market_type=market_type,
            symbol=symbol,
            discretization_s=discretization_s,
            start_dt=start_dt,
            end_dt=end_dt,
        )
        df = grouped_df_s[0].copy()
        return df
    except Exception as ex:
        return f"ERROR [{symbol}|{market_type}|{discretization_s}]: {str(ex)}"


def fetch_featurize(market_type, symbol_s, discretization_s, start_dt, num_workers=5):
    args = [{
        'symbol': symbol,
        'market_type': market_type,
        'discretization_s': discretization_s,
        'start_dt': start_dt
    } for symbol in symbol_s]
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(fetch_featurize_all_unwrap, args)

    df_s = []
    for res in results:
        if isinstance(res, pd.DataFrame):
            df_s.append(res)
        else:
            print(res)

    return df_s


@lru_cache(maxsize=None)
def read_df_cached(cached_file_path):
    return read_df(cached_file_path)


def read_df(cached_file_path):
    try:
        df = pd.read_csv(cached_file_path, parse_dates=[_UTC_TIMESTAMP, _KIEV_TIMESTAMP], infer_datetime_format=True)
        if type(df.iloc[0][_UTC_TIMESTAMP]).__name__ == 'str':
            df[_UTC_TIMESTAMP] = df[_UTC_TIMESTAMP].apply(lambda str_dt: datetime.fromisoformat(str_dt))
            df[_KIEV_TIMESTAMP] = df[_KIEV_TIMESTAMP].apply(lambda str_dt: datetime.fromisoformat(str_dt))
    except:
        df = pd.read_csv(cached_file_path, parse_dates=[_UTC_TIMESTAMP], infer_datetime_format=True)
        if type(df.iloc[0][_UTC_TIMESTAMP]).__name__ == 'str':
            df[_UTC_TIMESTAMP] = df[_UTC_TIMESTAMP].apply(lambda str_dt: datetime.fromisoformat(str_dt))

        df[_KIEV_TIMESTAMP] = df.apply(lambda row: as_kiev_tz(row[_UTC_TIMESTAMP]), axis=1)

    df = order_main_cols_df(df)
    df[_TIMESTAMP] = df[_UTC_TIMESTAMP]
    df.set_index(_TIMESTAMP, inplace=True)
    df.index = pd.to_datetime(df.index)

    return df


def retrieve(symbol, discretization):
    cached_file_path = CACHED_FILE_PATH(symbol=symbol, discretization=discretization)
    df = read_df(cached_file_path)

    return df


def featurize(df):
    import pandas_ta as ta  # <-- important, this registers the `.ta` accessor

    open_col = 'open'
    high_col = 'high'
    low_col = 'low'
    close_col = 'close'

    discretization = df.iloc[0][_DISCRETIZATION]

    close_mean = df[close_col].mean()
    order_of_magnitude = int(math.floor(math.log10(abs(close_mean))))
    if order_of_magnitude < 1:
        order_of_multiply = math.pow(10, abs(order_of_magnitude))
        df['close_scaled'] = df[close_col] * order_of_multiply
    else:
        df['close_scaled'] = df[close_col]

    df.ta.sma(append=True, close=close_col, length=24)
    df.ta.sma(append=True, close=close_col, length=12)
    df.ta.sma(append=True, close=close_col, length=6)

    df.ta.rsi(append=True, close='close_scaled')
    df.ta.macd(append=True, close=close_col)
    df.ta.atr(14, append=True, close=close_col)
    df = imacd(df, high=high_col, low=low_col, close=close_col)

    del df['close_scaled']

    volume_grad_config = [
        {'feature': 'volume', 'window': 1},
        {'feature': 'volume', 'window': 2},
        {'feature': 'volume', 'window': 3},
        {'feature': 'volume', 'window': 5},
        {'feature': 'volume', 'window': 9},
    ]

    df = featurize_gradient(df, discretization, volume_grad_config)
    df, grad_diff_col_s = featurize_gradient_extremums(df, feature='volume')

    close_grad_config = [
        {'feature': close_col, 'window': 1},
        {'feature': close_col, 'window': 2},
        {'feature': close_col, 'window': 3},
        {'feature': close_col, 'window': 5},
        {'feature': close_col, 'window': 9},
    ]

    df = featurize_gradient(df, discretization, close_grad_config)
    df, grad_diff_col_s = featurize_gradient_extremums(df, feature=close_col)

    macd_grad_config = [
        {'feature': 'MACDh_12_26_9', 'window': 1},
        {'feature': 'MACDh_12_26_9', 'window': 2},
        {'feature': 'MACDh_12_26_9', 'window': 3},
        {'feature': 'MACDh_12_26_9', 'window': 5},
        {'feature': 'MACDh_12_26_9', 'window': 9}
    ]

    for feature_diff_config in macd_grad_config:
        feature = feature_diff_config['feature']
        window = feature_diff_config['window']

        df = calculate_diff(df=df, feature=feature, window=window)
        df = featurize_macd(df=df, discretization=discretization, window=window)

    rsi_grad_config = [
        {'feature': 'RSI_14', 'window': 1},
        {'feature': 'RSI_14', 'window': 2},
        {'feature': 'RSI_14', 'window': 3},
        {'feature': 'RSI_14', 'window': 5},
        {'feature': 'RSI_14', 'window': 9},
    ]

    df = featurize_gradient(df, discretization, rsi_grad_config)

    atr_grad_config = [
        {'feature': 'ATRr_14', 'window': 1},
        {'feature': 'ATRr_14', 'window': 2},
        {'feature': 'ATRr_14', 'window': 3},
        {'feature': 'ATRr_14', 'window': 5},
        {'feature': 'ATRr_14', 'window': 9},
    ]

    df = featurize_gradient(df, discretization, atr_grad_config)

    return df


def featurize_supertrend(df, period=10, multiplier=3.0):
    import pandas as pd
    import pandas_ta as ta

    hl2 = (df['high'] + df['low']) / 2
    atr = df.ta.atr(length=period, mamode='rma')

    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)

    supertrend = [np.nan] * len(df)
    direction = [True] * len(df)

    for i in range(1, len(df)):
        curr_close = hl2.iloc[i]
        prev_close = hl2.iloc[i - 1]

        if curr_close > upperband.iloc[i - 1]:
            direction[i] = True
        elif curr_close < lowerband.iloc[i - 1]:
            direction[i] = False
        else:
            direction[i] = direction[i - 1]

            if direction[i] and lowerband.iloc[i] < lowerband.iloc[i - 1]:
                lowerband.iloc[i] = lowerband.iloc[i - 1]
            if not direction[i] and upperband.iloc[i] > upperband.iloc[i - 1]:
                upperband.iloc[i] = upperband.iloc[i - 1]

        supertrend[i] = lowerband.iloc[i] if direction[i] else upperband.iloc[i]

    df['supertrend'] = supertrend
    df['supertrend_direction'] = direction
    df['supertrend_buy'] = (df['supertrend_direction'] != df['supertrend_direction'].shift(1)) & df['supertrend_direction']
    df['supertrend_sell'] = (df['supertrend_direction'] != df['supertrend_direction'].shift(1)) & ~df['supertrend_direction']

    df[f"supertrand_signal_{period}_{multiplier}"] = df.apply(lambda row: _LONG if row["supertrend_buy"] else _SHORT if row["supertrend_sell"] else _IGNORE, axis=1)

    del df['supertrend']
    del df['supertrend_direction']
    del df['supertrend_buy']
    del df['supertrend_sell']

    return df


def fetch_dirty(market_type, symbol, discretization, start_dt: datetime=None, end_dt: datetime=None):
    from SRC.LIBRARIES.binance_helpers import produce_binance_client_singleton

    market_type = _market_type_binance(market_type)

    start_dt_str = start_dt and start_dt.strftime("%d %b %Y %H:%M:%S")
    end_dt_str = end_dt and end_dt.strftime("%d %b %Y %H:%M:%S")

    tryalls_count = 5

    binance_client = produce_binance_client_singleton()

    klines = tryall_delegate(lambda: binance_client.get_historical_klines(symbol, discretization.lower(), start_dt_str, end_dt_str, klines_type=market_type), label=f"get_historical_klines > {market_type} | {symbol} || {discretization}", tryalls_count=tryalls_count)

    df = pd.DataFrame(klines, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])
    df['open_time'] = pd.to_numeric(df['open_time'], errors='coerce')
    df['close_time'] = pd.to_numeric(df['close_time'], errors='coerce')

    df['open'] = pd.to_numeric(df['open'], errors='coerce')
    df['high'] = pd.to_numeric(df['high'], errors='coerce')
    df['low'] = pd.to_numeric(df['low'], errors='coerce')
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')

    try:
        df = featurize_lambda(df, 'close_time', _UTC_TIMESTAMP, lambda ts: datetime.fromtimestamp(ts / 1000).astimezone(UTC_TZ) + timedelta(milliseconds=1))

        df = featurize_lambda(df, 'open_time', f"open_{_UTC_TIMESTAMP}", lambda ts: datetime.fromtimestamp(ts / 1000).astimezone(UTC_TZ))
        df = featurize_lambda(df, 'close_time', f"close_{_UTC_TIMESTAMP}", lambda ts: datetime.fromtimestamp(ts / 1000).astimezone(UTC_TZ))
    except ValueError:
        print(f"ValueError symbol: {symbol} | discretization: {discretization} | start_dt: {start_dt} | end_dt: {end_dt}")
        raise

    remove_col_s = ['open_time', 'close_time', 'ignore', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av']
    df = remove_cols_from_df(df, remove_col_s)

    df[_SYMBOL] = symbol
    df[_DISCRETIZATION] = discretization

    df[_TIMESTAMP] = df[_UTC_TIMESTAMP]
    df.set_index(_TIMESTAMP, inplace=True)
    df.index = pd.to_datetime(df.index)

    del df[_UTC_TIMESTAMP]

    return df


@lru_cache(maxsize=None)
def fetch_cached(market_type, symbol, discretization, start_dt: datetime=None, end_dt: datetime=None, validate=True):
    return fetch(market_type, symbol, discretization, start_dt=start_dt, end_dt=end_dt, validate=validate)


def fetch(market_type, symbol, discretization, start_dt: datetime=None, end_dt: datetime=None, validate=True):
    df = fetch_dirty(market_type, symbol, discretization, start_dt, end_dt)

    start_timedelta = pd.Timedelta(microseconds=0)
    end_timedelta = pd.Timedelta(df.index.max() - df.index.min())
    discretization_step = pd.Timedelta(TIME_DELTA(discretization))
    time_delta_range = pd.timedelta_range(start=start_timedelta, end=end_timedelta, freq=discretization_step)
    date_range = [df.index.min() + time_delta for time_delta in time_delta_range]
    df_reindexed = df.reindex(date_range)

    for column in [column for column in list(df_reindexed.columns) if column not in [_SYMBOL, _DISCRETIZATION, _UTC_TIMESTAMP, _KIEV_TIMESTAMP]]:
        df_reindexed[column] = df_reindexed[column].interpolate(method='linear')

    df_reindexed[_SYMBOL] = symbol
    df_reindexed[_DISCRETIZATION] = discretization
    df_reindexed[_UTC_TIMESTAMP] = df_reindexed.index
    df_reindexed = featurize_lambda(df_reindexed, _UTC_TIMESTAMP, _KIEV_TIMESTAMP, lambda utc_ts: as_kiev_tz(utc_ts))

    columns = df_reindexed.columns.to_list()
    columns.remove(_SYMBOL)
    columns.remove(_DISCRETIZATION)
    columns.remove(_KIEV_TIMESTAMP)
    columns.remove(_UTC_TIMESTAMP)
    desired_order = [*[_SYMBOL, _DISCRETIZATION, _KIEV_TIMESTAMP, _UTC_TIMESTAMP], *columns]
    df_reindexed = df_reindexed[desired_order]

    if validate and 'df' in locals():
        print_out = not (_DASHBOARD_SEGMENT in os.environ and _DASHBOARD_SEGMENT_AUTOTRADING in os.environ[_DASHBOARD_SEGMENT])
        validate_timeseries_df(df_reindexed, _UTC_TIMESTAMP, origin_df=df, print_out=print_out)

    return df_reindexed


def linear_interpolate(df, except_cols):
    symbol = df.iloc[0][_SYMBOL]
    discretization = df.iloc[0][_DISCRETIZATION]
    start_timedelta = pd.Timedelta(microseconds=0)
    end_timedelta = pd.Timedelta(df.index.max() - df.index.min())
    discretization_step = pd.Timedelta(TIME_DELTA(discretization))
    time_delta_range = pd.timedelta_range(start=start_timedelta, end=end_timedelta, freq=discretization_step)
    date_range = [df.index.min() + time_delta for time_delta in time_delta_range]
    df_reindexed = df.reindex(date_range)

    for column in [column for column in list(df_reindexed.columns) if column not in [*except_cols, *[_KIEV_TIMESTAMP]]]:
        try:
            df_reindexed[column] = df_reindexed[column].interpolate(method='linear')
        except Exception as ex:
            print(f"LINEAR INTERPOLATE ERROR: {symbol} | {column} | {str(ex)}")

    if _KIEV_TIMESTAMP in list(df_reindexed.columns):
        df_reindexed[_KIEV_TIMESTAMP] = df_reindexed.apply(lambda row: as_kiev_tz(row.name), axis=1)

    return df_reindexed


def zigzagize(df, threshold):
    from zigzag.core import peak_valley_pivots
    # if is_cloud():
    #     from zigzag import peak_valley_pivots
    # else:
    #     from zigzag_darwin import peak_valley_pivots

    _signal_threshold = f'{_SIGNAL}_{threshold}'

    df[_signal_threshold] = peak_valley_pivots(df[_CLOSE], threshold, -threshold)
    df_no_0 = df[~df[_signal_threshold].isin([0])]
    if df_no_0.iloc[-1][_signal_threshold] == df_no_0.iloc[-2][_signal_threshold]:
        df = df.iloc[:-1]

    df[_signal_threshold] = df[_signal_threshold].apply(lambda pivot: SIGNAL_LONG_IN if pivot == -1 else SIGNAL_SHORT_IN if pivot == 1 else SIGNAL_IGNORE)

    return df


def generate_signals(df, diff_ratio=1.005, max_distance=10):
    df_signal = df.copy(deep=True)
    close_col = 'close'

    df_signal[_LONG] = df_signal[close_col][(df_signal[close_col].shift(1) > df_signal[close_col]) & (df_signal[close_col].shift(-1) > df_signal[close_col])]
    df_signal[_SHORT] = df_signal[close_col][(df_signal[close_col].shift(1) < df_signal[close_col]) & (df_signal[close_col].shift(-1) < df_signal[close_col])]

    df_signal[_SIGNAL] = SIGNAL_IGNORE

    last_index = None
    time_delta = TIME_DELTA(df.iloc[0][_DISCRETIZATION])
    for idx, row in df_signal.iterrows():
        utc_timestamp = row[_UTC_TIMESTAMP]
        long_value = row[_LONG]
        short_value = row[_SHORT]
        if math.isnan(long_value) and math.isnan(short_value):
            continue

        last_s = df_signal[df_signal[_UTC_TIMESTAMP] < utc_timestamp]

        if math.isnan(short_value):  # long here
            last_short_s = last_s.dropna(subset=[_SHORT])
            if len(last_short_s) == 0:
                last_index = idx
                continue

            if row[_KIEV_TIMESTAMP] == parser.parse("2023-08-02T15:20:00Z"):
                pass

            last = last_short_s.iloc[-1]
            last_short_value = last[_SHORT]
            last_signal = last[_SIGNAL]

            if last_short_value / long_value >= diff_ratio:
                if last_signal == SIGNAL_IGNORE or last_signal == SIGNAL_LONG_OUT:
                    df_signal.at[last_index, _SIGNAL] = SIGNAL_SHORT_IN
                    df_signal.at[idx, _SIGNAL] = SIGNAL_SHORT_OUT
            else:
                non_ignore = df_signal[df_signal[_SIGNAL] != SIGNAL_IGNORE]
                if len(non_ignore) > 0:
                    last_non_ignore = non_ignore.iloc[-1]
                    last_signal = last_non_ignore[_SIGNAL]
                    last_value = last_non_ignore[_SHORT]
                    last_index = last_non_ignore.name
                    if last_signal == SIGNAL_LONG_OUT and last_value / long_value >= diff_ratio and (idx - last_index) <= time_delta * max_distance:
                        df_signal.at[last_index, _SIGNAL] = SIGNAL_SHORT_IN
                        df_signal.at[idx, _SIGNAL] = SIGNAL_SHORT_OUT
                    else:
                        # last_n = last_s.tail(max_distance)
                        # has_no_signals = last_n.iloc[0][_UTC_TIMESTAMP] >= last_non_ignore[_UTC_TIMESTAMP]
                        # if has_no_signals:
                        # 	lowest_long_row = last_n.sort_values(by=[_SHORT], ascending=[False]).iloc[0]
                        # 	lowest_long_row_indx = lowest_long_row.name
                        # 	if lowest_long_row[_SHORT] / row[_LONG] > diff_ratio:
                        # 		df_signal.at[lowest_long_row_indx, _SIGNAL] = SIGNAL_SHORT_IN
                        # 		df_signal.at[indx, _SIGNAL] = SIGNAL_SHORT_OUT

                        last_n = last_s.tail(max_distance)
                        last_n_non_ignore = last_n[last_n[_SIGNAL] != SIGNAL_IGNORE]
                        if len(last_n_non_ignore) > 0:
                            last_signal_indx = last_n_non_ignore.iloc[-1].name
                            lowest_long_row = last_n.loc[last_signal_indx:].sort_values(by=[_SHORT], ascending=[False]).iloc[0]
                        else:
                            lowest_long_row = last_n.sort_values(by=[_SHORT], ascending=[False]).iloc[0]
                        lowest_long_row_indx = lowest_long_row.name
                        if lowest_long_row[_SHORT] / row[_LONG] > diff_ratio:
                            df_signal.at[lowest_long_row_indx, _SIGNAL] = SIGNAL_SHORT_IN
                            df_signal.at[idx, _SIGNAL] = SIGNAL_SHORT_OUT
                else:
                    last_n = last_s.tail(max_distance)
                    last_n_non_ignore = last_n[last_n[_SIGNAL] != SIGNAL_IGNORE]
                    if len(last_n_non_ignore) > 0:
                        last_signal_indx = last_n_non_ignore.iloc[-1].name
                        lowest_long_row = last_n.loc[last_signal_indx:].sort_values(by=[_SHORT], ascending=[False]).iloc[0]
                    else:
                        lowest_long_row = last_n.sort_values(by=[_SHORT], ascending=[False]).iloc[0]
                    lowest_long_row_indx = lowest_long_row.name
                    if lowest_long_row[_SHORT] / row[_LONG] > diff_ratio:
                        df_signal.at[lowest_long_row_indx, _SIGNAL] = SIGNAL_SHORT_IN
                        df_signal.at[idx, _SIGNAL] = SIGNAL_SHORT_OUT
        else:  # short here
            last_long_s = last_s.dropna(subset=[_LONG])
            if len(last_long_s) == 0:
                last_index = idx
                continue

            if row[_KIEV_TIMESTAMP] == parser.parse("2023-08-02T15:20:00Z"):
                pass

            last = last_long_s.iloc[-1]
            last_long_value = last[_LONG]
            last_signal = last[_SIGNAL]
            if short_value / last_long_value >= diff_ratio:
                if last_signal == SIGNAL_IGNORE or last_signal == SIGNAL_SHORT_OUT:
                    df_signal.at[last_index, _SIGNAL] = SIGNAL_LONG_IN
                    df_signal.at[idx, _SIGNAL] = SIGNAL_LONG_OUT
            else:
                non_ignore = df_signal[df_signal[_SIGNAL] != SIGNAL_IGNORE]
                if len(non_ignore) > 0:
                    last_non_ignore = non_ignore.iloc[-1]
                    last_signal = last_non_ignore[_SIGNAL]
                    last_value = last_non_ignore[_LONG]
                    last_index = last_non_ignore.name
                    if last_signal == SIGNAL_SHORT_OUT and short_value / last_value >= diff_ratio and (idx - last_index) <= time_delta * max_distance:
                        df_signal.at[last_index, _SIGNAL] = SIGNAL_LONG_IN
                        df_signal.at[idx, _SIGNAL] = SIGNAL_LONG_OUT
                    else:
                        # last_n = last_s.tail(max_distance)
                        # has_no_signals = last_n.iloc[0][_UTC_TIMESTAMP] >= last_non_ignore[_UTC_TIMESTAMP]
                        # if has_no_signals:
                        # 	lowest_long_row = last_n.sort_values(by=[_LONG], ascending=[True]).iloc[0]
                        # 	lowest_long_row_indx = lowest_long_row.name
                        # 	if row[_SHORT] / lowest_long_row[_LONG]> diff_ratio:
                        # 		df_signal.at[lowest_long_row_indx, _SIGNAL] = SIGNAL_LONG_IN
                        # 		df_signal.at[indx, _SIGNAL] = SIGNAL_LONG_OUT

                        last_n = last_s.tail(max_distance)
                        last_n_non_ignore = last_n[last_n[_SIGNAL] != SIGNAL_IGNORE]
                        if len(last_n_non_ignore) > 0:
                            last_signal_indx = last_n_non_ignore.iloc[-1].name
                            lowest_long_row = last_n.loc[last_signal_indx:].sort_values(by=[_LONG], ascending=[True]).iloc[0]
                        else:
                            lowest_long_row = last_n.sort_values(by=[_LONG], ascending=[True]).iloc[0]

                        lowest_long_row_indx = lowest_long_row.name
                        if row[_SHORT] / lowest_long_row[_LONG] > diff_ratio:
                            df_signal.at[lowest_long_row_indx, _SIGNAL] = SIGNAL_LONG_IN
                            df_signal.at[idx, _SIGNAL] = SIGNAL_LONG_OUT
                else:
                    last_n = last_s.tail(max_distance)
                    last_n_non_ignore = last_n[last_n[_SIGNAL] != SIGNAL_IGNORE]
                    if len(last_n_non_ignore) > 0:
                        last_signal_indx = last_n_non_ignore.iloc[-1].name
                        lowest_long_row = last_n.loc[last_signal_indx:].sort_values(by=[_LONG], ascending=[True]).iloc[0]
                    else:
                        lowest_long_row = last_n.sort_values(by=[_LONG], ascending=[True]).iloc[0]

                    lowest_long_row_indx = lowest_long_row.name
                    if row[_SHORT] / lowest_long_row[_LONG] > diff_ratio:
                        df_signal.at[lowest_long_row_indx, _SIGNAL] = SIGNAL_LONG_IN
                        df_signal.at[idx, _SIGNAL] = SIGNAL_LONG_OUT

        last_index = idx

    df_signal_no_nan = df_signal[df_signal[_SIGNAL] != SIGNAL_IGNORE]
    last_index = None
    for indx, row_no_nan in df_signal_no_nan.iterrows():
        if last_index is None:
            last_index = indx
            continue
        last_row = df_signal_no_nan.loc[last_index]
        curr_row = row_no_nan
        if (last_row[_SIGNAL] == SIGNAL_SHORT_IN or last_row[_SIGNAL] == SIGNAL_LONG_OUT) and (curr_row[_SIGNAL] == SIGNAL_SHORT_IN or curr_row[_SIGNAL] == SIGNAL_LONG_OUT):
            df_between = df_signal.loc[last_row.name:curr_row.name]
            last_short = last_row[_SHORT]
            curr_short = curr_row[_SHORT]
            has_no_higher_between = len(df_between[(df_between[_SHORT] > last_short) & (df_between[_SHORT] > curr_short)]) == 0
            if has_no_higher_between and (len(df_between) <= max_distance / 2) and last_short != curr_short:
                if last_short > curr_short:
                    df_signal.at[last_index, _SIGNAL] = SIGNAL_SHORT_IN
                    df_signal.at[indx, _SIGNAL] = SIGNAL_IGNORE
                else:
                    df_signal.at[last_index, _SIGNAL] = SIGNAL_IGNORE
                    df_signal.at[indx, _SIGNAL] = SIGNAL_SHORT_IN
        if (last_row[_SIGNAL] == SIGNAL_SHORT_OUT or last_row[_SIGNAL] == SIGNAL_LONG_IN) and (curr_row[_SIGNAL] == SIGNAL_SHORT_OUT or curr_row[_SIGNAL] == SIGNAL_LONG_IN):
            df_between = df_signal.loc[last_row.name:curr_row.name]
            last_long = last_row[_LONG]
            curr_long = curr_row[_LONG]
            has_no_lower_between = len(df_between[(df_between[_LONG] < last_long) & (df_between[_LONG] < curr_long)]) == 0
            if has_no_lower_between and (len(df_between) <= max_distance / 2) and last_long != curr_long:
                if last_long < curr_long:
                    df_signal.at[last_index, _SIGNAL] = SIGNAL_LONG_IN
                    df_signal.at[indx, _SIGNAL] = SIGNAL_IGNORE
                else:
                    df_signal.at[last_index, _SIGNAL] = SIGNAL_IGNORE
                    df_signal.at[indx, _SIGNAL] = SIGNAL_LONG_IN

        last_index = indx

    return df_signal


def featurize_gradient_extremums(df, feature):
    suffix = f'{feature}_grad_'
    _grad_diff_col = lambda grad_num: f'{feature}_grad_diff_{grad_num}'

    find_pattern = fr"^{suffix}\d+$"
    value_pattern = fr"^{suffix}(\d+)$"

    gad_col_s = [grad_col for grad_col in df.columns if re.match(find_pattern, grad_col)]
    grad_diff_col_s = []
    for gad_col in gad_col_s:
        grad_num = int(re.match(value_pattern, gad_col).group(1))
        grad_diff_col = _grad_diff_col(grad_num)
        df[grad_diff_col] = np.nan

        # print(f"---- GRAD DIFF {grad_diff_col} ----")

        last_idx = df.iloc[-1].name
        for idx, row in df.dropna(subset=[gad_col]).iterrows():
            next_idx = idx + (TIME_DELTA(df.iloc[0][_DISCRETIZATION]) * grad_num)
            if next_idx > last_idx:
                break

            curr_grad = row[gad_col]
            next_grad = df.loc[next_idx][gad_col]

            curr_minus_next = (curr_grad - next_grad)
            divider = 1 + abs(abs(curr_grad) - abs(next_grad))
            grad_diff = curr_minus_next / divider

            # print(f"{datetime_Y_m_d__h_m_s(idx)} | curr: {curr_grad:.6f} | next: {next_grad:.6f} | minus: {curr_minus_next:.6f} | divider: {divider:.6f} | grad_diff: {grad_diff:.9f}")

            # df.loc[next_idx, grad_diff_col] = grad_diff
            df.loc[idx, grad_diff_col] = grad_diff

        grad_diff_col_s.append(grad_diff_col)

    return df, grad_diff_col_s


def featurize_gradient(df, discretization, grad_feature_config_s):
    for grad_feature_config in grad_feature_config_s:
        feature = grad_feature_config['feature']
        window = grad_feature_config['window']

        try:
            rolling_diff_window_freq = 2
            partitioning_map = PARTITIONING_MAP[PARTITIONING(discretization)]
            diff_window = f'{rolling_diff_window_freq * INTERVAL(discretization)}{partitioning_map}'

            abs_diff = df[feature].rolling(diff_window, min_periods=rolling_diff_window_freq).apply(calc_mean_abs_diff)
            rel_diff = abs_diff.apply(calc_mean_rel_diff)

            grad_col = f'{feature}_grad_{window}'
            grad_window = f'{window * INTERVAL(discretization)}{partitioning_map}'

            df[grad_col] = rel_diff.rolling(grad_window, min_periods=window).apply(calc_grad)
        except Exception:
            symbol = df.iloc[0][_SYMBOL]

            print(f"### EXCEPTION | symbol: {symbol} | discretization: {discretization} | feature: {feature} | window: {window} ###")

            traceback.print_exc()
            sys.stdout.flush()

            raise

    return df


def imacd(df, **kwargs):
    import pandas_ta as pta

    data = df
    open_col = kwargs.pop("open", "open")
    high_col = kwargs.pop("high", "high")
    low_col = kwargs.pop("low", "low")
    close_col = kwargs.pop("close", "close")

    lengthMA: int = 26  # input(34)
    lengthSignal: int = 9  # input(9)

    def calc_smma(src: np.ndarray, length: int) -> np.ndarray:
        """
        Calculate Smoothed Moving Average (SMMA) for a given numpy array `src` with a specified `length`.
        """
        smma = np.full_like(src, fill_value=np.nan)
        sma = pta.sma(pd.Series(src), length)

        for i in range(1, len(src)):
            smma[i] = (
                sma[i]
                if np.isnan(smma[i - 1])
                else (smma[i - 1] * (length - 1) + src[i]) / length
            )

        return smma

    def calc_zlema(src: np.ndarray, length: int) -> np.ndarray:
        """
        Calculates the zero-lag exponential moving average (ZLEMA) of the given price series.
        """
        ema1 = pta.ema(pd.Series(src), length)
        ema2 = pta.ema(pd.Series(ema1), length)
        d = ema1 - ema2

        return ema1 + d

    src = (data[high_col].to_numpy(dtype=np.double) + data[low_col].to_numpy(dtype=np.double) + data[close_col].to_numpy(dtype=np.double)) / 3
    hi = calc_smma(data[high_col].to_numpy(dtype=np.double), lengthMA)
    lo = calc_smma(data[low_col].to_numpy(dtype=np.double), lengthMA)
    mi = calc_zlema(src, lengthMA)

    md = np.full_like(mi, fill_value=np.nan)

    conditions = [mi > hi, mi < lo]
    choices = [mi - hi, mi - lo]

    md = np.select(conditions, choices, default=0)

    sb = pta.sma(pd.Series(md), lengthSignal)
    sh = md - sb

    df['iMACD'] = md
    df['iMACDh'] = sh.to_list()
    df['iMACDs'] = sb.to_list()

    return df


def calculate_diff(df, feature, window):
    # PREPARE
    x = list(range(len(df.index.to_list())))
    y = df[feature].to_list()

    def f(x):
        return y[x]

    def central_difference(f, x, w):
        f_x = f(x)
        f_x_w = f(x - w)
        # diff = (f_x - f_x_w) / ((f_x + f_x_w) / 2)
        diff = (f_x - f_x_w) / 2 * w

        return diff

    # plt.plot(x, list(map(lambda x_: f(x_), x)))

    # DIFF
    f_prime = list(map(lambda x_: central_difference(f, x_, window), x[window:]))
    df[f'{feature}_DIFF_grad_{window}'] = [*np.full(window, np.nan), *f_prime]
    # plt.plot(x, df[f'{feature}_DIFF'].to_list())

    # SLOPE
    # arctan = [np.arctan(x) for x in f_prime]
    # df[f'{feature}_SLOPE_grad_{window}'] = [*np.full(window, np.nan), *arctan]
    # plt.plot(x, df[f'{feature}_SLOPE'].to_list())

    return df


def featurize_macd(df, discretization, window):
    partitioning_map = PARTITIONING_MAP[PARTITIONING(discretization)]
    grad_window = f'{window * INTERVAL(discretization)}{partitioning_map}'

    # df['MACDh_12_26_9_S_grad_1'] = df['MACDh_12_26_9_SLOPE_grad_1'].rolling(grad_window, min_periods=window).apply(calc_grad)
    # df['MACDh_12_26_9_S_N_grad_1'] = df['MACDh_12_26_9_S_grad_1'].apply(lambda val: normalize_minus_1__plus_1(val, outer_max=max(df.dropna(subset=['MACDh_12_26_9_S_grad_1'])['MACDh_12_26_9_S_grad_1']), outer_min=min(df.dropna(subset=['MACDh_12_26_9_S_grad_1'])['MACDh_12_26_9_S_grad_1'])))

    # window = 1
    # partitioning_map = PARTITIONING_MAP[PARTITIONING(discretization)]
    # grad_window = f'{window * INTERVAL(discretization)}{partitioning_map}'
    df[f'MACDh_12_26_9_D_grad_{window}'] = df[f'MACDh_12_26_9_DIFF_grad_{window}'].rolling(grad_window, min_periods=window).apply(calc_grad)
    # df['MACDh_12_26_9_D_N_grad_1'] = df['MACDh_12_26_9_D_grad_1'].apply(lambda val: normalize_minus_1__plus_1(val, outer_max=max(df.dropna(subset=['MACDh_12_26_9_D_grad_1'])['MACDh_12_26_9_D_grad_1']), outer_min=min(df.dropna(subset=['MACDh_12_26_9_D_grad_1'])['MACDh_12_26_9_D_grad_1'])))

    # window = 1
    # partitioning_map = PARTITIONING_MAP[PARTITIONING(discretization)]
    # grad_window = f'{window * INTERVAL(discretization)}{partitioning_map}'
    df[f'MACDh_12_26_9_H_grad_{window}'] = df['MACDh_12_26_9'].rolling(grad_window, min_periods=window).apply(calc_grad)

    return df


def cache_df(arg):
    df = arg['df']
    threshold = arg['threshold']
    unfinished_data_s = arg['unfinished_data_s']

    symbol = df.iloc[0][_SYMBOL]
    discretization = df.iloc[0][_DISCRETIZATION]

    try:
        df_stored = retrieve(symbol, discretization)
        df_fresh = df.copy()

        threshold_col_s = [col for col in df.columns if f'{threshold}' in col]
        not_threshold_col_s = [col for col in df_stored.columns if f'{threshold}' not in col]

        try:
            if all([stored_col for stored_col in df_stored.columns if stored_col in threshold_col_s]):
                return f"ALREADY HAS FEATURES: {threshold_col_s}"

            df_fresh[not_threshold_col_s] = df_stored[not_threshold_col_s]
            df_fresh = df_fresh[[*not_threshold_col_s, *threshold_col_s]]

            write_cache_df([df_fresh], print_out=False)

            return df_stored
        except:
            with FileLock(f"{project_root_dir()}/locks/rewrite_differentiated_df.lock"):
                broken_differentify_folder_path = f"{project_root_dir()}/OUT/BROKEN_DIFFERENTIFY/{discretization}-{threshold}"
                os.makedirs(broken_differentify_folder_path, exist_ok=True)

                broken_differentify_file_stored_path = f"{broken_differentify_folder_path}/{symbol}-stored.csv"
                broken_differentify_file_fresh_path = f"{broken_differentify_folder_path}/{symbol}-fresh.csv"
                df_stored.to_csv(broken_differentify_file_stored_path)
                df_fresh.to_csv(broken_differentify_file_fresh_path)

                unfinished_data_s.append({
                    'symbol': symbol,
                    'discretization': discretization,
                    'threshold': threshold,
                    'df_stored': df_stored,
                    'df_fresh': df_fresh,

                })
                print(f"ERROR REWRITE: {symbol} | {discretization}")
                print(f"threshold_col_s: {threshold_col_s}")
                print(f"not_threshold_col_s: {not_threshold_col_s}")
                print(f"df_stored.cloums: {df_stored.columns}")

            return None
    except:
        with FileLock(f"{project_root_dir()}/locks/rewrite_differentiated_df.lock"):
            print(f"ERROR REWRITE: {symbol} | {discretization}")
            traceback.print_exc()

        return None


def write_cache_df(discret_df_s, drop_na=False, print_out=True):
    df_s = []
    symbol = discret_df_s[0].iloc[0][_SYMBOL]
    for df in discret_df_s:
        if drop_na:
            df = df.dropna(subset=df.columns.difference(['long', 'short']))

        file_path = CACHED_FILE_PATH(symbol=symbol, discretization=df.iloc[0][_DISCRETIZATION])
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path)
        df_s.append(df)

    discr_present = [df.iloc[0][_DISCRETIZATION] for df in df_s]
    date_constraint_present = f"{datetime_Y_m_d__h_m_s(df_s[-1].iloc[0][_UTC_TIMESTAMP])} - {datetime_Y_m_d__h_m_s(df_s[0].iloc[-1][_UTC_TIMESTAMP])}"

    if print_out:
        print(f"CACHED: {symbol} || {discr_present} || {date_constraint_present}")
        sys.stdout.flush()


def write_broken_prediction_group(group_df_s, transaction_id):
    for df in group_df_s:
        file_path = _PREDICTION_BROKEN_DF_FILE_PATH(transaction_id=transaction_id, discretization=df.iloc[0][_DISCRETIZATION])
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path)


def write_group_df(discret_df_s):
    max_segments = GROUP_SEGMENTS_MAX_LENGTH()

    discretization_s = [df.iloc[0][_DISCRETIZATION] for df in discret_df_s]

    group_constraint_s = []
    for idx in list(range(1, len(discret_df_s[0]))):
        last_end_position = None
        segment_s = []
        for discret_df in discret_df_s:
            if last_end_position is None:
                segment = discret_df.iloc[:idx]
                last_end_position = segment.iloc[-1].name
            else:
                segment = discret_df[discret_df.index <= last_end_position]

            segment_s.append(segment)

        if any(len(segment) < max_segments for segment in segment_s):
            continue

        group_constraint = {f"{discretization_s[0]}_out": segment_s[0].iloc[-1].name + TIME_DELTA(discretization_s[0])}
        # THIS:
        for segment in [seg.tail(max_segments) for seg in segment_s]:
            discretization = segment.iloc[0][_DISCRETIZATION]
            group_constraint[f"{discretization}_start"] = segment.iloc[0].name
            group_constraint[f"{discretization}_end"] = segment.iloc[-1].name
        # OR THIS:
        # for segment in segment_s:
        # 	discretization = segment.iloc[0][_DISCRETIZATION]
        # 	group_constraint[f"{discretization}_start"] = segment.iloc[-1].name - TIME_DELTA(discretization) * (max_segments - 1)
        # 	group_constraint[f"{discretization}_end"] = segment.iloc[-1].name

        for discretization in discretization_s:
            times = int((group_constraint[f'{discretization}_end'] - group_constraint[f'{discretization}_start']) / TIME_DELTA(discretization) + 1)
            assert times == max_segments, f"Times {times} is not equal to {max_segments}"

        group_constraint_s.append(group_constraint)

    discretization_s = [discret_df.iloc[0][_DISCRETIZATION] for discret_df in discret_df_s]
    group_df = pd.DataFrame(group_constraint_s)

    for feature in [feature for feature in group_df.columns if any([key in feature for key in ['start', 'end', 'out']])]:
        validate_group_constraints_df(group_df, feature, times=1 if discretization_s[0] == feature.split("_")[0] else 2)

    group_df.to_csv(DISCRETIZATIONS_GROUP_FILE_PATH(discret_s=discretization_s))

    print(f"DISCRETIZATION STEPS WRITTEN: {discretization_s} | start_dt: {discret_df_s[0].iloc[0].name} | end_dt: {discret_df_s[0].iloc[-1].name}")
    sys.stdout.flush()


@lru_cache(maxsize=None)
def read_group_df(discretization_s):
    discr_set_s = [
      ["1M", "3M", "5M", "15M", "30M", "1H", "2H", "4H", "8H"],
      ["3M", "5M", "15M", "30M", "1H", "2H", "4H", "8H"],
      ["5M", "15M", "30M", "1H", "2H", "4H", "8H"],
      ["15M", "30M", "1H", "2H", "4H", "8H"],
      ["30M", "1H", "2H", "4H", "8H"],
      ["1H", "2H", "4H", "8H"],
      ["2H", "4H", "8H"]
    ]
    discr_set = [discr_set for discr_set in discr_set_s if discr_set[0] == discretization_s[0]][0]
    segments_file_path = DISCRETIZATIONS_GROUP_FILE_PATH(discret_s=discr_set)

    out_discretization = discretization_s[0]
    parse_date_cols = [f"{out_discretization}_out"]
    for discretization in discretization_s:
        parse_date_cols.append(f"{discretization}_start")
        parse_date_cols.append(f"{discretization}_end")

    group_df = pd.read_csv(segments_file_path, parse_dates=parse_date_cols)
    group_df = group_df[parse_date_cols]
    group_df = group_df.reset_index(drop=True)
    group_df = group_df.set_index(group_df.columns[0], drop=False)
    assert group_df.iloc[0].name == group_df.iloc[0][group_df.columns[0]], "WRONG INDEXING GROUPING DATAFRAME"

    print(f"RETRIED GROUP: {discr_set} | {group_df.columns.to_list()}")

    should_validate = True if _SHOULD_VALIDATE_GROUP_CONSTRAINTS not in os.environ else string_bool(os.environ[_SHOULD_VALIDATE_GROUP_CONSTRAINTS])
    if should_validate:
        for feature in [feature for feature in group_df.columns if any([key in feature for key in ['start', 'end', 'out']])]:
            validate_group_constraints_df(group_df, feature, times=1 if out_discretization == feature.split("_")[0] else 2)

        group_constraints = list(OrderedDict.fromkeys([col.split("_")[0] for col in list(group_df.columns)[1:]]))
        print(f"VALIDATED: GROUP | {group_constraints}")
        sys.stdout.flush()

    return group_df


def get_group(row, segments, df_s):
    group = []
    for df in df_s:
        dicretization = df.iloc[0][_DISCRETIZATION]
        start = row[f"{dicretization}_start"]
        end = row[f"{dicretization}_end"]
        segment_i = df[start:end]
        group_i = segment_i.tail(segments)
        group.append(group_i)

    target_df = df_s[0]
    discretization_0 = target_df.iloc[0][_DISCRETIZATION]
    group.append(target_df.loc[row[f"{discretization_0}_out"]:].head(1))

    return group


def get_group_constraints_df(df_s, group_df, start_dt=None, end_dt=None):
    highest_discretization = df_s[-1].iloc[0][_DISCRETIZATION]
    out_discretization = df_s[0].iloc[0][_DISCRETIZATION]
    out_less_then = min([df[_UTC_TIMESTAMP].max() for df in df_s])
    start_highest_more_then = max([df[_UTC_TIMESTAMP].min() for df in df_s])
    filtered_group_df = group_df[group_df[f'{highest_discretization}_start'] >= start_highest_more_then][group_df[f'{out_discretization}_out'] <= out_less_then]

    if start_dt is not None:
        filtered_group_df = filtered_group_df[filtered_group_df[f'{out_discretization}_out'] >= as_utc_tz(start_dt)]

    if end_dt is not None:
        filtered_group_df = filtered_group_df[filtered_group_df[f'{out_discretization}_out'] <= as_utc_tz(end_dt)]

    # should_validate = True if _SHOULD_VALIDATE_GROUP_CONSTRAINTS not in os.environ else string_bool(os.environ[_SHOULD_VALIDATE_GROUP_CONSTRAINTS])
    # if should_validate:
    # 	for feature in [feature for feature in filtered_group_df.columns if any([key in feature for key in ['start', 'end', 'out']])]:
    # 		validate_group_constraints_df(group_df, feature, times=1 if out_discretization == feature.split("_")[0] else 2)
    #
    # 	group_constraints = list(OrderedDict.fromkeys([col.split("_")[0] for col in list(filtered_group_df.columns)[1:]]))
    # 	print(f"VALIDATED: GROUP | {group_constraints}")
    # 	sys.stdout.flush()

    return filtered_group_df


def save_extremums(symbol_s, discretization_s, input_feature_s, configs_suffix, num_workers=1):
    printmd(f"**CALCULATE & SAVE EXTREMUMS:**")
    print(f'{discretization_s}\r\n')
    print(f'{input_feature_s}\r\n')
    print(f'{symbol_s}\r\n')

    data_map = read_json_safe(STAGE4_DATA_MAP_FILE_PATH(suffix=configs_suffix), {})
    input_data_setup = {}
    for discretization in discretization_s:
        df_in_s = initialize_dataframe_s(symbol_s, discretization, input_feature_s, num_workers=num_workers)
        concat_df = pd.concat(df_in_s)
        feature_max_d = pd.DataFrame(concat_df[input_feature_s].abs().max()).transpose().iloc[0].to_dict()
        feature_mean_d = pd.DataFrame(concat_df[input_feature_s].mean()).transpose().iloc[0].to_dict()
        feature_std_d = pd.DataFrame(concat_df[input_feature_s].std()).transpose().iloc[0].to_dict()
        feature_stats_d = {
            k: {
                'abs_max': feature_max_d.get(k),
                'mean': feature_mean_d.get(k),
                'std': feature_std_d.get(k)
            }
            for k in feature_max_d
        }

        input_data_setup[f"{discretization}"] = feature_stats_d

    data_map = merge_dicts(data_map, {
        'input_data_setup': input_data_setup
    })

    write_json(data_map, STAGE4_DATA_MAP_FILE_PATH(suffix=configs_suffix))

    printmd(f'**EXTREMUMS SAVED**')


@lru_cache(maxsize=None)
def get_xcross_diff_dist_cl_s(diff_classes_count, dist_classes_count):
    diff_center_cl = int((diff_classes_count - 1) / 2)

    xcross_diff_dist_cl_s = []

    for diff_cl in range(diff_classes_count):
        if diff_cl == diff_center_cl:
            continue

        xcross_diff_dist_cl_s.append([diff_cl, 0])

    for dist_cl in range(1, dist_classes_count):
        xcross_diff_dist_cl_s.append([diff_center_cl, dist_cl])

    return xcross_diff_dist_cl_s


def save__weights__diff_dist_data(df_concat, diff_dist_data, threshold):
    printmd(f"**START CALCULATE & SAVE WEIGHTS & DIFF DIST DATA | STAGE 3:**")

    discretization = df_concat.iloc[0][_DISCRETIZATION]
    diff_classes_count = len(diff_dist_data['diff']['classes'])
    dist_classes_count = len(diff_dist_data['dist']['classes'])

    out_feature = __DD_CL(threshold)
    weights_labels_classes = calc_weigth_df__dd_cl(df_concat, threshold, diff_dist_data)
    discretization_feature__diff_dist_data__weights_labels_classes__map = merge_dicts({'diff_dist_data': diff_dist_data}, weights_labels_classes)

    feature_weight_diff_dist_data_map = read_json_safe(WEIGHTS_DIFF_DIST_DATA_MAP_FILE_PATH(), {})
    if discretization in feature_weight_diff_dist_data_map:
        feature_weight_diff_dist_data_map[discretization][out_feature] = discretization_feature__diff_dist_data__weights_labels_classes__map
    else:
        feature_weight_diff_dist_data_map[discretization] = {out_feature: discretization_feature__diff_dist_data__weights_labels_classes__map}

    write_json(feature_weight_diff_dist_data_map, WEIGHTS_DIFF_DIST_DATA_MAP_FILE_PATH())

    printmd(f"**{discretization}** | **{out_feature}**")

    for key, val in weights_labels_classes.items():
        printmd(f"***{key}***:\r\n{val}")

    print(f'FINISHED WEIGHTS & DIFF DIST DATA SAVED')


def save_weights(symbol_s, discretization_s, out_feature_s, suffix="", num_workers=1):
    printmd(f"**CALCULATE & SAVE WEIGHTS | STAGE 2:**")
    print(f'{discretization_s}\r\n')
    print(f'{out_feature_s}\r\n')
    print(f'{symbol_s}\r\n')

    discretization_feature_weight_map = prepare_weights_df(symbol_s, discretization_s, out_feature_s, calc_weigth_df, num_workers)

    write_json(discretization_feature_weight_map, WEIGHTS_FILE_PATH(suffix=suffix))

    printmd(f'**WEIGHTS SAVED**')


def calc_weigth_df(df_concat, discretization, out_feature):
    signal_encoder = produce_signal_encoder()

    signal_df = pd.DataFrame(tuple(df_concat[out_feature].value_counts().to_dict().items()), columns=['labels', 'counts'])
    signal_df['clazzs'] = signal_df['labels'].apply(lambda x: signal_encoder.label__clazz(x)[0])
    signal_df['weights'] = signal_df['counts'].apply(lambda x: 1 / x)
    signal_df['weights_norm'] = signal_df.apply(lambda row: row['weights'] / signal_df['weights'].sum(), axis=1)
    signal_clazz_sorted_df = signal_df.sort_values(by='clazzs')
    signal_clazz_sorted_df.set_index('clazzs', inplace=True)
    weights_map = signal_clazz_sorted_df.to_dict()

    return weights_map


def calc_weigth_df__dd_cl(df_concat, threshold, diff_dist_data):
    out_feature = __DD_CL(threshold)
    diff_cl_s, dist_cl_s = diff_dist_data['diff']['classes'], diff_dist_data['dist']['classes']

    signal_df = pd.DataFrame(tuple(df_concat[out_feature].value_counts().to_dict().items()), columns=['clazzes', 'counts'])
    signal_df['counts'] = signal_df[['counts']].apply(lambda row: 1 if row['counts'] == 0 else row['counts'], axis=1)
    existing_classes = signal_df['clazzes'].to_list()
    for dd_cl in range(get_clazzes_count(diff_dist_data['diff']['classes'], diff_dist_data['dist']['classes'])):
        if dd_cl in existing_classes:
            continue

        signal_df = pd.concat([signal_df, pd.DataFrame([{'clazzes': dd_cl, 'counts': 0}])], ignore_index=True)

    if len(signal_df['clazzes'].unique()) != len(signal_df['clazzes']):
        raise RuntimeError(f"DUPLICATE VALUES FOUND IN signal_df['clazzes']")

    signal_df['weights'] = signal_df.apply(lambda row: 0 if row['counts'] == 0 else 1 / row['counts'], axis=1)
    signal_df['weights_norm'] = signal_df.apply(lambda row: row['weights'] / signal_df['weights'].sum(), axis=1)
    signal_df['labels'] = signal_df['clazzes'].apply(lambda dd_cl: dd_cl__diff_dist_cl(diff_cl_s, dist_cl_s, dd_cl))

    signal_df = signal_df[['clazzes', 'labels', 'counts', 'weights_norm']]
    signal_clazz_sorted_df = signal_df.sort_values(by='clazzes')
    signal_clazz_sorted_df['counts'] = signal_clazz_sorted_df['counts'].astype('int64')
    signal_clazz_sorted_df['clazzes'] = signal_clazz_sorted_df['clazzes'].astype('int16')

    signal_clazz_sorted_df.set_index('clazzes', inplace=True)
    weights_labels_classes = signal_clazz_sorted_df.to_dict()

    return weights_labels_classes


def prepare_weights_df(symbol_s, discretization_s, out_feature_s, _calc_weigth_df, num_workers):
    discretization_feature_weight_map = {}
    for discretization in discretization_s:
        df_out_s = initialize_dataframe_s(symbol_s, discretization, out_feature_s, num_workers=num_workers)
        df_concat = pd.concat(df_out_s)
        feature_weight_map = {}
        for out_feature in out_feature_s:
            weights_map = _calc_weigth_df(df_concat, discretization, out_feature)
            feature_weight_map[out_feature] = weights_map

        discretization_feature_weight_map[discretization] = feature_weight_map

    return discretization_feature_weight_map


def initialize_dataframe(data):
    symbol = data[_SYMBOL]
    discretization = data[_DISCRETIZATION]
    feature_s = data['feature_s']
    unfinished_symbol_s = data['unfinished_symbol_s']

    try:
        df = retrieve_all(symbol, [discretization])[0]
        if len(feature_s) > 0:
            df = df[feature_s]

        return df
    except Exception as ex:
        printmd(f"**BROKEN** FEATURES DATA: **{symbol}** | {discretization} || {str(ex)}")

        unfinished_symbol_s.append(symbol)


def initialize_dataframe_s(symbol_s, discretization, feature_s=[], num_workers=1):
    manager = multiprocessing.Manager()
    unfinished_symbol_s = manager.list()

    init_configuration_s = [{_SYMBOL: symbol, _DISCRETIZATION: discretization, 'feature_s': feature_s, 'unfinished_symbol_s': unfinished_symbol_s} for symbol in symbol_s]
    if num_workers == 1:
        symbol_frame_s = []
        for init_configuration in init_configuration_s:
            symbol_dataframe = initialize_dataframe(init_configuration)
            symbol_frame_s.append(symbol_dataframe)

        df_s = [symbol_dataset for symbol_dataset in symbol_frame_s if symbol_dataset is not None]
    else:
        with multiprocessing.Pool(num_workers) as pool:
            symbol_frame_s = pool.map(initialize_dataframe, init_configuration_s)

            df_s = [symbol_dataset for symbol_dataset in symbol_frame_s if symbol_dataset is not None]

    if len(unfinished_symbol_s) > 0:
        printmd(f"**####FAILED SYMBOLS RETRIEVE:####**\r\n\r\n{unfinished_symbol_s}")

    printmd(f"symbols: {len(df_s)} | discretization: {discretization} | features: {feature_s}")

    return df_s


PEAK, VALLEY = 1, -1


def _identify_initial_pivot(X, up_thresh, down_thresh):
    x_0 = X[0]
    max_x = x_0
    max_t = 0
    min_x = x_0
    min_t = 0
    up_thresh += 1
    down_thresh += 1

    for t in range(1, len(X)):
        x_t = X[t]

        if x_t / min_x >= up_thresh:
            return VALLEY if min_t == 0 else PEAK

        if x_t / max_x <= down_thresh:
            return PEAK if max_t == 0 else VALLEY

        if x_t > max_x:
            max_x = x_t
            max_t = t

        if x_t < min_x:
            min_x = x_t
            min_t = t

    t_n = len(X) - 1
    return VALLEY if x_0 < X[t_n] else PEAK


def validate_symbols_discretizations_df_unwrap(data):
    symbol = data['symbol']
    discretization_s = data['discretization_s']
    unfinished_symbol_s = data['unfinished_symbol_s']
    updated_dt = data['updated_dt']
    ignore_end_date_validation = data['ignore_end_date_validation']

    try:
        df_s = retrieve_all(symbol, discretization_s, drop_na=True, print_out=False)

        for df in df_s:
            validate_timeseries_df(df, _UTC_TIMESTAMP, print_out=False)

        if not ignore_end_date_validation:
            assert all([df.iloc[-1][_UTC_TIMESTAMP] >= updated_dt for df in df_s]), "END DATE VALIDATION"

        return f"VALID TIME SERIES DATA [R: %s | F: %s | T: %s]: {symbol} | {discretization_s}"
    except Exception as ex:
        unfinished_symbol_s.append(symbol)
        return f"**BROKEN** TIME SERIES DATA [R: %s | F: %s | T: %s]: **{symbol}** | {discretization_s} || Ex: ***{str(ex)}***"


def validate_symbols_discretizations_df_s(pair_s, ignore_end_date_validation_symbol_s):
    manager = multiprocessing.Manager()
    unfinished_symbol_s = manager.list()
    for data in pair_s:
        data['unfinished_symbol_s'] = unfinished_symbol_s
        data['ignore_end_date_validation'] = data['symbol'] in ignore_end_date_validation_symbol_s

    is_parralel_execution = len(pair_s) > 1
    execution_type = 'CONCURRENT ' if is_parralel_execution else ''
    printmd(f"**START {execution_type}TIME SERIES VALIDATION [{len(pair_s)}]:** \r\n\r\n{[data['symbol'] for data in pair_s]}")

    run_multi_process(validate_symbols_discretizations_df_unwrap, pair_s, finished_title=f"VALIDATE TIMESERIES", print_result_full=False)

    printmd(f"**END {execution_type}TIME SERIES VALIDATION**")
    print(f"SYMBOLS NEED UPDATE: \r\n\r\n{unfinished_symbol_s}")


def validate_inputs_outputs_df_discr_unwrap(data):
    symbol = data['symbol']
    discretization_s = data['discretization_s']
    input_features = data['input_features']
    threshold_s = data['threshold_s']
    unfinished_symbol_s = data['unfinished_symbol_s']
    signal_feature_s = [f"{_SIGNAL}_{threshold}" for threshold in threshold_s]

    try:
        df_s = retrieve_all(symbol, discretization_s, print_out=False)

        assert all([all(col in df.columns for col in input_features) for df in df_s]), "INPUT FEATURES VALIDATION"
        assert all([all(col in df.columns for col in signal_feature_s) for df in df_s]), "SIGNAL FEATURES VALIDATION"

        return [symbol, discretization_s]
    except Exception as ex:
        unfinished_symbol_s.append(symbol)

        return f"**BROKEN** FEATURES DATA: **{symbol}** | {discretization_s} || {str(ex)}"


def validate_inputs_outputs_df(pair_s):
    manager = multiprocessing.Manager()
    unfinished_symbol_s = manager.list()

    for pair in pair_s:
        pair['unfinished_symbol_s'] = unfinished_symbol_s

    is_parralel_execution = len(pair_s) > 1
    execution_type = 'CONCURRENT ' if is_parralel_execution else ''
    printmd(f"**START {execution_type}FEATURES VALIDATION [{len(pair_s)}]:** \r\n\r\n{[pair['symbol'] for pair in pair_s]}")

    run_multi_process(validate_inputs_outputs_df_discr_unwrap, pair_s, is_parralel_execution, finished_title=f"VALIDATE FEATURES", print_result_full=False)

    printmd(f"**END {execution_type}FEATURES VALIDATION || SYMBOLS NEED UPDATE:** \r\n\r\n{unfinished_symbol_s}")


def validate_group_constraints_df(df, feature, times=1):
    df[feature] = pd.to_datetime(df[feature])
    seconds_df = df[feature].diff().iloc[1:].apply(lambda dt: dt.seconds)
    assert len(seconds_df.unique()) == times, f"{feature} | {str(seconds_df.nunique())} != {times}"

    print(f"VALIDATED: {feature} | count = {len(seconds_df.unique())} | unique = {list(seconds_df.unique())}")
    sys.stdout.flush()


def validate_timeseries_df(df, feature, origin_df=None, print_out=True):
    symbol = df.iloc[0][_SYMBOL]
    discretization = df.iloc[0][_DISCRETIZATION]
    first_utc_dt = df.iloc[0][_UTC_TIMESTAMP]
    last_utc_dt = df.iloc[-1][_UTC_TIMESTAMP]
    df[feature] = pd.to_datetime(df[feature])
    seconds_df = df[feature].diff().iloc[1:].apply(lambda dt: dt.seconds)
    assert len(seconds_df.unique()) == 1, f"{symbol} | {feature} | {discretization} | {str(seconds_df.nunique())} != {1}"

    if origin_df is not None:
        assert len(df) >= len(origin_df), f"FAILED TIMESERIES VALIDATION: {symbol} | {feature} | {discretization} | {len(df)} < {len(origin_df)}"
        assert df.iloc[0].name == origin_df.iloc[0].name, f"FAILED TIMESERIES VALIDATION: {symbol} | {feature} | {discretization} | {df.iloc[0].name} != {origin_df.iloc[0].name}"
        assert df.iloc[-1].name == origin_df.iloc[-1].name, f"FAILED TIMESERIES VALIDATION: {symbol} | {feature} | {discretization} | {df.iloc[-1].name} != {origin_df.iloc[-1].name}"

    if print_out:
        print(f"VALIDATED: {symbol} | {feature} | {discretization} | count = {len(seconds_df.unique())} | unique = {seconds_df.unique()} | first_utc_dt = {datetime_h_m__d_m_y(first_utc_dt)} | last_utc_dt = {datetime_h_m__d_m_y(last_utc_dt)}")
        sys.stdout.flush()


def produce_balance_df(balance_dict):
    date_format = "%H:%M:%S %d-%m-%Y"
    balances_df = pd.DataFrame(balance_dict)

    if len(balances_df) == 0:
        return balances_df

    balances_df.set_index('date_time', inplace=True)
    balances_df = balances_df.sort_index()

    balances_df[_KIEV_TIMESTAMP] = balances_df.apply(lambda row: as_kiev_tz(row.name), axis=1)
    balances_df[_UTC_TIMESTAMP] = balances_df.apply(lambda row: as_utc_tz(row.name), axis=1)

    balances_df["transaction_type"] = balances_df["transaction_type"].fillna(balances_df["transaction_id"].astype(str))
    balances_df["transaction_result"] = balances_df["transaction_result"].fillna(balances_df["transaction_result"].astype(str))

    balances_df = order_main_cols_df(balances_df)

    return balances_df


def remove_cols_from_df(df, col_s):
    for remove_col in col_s:
        if remove_col in df:
            del df[remove_col]

    return df


def order_main_cols_df(df):
    columns = df.columns.to_list()

    ordered_col_s = []
    if _SYMBOL in columns:
        columns.remove(_SYMBOL)
        ordered_col_s.append(_SYMBOL)

    if _DISCRETIZATION in columns:
        columns.remove(_DISCRETIZATION)
        ordered_col_s.append(_DISCRETIZATION)

    if _KIEV_TIMESTAMP in columns:
        columns.remove(_KIEV_TIMESTAMP)
        ordered_col_s.append(_KIEV_TIMESTAMP)

    if _UTC_TIMESTAMP in columns:
        columns.remove(_UTC_TIMESTAMP)
        ordered_col_s.append(_UTC_TIMESTAMP)

    desired_order = [*ordered_col_s, *columns]
    df = df[desired_order]

    return df


def init_symbol_disc_dfs(data):
    group_df = data['group_df']
    symbol = data['symbol']
    date_constraints = data['date_constraints']
    discretization_s = data['discretization_s']
    in_out_features = data['in_out_features']

    start_dt = parser.parse(date_constraints[0])
    end_dt = parser.parse(date_constraints[1])

    try:
        df_discr_s = retrieve_all(symbol, discretization_s, start_dt, end_dt, in_out_features, drop_na=True, print_out=False)
        group_constraints_df = get_group_constraints_df(df_discr_s, group_df)
        if len(group_constraints_df) < 100:
            raise AssertionError(f"GROUP CONSTRAINTS DATAFRAME LEN [{len(group_constraints_df)}] < MINIMUM LEN ALLOWED [10]")

        out_discr_feature_ts = group_constraints_df[f"{discretization_s[0]}_out"]
        actual_start_dt = out_discr_feature_ts.iloc[0]
        actual_end_dt = out_discr_feature_ts.iloc[-1]

        # _col_s = lambda df_discr_k: [col for col in df_discr_k.columns if 'included_' in col or 'diff_' in col or 'dist_' in col]
        # is_any_df_contains_nan = any([df_discr_k.drop(columns=_col_s(df_discr_k)).isnull().values.any() for df_discr_k in df_discr_s])
        # if is_any_df_contains_nan:
        #     raise RuntimeError(f"df_s is nan values")

        if check_env_true('PRINT_OUT', False):
            total_memory = format_memory(sum([df_discr.memory_usage(index=True).sum() for df_discr in df_discr_s]))
            print(f'{symbol} | {discretization_s} | samples={len(group_constraints_df)} | range: {actual_start_dt} - {actual_end_dt} | memory: {total_memory} - OK!!')
            sys.stdout.flush()

        return {'symbol': symbol, 'df_discr_s': df_discr_s, 'group_constraints_df': group_constraints_df, 'date_constraints': date_constraints, 'actual_date_range [DROPPED NANs]': f'{datetime_Y_m_d__h_m_s(actual_start_dt)} - {datetime_Y_m_d__h_m_s(actual_end_dt)}', }
    except KeyError as err:
        return None
    except AssertionError as err:
        print(f"    ERROR INIT GROUP [{symbol} | {discretization_s} | {datetime_Y_m_d__h_m_s(start_dt)} | {datetime_Y_m_d__h_m_s(end_dt)}]:"
              f"\r\n        CONSTRAINTS ERROR: {str(err)}")

        return None
    except Exception as ex:
        print(f"    ERROR INIT GROUP [{symbol} | {discretization_s} | {datetime_Y_m_d__h_m_s(start_dt)} | {datetime_Y_m_d__h_m_s(end_dt)}]:")
        traceback.print_exc()

        return None


def init_symbol_discr_dfs_s(group_df, symbol_date_constraint_s, discretization_s, in_out_features, num_workers, is_silent_load=False, print_result_full=False):
    arg_s = [{
        'group_df': group_df,
        'symbol': symbol_date_constraint['symbol'],
        'date_constraints': symbol_date_constraint['date_constraints'],
        'discretization_s': discretization_s,
        'in_out_features': in_out_features,
    } for symbol_date_constraint in symbol_date_constraint_s]

    if is_silent_load:
        if num_workers > 1 and len(arg_s) > 1:
            with multiprocessing.Pool(num_workers) as pool:
                return pool.map(init_symbol_disc_dfs, arg_s)
        else:
            return [init_symbol_disc_dfs(param) for param in arg_s]
    else:
        return func_multi_process(init_symbol_disc_dfs, arg_s, num_workers, print_result_full=print_result_full)


def modify_config_2(configs_suffix, net_cpu_empty):
    config = modify_config_1(configs_suffix=configs_suffix, net_cpu_empty=net_cpu_empty)
    for train_pair in config['train']:
        train_pair['date_constraints'][0] = config['cache_date_constraints'][0]

    return config


def modify_config_1(configs_suffix, net_cpu_empty):
    try:
        split_date_str = os.environ['SPLIT_DATE_STR']
    except:
        split_date_str = "2025-09-01T00:00:00Z"

    from SRC.CORE._CONSTANTS import CONFIGS_FILE_PATH
    from SRC.CORE.utils import read_json

    config = read_json(CONFIGS_FILE_PATH(suffix=configs_suffix))
    exclude_modify_symbol_s = config['exclude_modify_symbol_s']
    exclude_from_train_symbol_s = config['exclude_symbol_s']
    
    printmd(f"SPLIT DATE TIME [TRAIN/TEST]: **{split_date_str}**")

    return modify_config(config, net_cpu_empty, split_date_str, exclude_modify_symbol_s, exclude_from_train_symbol_s)


def modify_config(config, net_cpu_empty, split_date_str, exclude_modify_symbol_s, exclude_from_train_symbol_s):
    existing_test_pair_s = config['test']

    test_pair_s = []
    train_pair_s = [pair for pair in config['train'] if pair['symbol'] not in exclude_from_train_symbol_s]
    for train_pair in train_pair_s:
        if train_pair['symbol'] in exclude_modify_symbol_s:
            continue

        if train_pair['symbol'] in [existing_test_pair['symbol'] for existing_test_pair in existing_test_pair_s]:
            continue

        train_pair["date_constraints"][1] = split_date_str
        test_pair = {
            "symbol": train_pair["symbol"],
            "date_constraints": [split_date_str, "2026-01-01T00:00:00Z"]
        }

        if train_pair['symbol'] not in [existing_test_pair['symbol'] for existing_test_pair in existing_test_pair_s]:
            train_pair["date_constraints"][1] = split_date_str

        test_pair_s.append(test_pair)

    config['train'] = train_pair_s
    config['test'] = [*test_pair_s, *existing_test_pair_s]

    for test_pair in config['test']:
        train_pair = find_first_list_item(config['train'], 'symbol', test_pair['symbol'])
        if train_pair is not None:
            original_test_start_dt = parser.parse(test_pair['date_constraints'][0])
            original_train_start_dt = parser.parse(train_pair['date_constraints'][0])
            required_start_dt = original_test_start_dt - net_cpu_empty.segments_count() * TIME_DELTA(net_cpu_empty.discretization_s()[-1])
            if required_start_dt < original_train_start_dt:
                continue

            test_pair['date_constraints'][0] = str(required_start_dt)

    return config


def delete_symbol_discrs_folder(symbol):
    folder_path = f"{project_root_dir()}/DATA/{symbol}"
    shutil.rmtree(folder_path)


def delete_symbol_discrs_folder_s(symbol_s):
    for symbol in symbol_s:
        delete_symbol_discrs_folder(symbol)


def delete_symbol_discrs_folder_except_symbol_s(excluded_symbol_s):
    directory_path = f"{project_root_dir()}/DATA/"
    all_symbol_s = [name for name in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, name))]

    for symbol in all_symbol_s:
        if symbol in excluded_symbol_s:
            continue

        delete_symbol_discrs_folder(symbol)


def get_data_symbol_s():
    directory_path = f"{project_root_dir()}/DATA/"
    data_symbol_s = [name for name in os.listdir(directory_path) if name != '___' and os.path.isdir(os.path.join(directory_path, name))]

    return data_symbol_s


def download_symbol_folder_s(configs_suffix, num_workers):
    measure = produce_measure("STARTED DOWNLOAD CONFIG SYMBOLS FOLDER", print_on_start=True)

    # from SRC.CORE.cloud_storage import download_configs_symbols
    # from SRC.CORE.google_drive_api import download_configs_symbols
    #
    # os.environ['WORKSPACE'] = f'{project_root_dir()}'
    #
    # os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = f"{project_root_dir()}/secret-timing-381413-8ffbaa088d55.json"
    # download_configs_symbols(configs_suffix, num_workers=num_workers)
    #
    # # !du -sh /$WORKSPACE
    # subprocess.run([f"du -sh /$WORKSPACE"], shell=True)
    #
    # measure("FINISHED DOWNLOAD CONFIG SYMBOLS FOLDER")


def delete_symbol_folder_s():
    data_symbol_folder_s = get_data_symbol_s()
    for data_symbol_folder in data_symbol_folder_s:
        shutil.rmtree(f"{project_root_dir()}/DATA/{data_symbol_folder}")

############## START ############## DIFF DIST CLASSES-BINS-LABELS-OHs ############### START #############
def get__clazz_bin_constraint__by__bins_val(bins, _condition_lambda):
    idx = 0
    for start_bin, end_bin in pairwise(list(bins)):
        if _condition_lambda(start_bin, end_bin):
            return idx, start_bin, end_bin

        idx += 1

    raise RuntimeError(f"NOT MATCHED ANY BINS: {bins}")


def get__clazz__diff_bins(diff_bin_s, diff):
    if pd.isna(diff):
        return np.nan

    try:
        return get__clazz_bin_constraint__by__bins_val(diff_bin_s, lambda start_bin, end_bin: start_bin < diff < end_bin)
    except RuntimeError as err:
        print(f"DIFF VALUE = {diff}")

        raise err


def get__clazz__dist_bins(dist_bin_s, dist):
    if pd.isna(dist):
        return np.nan

    try:
        return get__clazz_bin_constraint__by__bins_val(dist_bin_s, lambda start_bin, end_bin: start_bin <= dist < end_bin)
    except RuntimeError as err:
        print(f"DIST VALUE = {dist}")

        raise err


def get_clazzes_count(diff_cl_s, dist_cl_s):
    clazzes_count = ((len(diff_cl_s) - 1) * (len(dist_cl_s) - 1)) + 1

    return clazzes_count


def get_diff_center_cl(diff_cl_s):
    diff_center_cl = int((len(diff_cl_s) - 1) / 2)

    return diff_center_cl


def get_ignore_clazz(diff_cl_s, dist_cl_s):
    clazzes_count = get_clazzes_count(diff_cl_s=diff_cl_s, dist_cl_s=dist_cl_s)
    ignore_clazz = int((clazzes_count - 1) / 2)

    return ignore_clazz


def get_ignore_diff_dist_cl(diff_cl_s):
    diff_ignore_cl = get_diff_center_cl(diff_cl_s=diff_cl_s)
    dist_ignore_cl = 0
    diff_dist_ignore_cl = [diff_ignore_cl, dist_ignore_cl]

    return diff_dist_ignore_cl


def get_oh_map(diff_cl_s, dist_cl_s):
    clazzes_count = get_clazzes_count(diff_cl_s=diff_cl_s, dist_cl_s=dist_cl_s)
    oh_map = get_oh_clazz_map(clazzes_count=clazzes_count)

    return oh_map


def get_label_map(labels):
    label_map = get_label_cl_map(hashabledict(labels))

    return label_map


def get_symmetric_diff_dist_cl_s(diff_cl_s, short_diff_dist_cl_s):
    short_diff_dist_cl_s = short_diff_dist_cl_s
    long_diff_dist_cl_s = calc_symmetric_long_diff_dist_cl_s(diff_cl_s, short_diff_dist_cl_s)
    symmetric_diff_dist_cl_s = [*short_diff_dist_cl_s, *long_diff_dist_cl_s]

    return symmetric_diff_dist_cl_s


def get_symmetric_clazz_s(clazzes_count, short_clazz_s):
    long_clazz_s = []
    for short_clazz in short_clazz_s:
        long_clazz = clazzes_count - short_clazz
        long_clazz_s.append(long_clazz)

    symmetric_clazz_s = [*short_clazz_s, *long_clazz_s]

    return symmetric_clazz_s


def calc_symmetric_long_diff_dist_cl_s(diff_cl_s, short_diff_dist_cl_s):
    center_diff_cl = int((len(diff_cl_s) - 1) / 2)
    long_diff_dist_cl_s = []
    for short_diff_dist_cl in short_diff_dist_cl_s:
        short_diff_cl = short_diff_dist_cl[0]
        short_dist_cl = short_diff_dist_cl[1]
        diff_offset = center_diff_cl - short_diff_cl
        long_diff_cl = center_diff_cl + diff_offset
        long_diff_dist_cl = [long_diff_cl, short_dist_cl]
        long_diff_dist_cl_s.append(long_diff_dist_cl)

    return long_diff_dist_cl_s


def get_clazz_map(diff_cl_s, dist_cl_s):
    xcross_diff_dist_cl_s = get_xcross_diff_dist_cl_s(len(diff_cl_s), len(dist_cl_s))
    clazz_map = np.zeros((len(diff_cl_s), len(dist_cl_s)))
    for diff_cl in diff_cl_s:
        for dist_cl in dist_cl_s:
            if [diff_cl, dist_cl] not in xcross_diff_dist_cl_s:
                clazz = diff_dist_cl__dd_cl(diff_cl_s=diff_cl_s, dist_cl_s=dist_cl_s, diff_dist_cl=[diff_cl, dist_cl])
                clazz_map[diff_cl, dist_cl] = clazz
            else:
                clazz_map[diff_cl, dist_cl] = np.nan

    return clazz_map


def dd_cl__diff_dist_cl(diff_cl_s, dist_cl_s, dd_cl, is_included=True):
    ignore_clazz = get_ignore_clazz(diff_cl_s=diff_cl_s, dist_cl_s=dist_cl_s)
    if not is_included or dd_cl == ignore_clazz:
        diff_dist_ignore_cl = get_ignore_diff_dist_cl(diff_cl_s=diff_cl_s)
        
        return hashablelist(diff_dist_ignore_cl)

    diff_center_cl = get_diff_center_cl(diff_cl_s=diff_cl_s)

    dd_cl_counter = 0
    try:
        for i in list(range(1, len(dist_cl_s)))[::-1]:
            for j in list(range(diff_center_cl)):
                if dd_cl_counter == dd_cl:
                    raise InterruptedError
                dd_cl_counter += 1

        dd_cl_counter += 1
        if dd_cl_counter == ignore_clazz:
            diff_dist_ignore_cl = get_ignore_diff_dist_cl(diff_cl_s=diff_cl_s)
            return diff_dist_ignore_cl

        for i in list(range(1, len(dist_cl_s))):
            for j in list(range(diff_center_cl + 1, len(diff_cl_s))):
                if dd_cl_counter == dd_cl:
                    raise InterruptedError
                dd_cl_counter += 1

    except InterruptedError:
        pass

    diff_dist_cl = [j, i]

    return hashablelist(diff_dist_cl)


# @lru_cache(maxsize=None)
def diff_dist_cl__dd_cl(diff_cl_s, dist_cl_s, diff_dist_cl, is_included=True):
    ignore_clazz = get_ignore_clazz(diff_cl_s=diff_cl_s, dist_cl_s=dist_cl_s)
    diff_dist_ignore_cl = get_ignore_diff_dist_cl(diff_cl_s=diff_cl_s)
    if not is_included or diff_dist_ignore_cl == diff_dist_cl:
        return ignore_clazz

    diff_center_cl = get_diff_center_cl(diff_cl_s=diff_cl_s)
    clazzes_count = get_clazzes_count(diff_cl_s=diff_cl_s, dist_cl_s=dist_cl_s)

    dist_cl = diff_dist_cl[1]
    diff_cl = diff_dist_cl[0]

    xcross_diff_dist_cl_s = get_xcross_diff_dist_cl_s(len(diff_cl_s), len(dist_cl_s))
    if [diff_cl, dist_cl] in xcross_diff_dist_cl_s:
        raise RuntimeError(f"UNABLE TO CALCULATE CLAZZ for: {[diff_cl, dist_cl]}")

    dd_cl = 0
    try:
        for i in list(range(1, len(dist_cl_s)))[::-1]:
            for j in list(range(diff_center_cl)):
                if i == dist_cl and j == diff_cl:
                    raise InterruptedError
                dd_cl += 1

        dd_cl += 1
        if dd_cl == ignore_clazz:
            raise InterruptedError

        for i in list(range(1, len(dist_cl_s))):
            for j in list(range(diff_center_cl + 1, len(diff_cl_s))):
                if i == dist_cl and j == diff_cl:
                    raise InterruptedError
                dd_cl += 1
    except InterruptedError:
        pass

    assert dd_cl < clazzes_count, f"`{dd_cl}` [DATA] >= {clazzes_count} [CONFIG]"

    return dd_cl


# @lru_cache(maxsize=None)
def diff_dist_bin__dd_cl(diff_cl_s, dist_cl_s, diff_bin_s, dist_bin_s, diff_dist, is_included=True):
    ignore_clazz = get_ignore_clazz(diff_cl_s=diff_cl_s, dist_cl_s=dist_cl_s)
    if not is_included:
        return ignore_clazz

    diff = diff_dist[0]
    dist = diff_dist[1]
    if np.isnan(diff) and np.isnan(dist):
        return ignore_clazz

    diff_cl = int(get__clazz__diff_bins(diff_bin_s=diff_bin_s, diff=diff)[0])
    dist_cl = int(get__clazz__dist_bins(dist_bin_s=dist_bin_s, dist=dist)[0])
    diff_dist_cl = [diff_cl, dist_cl]

    dd_cl = diff_dist_cl__dd_cl(diff_cl_s=diff_cl_s, dist_cl_s=dist_cl_s, diff_dist_cl=hashablelist(diff_dist_cl), is_included=is_included)

    return dd_cl


# @lru_cache(maxsize=None)
def oh__diff_dist_cl(diff_cl_s, dist_cl_s, oh, is_included=True):
    dd_cl = oh__dd_cl(diff_cl_s=diff_cl_s, dist_cl_s=dist_cl_s, oh=oh, is_included=is_included)
    diff_dist_cl = dd_cl__diff_dist_cl(diff_cl_s=diff_cl_s, dist_cl_s=dist_cl_s, dd_cl=dd_cl, is_included=is_included)

    return hashablelist(diff_dist_cl)


# @lru_cache(maxsize=None)
def dd_cl__oh(diff_cl_s, dist_cl_s, dd_cl, is_included=True):
    if not is_included:
        ignore_clazz = get_ignore_clazz(diff_cl_s=diff_cl_s, dist_cl_s=dist_cl_s)
        oh = get_one_hot_from_clazz(ignore_clazz, get_clazzes_count(diff_cl_s=diff_cl_s, dist_cl_s=dist_cl_s))
    else:
        oh = get_one_hot_from_clazz(dd_cl, get_clazzes_count(diff_cl_s=diff_cl_s, dist_cl_s=dist_cl_s))

    return hashablelist(oh)


# @lru_cache(maxsize=None)
def oh__dd_cl(diff_cl_s, dist_cl_s, oh, is_included=True):
    if not is_included:
        ignore_clazz = get_ignore_clazz(diff_cl_s=diff_cl_s, dist_cl_s=dist_cl_s)

        return ignore_clazz

    oh__dd_cl__map = get_oh_map(diff_cl_s=diff_cl_s, dist_cl_s=dist_cl_s)
    dd_cl = oh__dd_cl__map[str(list(oh))]

    return dd_cl


# @lru_cache(maxsize=None)
def diff_dist_cl__oh(diff_cl_s, dist_cl_s, diff_dist_cl, is_included=True):
    dd_cl = diff_dist_cl__dd_cl(diff_cl_s=diff_cl_s, dist_cl_s=dist_cl_s, diff_dist_cl=diff_dist_cl, is_included=is_included)
    oh = dd_cl__oh(diff_cl_s=diff_cl_s, dist_cl_s=dist_cl_s, dd_cl=dd_cl, is_included=is_included)

    return hashablelist(oh)


def produce__diff_dist__data(df_concat, threshold, space__diff, space__dist, log=False):
    import numpy as np
    from SRC.CORE.utils import pairwise

    df_concat__ignores_excluded = df_concat[(df_concat[__SIGNAL(threshold)] == SIGNAL_IGNORE) | (df_concat[__INCLUDED(threshold)] == False)]

    diff_data = df_concat[df_concat[__INCLUDED(threshold)] == True][__DIFF(threshold)].to_list()
    dist_data = df_concat[df_concat[__INCLUDED(threshold)] == True][__DIST(threshold)].to_list()

    diff_abs_s = pd.DataFrame(np.array(diff_data)).abs()
    dist_abs_s = pd.DataFrame(np.array(dist_data)).abs()

    diff_max = diff_abs_s.max().max()
    dist_max = dist_abs_s.max().max()

    diff_desc_ordered = [round(diff, 4) for diff in sorted(diff_abs_s[0].to_list())[::-1][:10]]
    dist_desc_ordered = [int(dist) for dist in sorted(dist_abs_s[0].to_list())[::-1][:10]]

    printmd(f"DIFF ABS MAX: \r\n\r\n***{round(diff_max, 4)}***")
    printmd(f"DIST ABS MAX: \r\n\r\n***{int(dist_max)}***")

    diff_count_s, diff_bin_s = np.histogram(diff_data, bins=space__diff)
    dist_count_s, dist_bin_s = np.histogram(dist_data, bins=space__dist)

    printmd(f"DIFF BINS SPACE:")
    printmd(f"***{[round(bin, 5) for bin in diff_bin_s]}***")

    printmd(f"DIST BINS SPACE:")
    printmd(f"***{[round(bin, 5) for bin in dist_bin_s]}***")

    try:
        try:
            assert space__diff[-1] >= diff_max, f"space__diff[-1]: {space__diff[-1]} < diff_max: {diff_desc_ordered}"
        except AssertionError as err:
            traceback.print_exc()
            assert space__dist[-1] >= dist_max, f"space__dist[-1]: {space__dist[-1]} < dist_max: {dist_desc_ordered}"
        try:
            assert space__dist[-1] >= dist_max, f"space__dist[-1]: {space__dist[-1]} < dist_max: {dist_desc_ordered}"
        except AssertionError as err:
            traceback.print_exc()
            assert space__diff[-1] >= diff_max, f"space__diff[-1]: {space__diff[-1]} < diff_max: {diff_desc_ordered}"
    except AssertionError as err:
        traceback.print_exc()

    heat_map, dist_bin_s, diff_bin_s = np.histogram2d(dist_data, diff_data, bins=[dist_bin_s, diff_bin_s])
    entries_count = int(heat_map.sum())
    ignores_count = len(df_concat__ignores_excluded)
    assert len(df_concat) == entries_count + ignores_count

    dist_ignore_cl = 0
    diff_ignore_cl = int((len(diff_bin_s) - 2) / 2)
    heat_map[dist_ignore_cl, diff_ignore_cl] = ignores_count

    xcross_diff_dist_cl_s = get_xcross_diff_dist_cl_s(len(diff_bin_s) - 1, len(dist_bin_s) - 1)
    for xcross_diff_dist_cl in xcross_diff_dist_cl_s:
        xcross_diff_cl = xcross_diff_dist_cl[0]
        xcross_dist_cl = xcross_diff_dist_cl[1]

        assert heat_map[xcross_dist_cl, xcross_diff_cl] == 0

        heat_map[xcross_dist_cl, xcross_diff_cl] = np.nan

    if log:
        heat_map = np.log10(heat_map + 1)

    x_s = []
    y_s = []
    z_s = []
    hist_list = heat_map.tolist()
    for x in range(len(hist_list)):
        for y in range(len(hist_list[x])):
            x_s.append(x)
            y_s.append(y)
            z = hist_list[x][y]
            z_s.append(z)

    heat_map = np.rot90(heat_map)
    heat_map = np.flip(heat_map, axis=0)

    diff_bin_presentation_s = []
    for start_bin, end_bin in pairwise(list(diff_bin_s)):
        bin_presentation = f"[{len(diff_bin_presentation_s)}] {'{:.5f}'.format(start_bin)} | {'{:.5f}'.format(end_bin)}"
        diff_bin_presentation_s.append(bin_presentation)

    dist_bin_presentation_s = []
    for start_bin, end_bin in pairwise(list(dist_bin_s)):
        if end_bin - start_bin == 1:
            if start_bin == 0:
                bin_presentation = f"[{len(dist_bin_presentation_s)}]"
            else:
                bin_presentation = f"[{len(dist_bin_presentation_s)}] {int(start_bin)}"
        else:
            bin_presentation = f"[{len(dist_bin_presentation_s)}] {int(start_bin)}-{int(end_bin) - 1}"
        dist_bin_presentation_s.append(bin_presentation)

    diff_cl_s = list(range(len(diff_bin_presentation_s)))
    diff_label_s = diff_bin_presentation_s

    dist_cl_s = list(range(len(dist_bin_presentation_s)))
    dist_label_s = dist_bin_presentation_s

    clazz_map = get_clazz_map(diff_cl_s, dist_cl_s)

    diff_dist_data = {
        'heat_map': heat_map,
        'clazz_map': clazz_map,
        'bar_3D': {
            'x_s': x_s,
            'y_s': y_s,
            'z_s': z_s
        },
        'dist': {
            'counts': dist_count_s,
            'bins': dist_bin_s,
            'classes': dist_cl_s,
            'labels': dist_label_s
        },
        'diff': {
            'counts': diff_count_s,
            'bins': diff_bin_s,
            'classes': diff_cl_s,
            'labels': diff_label_s,
        }
    }

    return diff_dist_data


def calc_feature_power_space(df_concat, feature, classes, power_degree, space_start=None, non_linearity_top=None, space_top=None):
    from SRC.CORE.utils import calc_symmetric_pow_space

    space_top = (space_top if space_top is not None else df_concat[feature].abs().max()) * 1.01
    non_linearity_top = space_top if non_linearity_top is None else non_linearity_top
    pow_space__diff = calc_symmetric_pow_space(classes, power_degree=power_degree, power_start=space_start, non_linearity_top=non_linearity_top, space_top=space_top)

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    printmd_HTML(f"POWER BINS SPACE:")
    print([round(val, 5) for val in pow_space__diff])
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    return pow_space__diff


def calc_feature_log_space(df_concat, feature, classes, log_base, space_start=None, non_linearity_top=None, space_top=None):
    from SRC.CORE.utils import calc_symmetric_log_space

    space_top = (space_top if space_top is not None else df_concat[feature].abs().max()) * 1.01
    non_linearity_top = space_top if non_linearity_top is None else non_linearity_top
    pow_space__diff = calc_symmetric_log_space(classes, log_base=log_base, log_start=space_start, non_linearity_top=non_linearity_top, space_top=space_top)

    printmd(f"LOG BINS SPACE:")
    print([round(val, 5) for val in pow_space__diff])

    return pow_space__diff


def calc_diff_power_space(df_concat, threshold, classes, power_degree, non_linearity_top=None, max_diff=None):
    feature = __DIFF(threshold)
    power_start = threshold

    power_diff_space = calc_feature_power_space(df_concat, feature, classes, power_degree, power_start, non_linearity_top, max_diff)

    return power_diff_space


def exclude__edge_custom_classes(result_df_s, diff_dist_data, threshold, diff_edges_count, dist_edges_count, exclude_custom_short_diff_dist_cl_s=[], num_workers=1, print_out=False):
    print_count__ignores__long_out__short_out(result_df_s, threshold=threshold)

    diff_cl_s = diff_dist_data['diff']['classes']
    dist_cl_s = diff_dist_data['dist']['classes']

    include_diff_cl_s = slice_list_start_end(diff_cl_s, diff_edges_count, diff_edges_count)
    include_dist_cl_s = slice_list_start_end(dist_cl_s, 0, dist_edges_count)

    exclude_diff_cl_s = [diff_class for diff_class in diff_cl_s if diff_class not in include_diff_cl_s]
    exclude_dist_cl_s = [dist_class for dist_class in dist_cl_s if dist_class not in include_dist_cl_s]

    duplicated_ignore_diff_dist_cl_s = []
    for diff_cl in diff_cl_s:
        for dist_cl in exclude_dist_cl_s:
            duplicated_ignore_diff_dist_cl_s.append([diff_cl, dist_cl])

    for dist_cl in dist_cl_s:
        for diff_cl in exclude_diff_cl_s:
            duplicated_ignore_diff_dist_cl_s.append([diff_cl, dist_cl])

    ignore_diff_dist_cl_s = []
    for item in duplicated_ignore_diff_dist_cl_s:
        if item not in ignore_diff_dist_cl_s:
            ignore_diff_dist_cl_s.append(item)

    diff_classes_count = len(diff_cl_s)
    dist_classes_count = len(dist_cl_s)
    xcross_diff_dist_cl_s = get_xcross_diff_dist_cl_s(diff_classes_count=diff_classes_count, dist_classes_count=dist_classes_count)
    exclude_custom_diff_dist_cl_s = get_symmetric_diff_dist_cl_s(diff_dist_data['diff']['classes'], exclude_custom_short_diff_dist_cl_s)

    zero__diff_dist_cl_s = [*xcross_diff_dist_cl_s, *ignore_diff_dist_cl_s, *exclude_custom_diff_dist_cl_s]

    arg_s = [{"df": result_df, "diff_dist_data": diff_dist_data, "threshold": threshold, "exclude_cl_s": zero__diff_dist_cl_s, "print_out": print_out} for result_df in result_df_s]
    df_excluded_s = func_multi_process(exclude__diff_dist__classes__unwrap, arg_s, num_workers, f"EXCLUDED EDGE & CUSTOM CLASSES", print_result_full=False)

    print_count__ignores__long_out__short_out(df_excluded_s, threshold=threshold)

    return df_excluded_s


def produce_diff(signal, close, next_close):
    if signal == SIGNAL_IGNORE:
        diff = 0
    if signal == SIGNAL_LONG_IN:
        diff = next_close / close - 1
    if signal == SIGNAL_SHORT_IN:
        diff = -(close / next_close - 1)

    return diff


def produce_tpr(direction, close, next_close):
    if direction == SIGNAL_LONG_IN:
        diff = next_close / close - 1
    if direction == SIGNAL_SHORT_IN:
        diff = -(close / next_close - 1)

    return diff


def produce__diff_dist__values(df, threshold):
    symbol = df.iloc[0][_SYMBOL]
    discretization = df.iloc[0][_DISCRETIZATION]
    _signal = __SIGNAL(threshold)
    _included = __INCLUDED(threshold)
    _dd_cl = __DD_CL(threshold)
    _next_close = f'next_close_{threshold}'
    _diff = __DIFF(threshold)
    _dist = __DIST(threshold)

    i = 0
    for curr_idx, row in df.iterrows():
        curr_signal = row[_signal]
        curr_price = row[_CLOSE]

        if curr_signal == SIGNAL_IGNORE:
            continue

        sliced_df = df.loc[curr_idx:]

        if curr_signal == SIGNAL_SHORT_IN:
            next_signal = SIGNAL_LONG_IN

        if curr_signal == SIGNAL_LONG_IN:
            next_signal = SIGNAL_SHORT_IN

        next_signal_s = sliced_df.loc[(sliced_df[_signal] == next_signal)]
        if len(next_signal_s) == 0:
            if len(sliced_df) > 1:
                for _idx, _row in sliced_df.iloc[1:].iterrows():
                    df.loc[_idx, _signal] = SIGNAL_IGNORE
                    df.loc[_idx, _next_close] = np.nan
                    df.loc[_idx, _diff] = np.nan
                    df.loc[_idx, _dist] = np.nan

            if curr_signal == SIGNAL_LONG_IN:
                next_price = curr_price * 1.01
            if curr_signal == SIGNAL_SHORT_IN:
                next_price = curr_price / 1.01

            diff = produce_diff(curr_signal, curr_price, next_price)
            dist = 5

            df.loc[curr_idx, _next_close] = next_price
            df.loc[curr_idx, _diff] = diff
            df.loc[curr_idx, _dist] = dist

            break

        next_row = next_signal_s.head(1).squeeze()
        next_price = next_row[_CLOSE]
        next_idx = next_row[_UTC_TIMESTAMP]
        dist = int((next_idx - curr_idx) / TIME_DELTA(discretization))
        diff = produce_diff(curr_signal, curr_price, next_price)

        df.loc[curr_idx, _next_close] = next_price
        df.loc[curr_idx, _diff] = diff
        df.loc[curr_idx, _dist] = dist

        if abs(diff) < threshold:
            df.loc[curr_idx, _signal] = SIGNAL_IGNORE
            df.loc[curr_idx, _next_close] = np.nan
            df.loc[curr_idx, _diff] = np.nan
            df.loc[curr_idx, _dist] = np.nan

        i += 1

    if _included in df.columns:
        df = df.drop(columns=[_included])

    if _dd_cl in df.columns:
        df = df.drop(columns=[_dd_cl])

    return df


def produce__differential(df, threshold):
    input_feature_col_s = get_input_feature_col_s(df)
    threshold_col_s = get_threshold_col_s(df, threshold)
    df = df[[*input_feature_col_s, *threshold_col_s]]

    _signal = __SIGNAL(threshold)
    _next_close = f'next_close_{threshold}'
    _tpr = __TPR(threshold)

    i = 0
    for curr_idx, row in df.iterrows():
        sliced_df = df.loc[curr_idx:]

        curr_signal = row[_signal]
        curr_price = row[_CLOSE]

        if curr_signal == SIGNAL_IGNORE:
            next_signal_row_s = sliced_df.loc[(sliced_df[_signal] != SIGNAL_IGNORE)].head(1)
            if len(next_signal_row_s) > 0:
                next_not_ignore_signal = next_signal_row_s.squeeze()[_signal]
                direction = SIGNAL_SHORT_IN if next_not_ignore_signal == SIGNAL_LONG_IN else SIGNAL_LONG_IN

        if curr_signal == SIGNAL_SHORT_IN:
            direction = SIGNAL_SHORT_IN

        if curr_signal == SIGNAL_LONG_IN:
            direction = SIGNAL_LONG_IN

        if direction == SIGNAL_SHORT_IN:
            next_signal = SIGNAL_LONG_IN

        if direction == SIGNAL_LONG_IN:
            next_signal = SIGNAL_SHORT_IN

        next_signal_row_s = sliced_df.loc[(sliced_df[_signal] == next_signal)]
        if len(next_signal_row_s) > 0:
            next_signal_row = next_signal_row_s.head(1).squeeze()
        else:
            next_signal_row = sliced_df.tail(1).squeeze()

        next_signal_price = next_signal_row[_CLOSE]
        diff = produce_tpr(direction, curr_price, next_signal_price)
        df.loc[curr_idx, _tpr] = diff

        i += 1

    return df


def calc_count__ignores__long_out__short_out(df, threshold):
    count = len(df.loc[df[__SIGNAL(threshold)] == SIGNAL_IGNORE]) + len(df.loc[df[__INCLUDED(threshold)] == False])

    return count


def print_count__ignores__long_out__short_out(df_s, threshold):
    concat_df = pd.concat(df_s)
    ignores_count = calc_count__ignores__long_out__short_out(concat_df, threshold)
    entries_count = len(concat_df) - ignores_count
    print(f"IGNORES: {format_num(ignores_count)}")
    print(f"ENTRIES: {format_num(entries_count)}")
    sys.stdout.flush()


def check_row_included_predicate(df: pd.DataFrame, threshold):
    _included = __INCLUDED(threshold)

    check_included_available = _included in df.columns
    if check_included_available:
        return lambda row: row[_included]
    else:
        return lambda row: True


def exclude__diff_dist__classes__unwrap(data):
    df = data['df']
    diff_dist_data = data['diff_dist_data']
    threshold = data['threshold']
    exclude_cl_s = data['exclude_cl_s']
    print_out = data['print_out']

    return exclude__diff_dist__classes(df, diff_dist_data, threshold, exclude_cl_s, print_out)


def exclude__diff_dist__classes(df_: pd.DataFrame, diff_dist_data, threshold, zero__diff_dist_cl_s, print_out=False):
    symbol = df_.iloc[0][_SYMBOL]
    discretization = df_.iloc[0][_DISCRETIZATION]

    exception_idx_s = []
    df = df_.copy()

    _signal = __SIGNAL(threshold)
    _included = __INCLUDED(threshold)
    _next_close = f'next_close_{threshold}'
    _diff = __DIFF(threshold)
    _dist = __DIST(threshold)

    ignores_count = calc_count__ignores__long_out__short_out(df, threshold)
    entries_count = len(df) - ignores_count
    if print_out:
        print_populated_char_n_times("=", 100, title=f"{symbol} | {discretization}")
        print(f"IGNORES: {format_num(ignores_count)}")
        print(f"ENTRIES: {format_num(entries_count)}")
        sys.stdout.flush()

    diff_bin_s, dist_bin_s = diff_dist_data['diff']['bins'], diff_dist_data['dist']['bins']

    _is_included = check_row_included_predicate(df, threshold)
    for curr_idx, row in df.iterrows():
        curr_signal = row[_signal]
        curr_included = row[_included]
        curr_price = row[_CLOSE]
        curr_diff = row[_diff]
        curr_dist = row[_dist]

        if curr_signal == SIGNAL_IGNORE or not curr_included:
            continue

        diff__clazz__start_bin__end_bin = get__clazz__diff_bins(diff_bin_s, curr_diff)
        dist__clazz__start_bin__end_bin = get__clazz__dist_bins(dist_bin_s, curr_dist)
        curr_diff_cl = diff__clazz__start_bin__end_bin[0]
        curr_dist_cl = dist__clazz__start_bin__end_bin[0]

        if [curr_diff_cl, curr_dist_cl] not in zero__diff_dist_cl_s and _is_included(row):
            continue

        if curr_signal == SIGNAL_SHORT_IN:
            df.loc[curr_idx, _included] = False

        if curr_signal == SIGNAL_LONG_IN:
            df.loc[curr_idx, _included] = False

        exception_idx_s.append(curr_idx)

    ignores_count = calc_count__ignores__long_out__short_out(df, threshold)
    entries_count = len(df) - ignores_count
    if print_out:
        print_populated_char_n_times("-", 50)
        print(f"IGNORES: {format_num(ignores_count)}")
        print(f"ENTRIES: {format_num(entries_count)}")
        sys.stdout.flush()

    return df


def retrieve__produce_diff_dist_values__unwrap(param_s):
    try:
        symbol = param_s['symbol']
        discretization = param_s['discretization']
        threshold = param_s['threshold']
        start_dt = param_s['start_dt'] if 'start_dt' in param_s else None
        end_dt = param_s['end_dt'] if 'end_dt' in param_s else None

        symbol_discr__df = retrieve_all(symbol, [discretization], start_dt, end_dt, print_out=False)[0]
        symbol_discr__diff_dist__df = produce__diff_dist__values(symbol_discr__df, threshold)

        ignore_count = len(symbol_discr__df[symbol_discr__df[__SIGNAL(threshold)] == SIGNAL_IGNORE])
        signal_count = len(symbol_discr__df[symbol_discr__df[__SIGNAL(threshold)] != SIGNAL_IGNORE])
        ignore_signal_count_ratio = ignore_count / signal_count
        if ignore_signal_count_ratio > 10.5:
            print(f"{symbol} || ignore_count: {ignore_count} | signal_count: {signal_count} | ignore_signal_count_ratio: {ignore_signal_count_ratio}")
            sys.stdout.flush()

        return symbol_discr__diff_dist__df
    except Exception as ex:
        printmd(f"**ERROR PRODUCE DIFF & DIST:** {symbol} | {discretization} | {threshold} || {str(ex)}")

        return None


def produce_next_grad_n_df__unwrap(pair):
    symbol = pair['symbol']
    discretization = pair['discretization']
    close_grad_order = pair['close_grad_order']

    df = retrieve(symbol, discretization)
    df[_CLOSE_GRAD_NEXT(close_grad_order)] = df[_CLOSE_GRAD(close_grad_order)].shift(-close_grad_order)
    df = df[[_SYMBOL, _DISCRETIZATION, _UTC_TIMESTAMP, _CLOSE_GRAD(close_grad_order), _CLOSE_GRAD_NEXT(close_grad_order)]]

    return df


def retrieve__produce_differential__unwrap(param_s):
    try:
        symbol = param_s['symbol']
        discretization = param_s['discretization']
        threshold = param_s['threshold']
        start_dt = param_s['start_dt'] if 'start_dt' in param_s else None
        end_dt = param_s['end_dt'] if 'end_dt' in param_s else None

        symbol_discr__df = retrieve_all(symbol, [discretization], start_dt, end_dt, print_out=False)[0]

        zigzagized_title = ""
        if __SIGNAL(threshold) not in symbol_discr__df:
            symbol_discr__df = zigzagize_all(symbol, [symbol_discr__df], [threshold], print_out=False)[0]
            zigzagized_title = f"ZIGZAGIZED: {threshold}"

        differentified_title = ""
        if __TPR(threshold) not in symbol_discr__df:
            symbol_discr__df = produce__differential(symbol_discr__df, threshold)
            differentified_title = f"DIFFERENTIFIED: {threshold}"

        symbol_discr__df.meta_data = [symbol, discretization, zigzagized_title, differentified_title]

        required_col_s = [col for col in symbol_discr__df.columns if str(threshold) in col]
        df = symbol_discr__df[[*[_SYMBOL, _DISCRETIZATION, _KIEV_TIMESTAMP, _UTC_TIMESTAMP, 'open', 'high', 'low', 'close'], *required_col_s]]

        return df
    except Exception as ex:
        printmd(f"**ERROR PRODUCE DIFFERENTIALS:** {symbol} | {discretization} | {threshold} || {str(ex)}")

        return None


def calc_clustering_polygon(radius, offset):
    figure_x_s = []
    figure_y_s = []

    higher_x_s, higher_y_s = calc_circle_segment(90, 180, radius)
    higher_x_s += radius + offset
    higher_y_s += offset
    figure_x_s.extend(higher_x_s)
    figure_y_s.extend(higher_y_s)

    lower_x_s, lower_y_s = calc_circle_segment(270, 360, radius)
    lower_x_s += offset
    lower_y_s += radius + offset
    figure_x_s.extend(lower_x_s)
    figure_y_s.extend(lower_y_s)

    figure_x_s.append(figure_x_s[0])
    figure_y_s.append(figure_y_s[0])

    polygon_points_np = np.array(list(zip(figure_x_s, figure_y_s)))
    clustering_polygon = Polygon(polygon_points_np)

    return clustering_polygon


def calculate_diff_clustering(df, clustering_polygon: Polygon, threshold):
    _included = __INCLUDED(threshold)
    _signal = __SIGNAL(threshold)
    _diff = __DIFF(threshold)
    _diff_cl = __DIFF_CL(threshold)

    _curr_abs_diff = f'curr_abs_{__DIFF(threshold)}'
    _prev_abs_diff = f'prev_abs_{__DIFF(threshold)}'

    diff_distribution_df = df[((df[_signal] == SIGNAL_LONG_IN) | (df[_signal] == SIGNAL_SHORT_IN))][[_SYMBOL, _UTC_TIMESTAMP, _diff]]
    diff_distribution_df[_curr_abs_diff] = diff_distribution_df[_diff].abs()
    diff_distribution_df[_prev_abs_diff] = diff_distribution_df[_curr_abs_diff].shift(1)
    diff_distribution_df[_included] = diff_distribution_df[[_curr_abs_diff, _prev_abs_diff]].apply(lambda row: clustering_polygon.contains(Point(row[_curr_abs_diff], row[_prev_abs_diff])), axis=1)
    df[_curr_abs_diff] = diff_distribution_df[_curr_abs_diff]
    df[_prev_abs_diff] = diff_distribution_df[_prev_abs_diff]
    df[_included] = diff_distribution_df[_included]

    return df, diff_distribution_df.iloc[1:]


def override_cache_start_date_constraint(train_test_symbol__date_constraint_s, symbol, default_start_dt):
    symbol_date = [symbol_date for symbol_date in train_test_symbol__date_constraint_s if symbol_date['symbol'] == symbol][0]
    symbol_start_dt = parser.parse(symbol_date['date_constraints'][0])
    if symbol_start_dt > default_start_dt:
        return symbol_start_dt

    return default_start_dt


def presave_unwrap(param):
    df_excluded = param['df']
    diff_dist_data = param['diff_dist_data']
    threshold = param['threshold']
    unfinished_symbol_s = param['unfinished_symbol_s']

    _diff = __DIFF(threshold)
    _dist = __DIST(threshold)

    diff_cl_s, dist_cl_s = diff_dist_data['diff']['classes'], diff_dist_data['dist']['classes']
    diff_bin_s, dist_bin_s = diff_dist_data['diff']['bins'], diff_dist_data['dist']['bins']

    def calc__dd_cl__s(row):
        diff_bin = np.nan if np.isnan(row[_diff]) else row[_diff]
        dist_bin = np.nan if np.isnan(row[_dist]) else row[_dist]
        is_included = False if np.isnan(row[__INCLUDED(threshold)]) else row[__INCLUDED(threshold)]
        dd_cl = diff_dist_bin__dd_cl(diff_cl_s, dist_cl_s, diff_bin_s, dist_bin_s, hashablelist([diff_bin, dist_bin]), is_included)

        return dd_cl

    symbol = df_excluded.iloc[0][_SYMBOL]
    discretization = df_excluded.iloc[0][_DISCRETIZATION]
    try:
        df_excluded[__DD_CL(threshold)] = df_excluded[[_diff, _dist, __INCLUDED(threshold)]].apply(calc__dd_cl__s, axis=1).astype('Int16')
        df_fresh = retrieve(symbol, discretization)
        threshold_col_s = [col for col in df_excluded.columns if f'{threshold}' in col]
        df_fresh[threshold_col_s] = df_excluded[threshold_col_s]
        write_cache_df([df_fresh], print_out=False)
    except Exception as ex:
        print(ex)
        sys.stdout.flush()

        unfinished_symbol_s.append(symbol)

        return None

    return df_excluded


def cache__save_weights__diff_dist_df_s(df_excluded_s, diff_dist_data__excluded, threshold, num_workers=1):
    param_s = [{"df": df_excluded, "diff_dist_data": diff_dist_data__excluded, "threshold": threshold} for df_excluded in df_excluded_s]
    manager = multiprocessing.Manager()
    unfinished_symbol_s = manager.list()
    for param in param_s:
        param['unfinished_symbol_s'] = unfinished_symbol_s

    df_excluded_s = func_multi_process(presave_unwrap, param_s, num_workers=num_workers, print_result_full=False, finished_title="CACHE DF`s & SAVE WEIGHTS | DIFF DIST DATA")
    df_excluded_s = [df_excluded for df_excluded in df_excluded_s if df_excluded is not None]

    printmd(f"**FINISHED ({len(df_excluded_s)}) | UNFINISHED ({len(unfinished_symbol_s)}) SYMBOLS [CLAZZIFY | CACHE]:** \r\n{unfinished_symbol_s}")

    df_excluded_concat = pd.concat(df_excluded_s)
    save__weights__diff_dist_data(df_excluded_concat, diff_dist_data__excluded, threshold)


############## END ############## DIFF DIST CLASSES-BINS-LABELS-OHs ############### END #############


def validate_group_length(group, segments_count):
    input_df_lenght_s = [len(df) for df in group[:-1]]
    assert (all([input_df_lenght >= segments_count for input_df_lenght in input_df_lenght_s])), f"{group[-1].iloc[0][_SYMBOL]} | {input_df_lenght_s}"


def calculate_backprop_metrics_chunk(arg_s):
    chunk_i = arg_s['chunk_i']
    act_oh_s = arg_s['act_oh_chunk_s']
    pred_oh_prob_s = arg_s['pred_oh_prob_chunk_s']
    net_cpu_empty = arg_s['net_cpu_empty']

    act_clazz_s = net_cpu_empty.one_hot__clazz(act_oh_s)
    pred_oh_s = net_cpu_empty.prob__one_hot(pred_oh_prob_s)
    pred_clazz_s = net_cpu_empty.one_hot__clazz(pred_oh_s)

    return {'chunk_i': chunk_i, 'act_clazz_chunk_s': act_clazz_s, 'pred_clazz_s': pred_clazz_s, 'pred_oh_chunk_s': pred_oh_s}


def calculate_backprop_metircs_deprecated(signal_encoder, act_oh_s, pred_oh_prob_s):
    act_clazz_s = signal_encoder.one_hot__clazz(act_oh_s)
    act_label_s = signal_encoder.clazz__label(act_clazz_s)

    pred_oh_s = signal_encoder.prob__one_hot(pred_oh_prob_s)
    pred_clazz_s = signal_encoder.one_hot__clazz(pred_oh_s)
    pred_label_s = signal_encoder.one_hot__label(pred_oh_s)

    return act_clazz_s, act_label_s, pred_oh_s, pred_clazz_s, pred_label_s


def calculate_backprop_metircs_concurrent(net_cpu_empty, act_oh_s, pred_oh_prob_s, cpu_count):
    split_chunk_size = int(len(act_oh_s) / (cpu_count - 1)) if cpu_count > 1 else len(act_oh_s)
    act_oh_chunk_s = split_list_into_chunks(act_oh_s, split_chunk_size)
    pred_oh_prob_chunk_s = split_list_into_chunks(pred_oh_prob_s, split_chunk_size)

    arg_s = [{
        'chunk_i': i,

        'discretization': net_cpu_empty.get_out_discretization(),
        'threshold': net_cpu_empty.get_threshold(),
        'clazzes': net_cpu_empty.clazzes_count(),
        'net_cpu_empty': net_cpu_empty,

        'act_oh_chunk_s': act_oh_chunk_s[i],
        'pred_oh_prob_chunk_s': pred_oh_prob_chunk_s[i]
    } for i in range(len(act_oh_chunk_s))]

    if cpu_count > 1:
        with multiprocessing.Pool(cpu_count) as pool:
            result_chunk_s_sorted = [result for result in pool.map(calculate_backprop_metrics_chunk, arg_s)]
    else:
        result_chunk_s_sorted = [calculate_backprop_metrics_chunk(arg) for arg in arg_s]

    act_clazz_chunk_s = [chunk['act_clazz_chunk_s'] for chunk in result_chunk_s_sorted]
    pred_oh_chunk_s = [chunk['pred_oh_chunk_s'] for chunk in result_chunk_s_sorted]
    pred_clazz_chunk_s = [chunk['pred_clazz_s'] for chunk in result_chunk_s_sorted]

    act_clazz_s = np.array(list(itertools.chain(*act_clazz_chunk_s)), dtype=np.int8)
    pred_oh_s = np.array(list(itertools.chain(*pred_oh_chunk_s)), dtype=np.int8)
    pred_clazz_s = np.array(list(itertools.chain(*pred_clazz_chunk_s)), dtype=np.int8)

    return act_clazz_s, pred_oh_s, pred_clazz_s


def apply_row(row, signal_encoder):
    act_clazz_s = signal_encoder.one_hot__clazz(row['act_oh_s'])[0]
    act_label_s = signal_encoder.clazz__label(act_clazz_s)[0]

    pred_oh_s = signal_encoder.prob__one_hot(row['pred_oh_prob_s'])[0]
    pred_label_s = signal_encoder.one_hot__label(pred_oh_s)[0]

    return pd.Series({
        'act_clazz_s': act_clazz_s,
        'act_label_s': act_label_s,
        'pred_oh_s': pred_oh_s,
        'pred_label_s': pred_label_s,
    })


def calculate_backprop_metircs_pandas_parralel(signal_encoder, act_oh_s, pred_oh_prob_s, cpu_count):
    from parallel_pandas import ParallelPandas

    ParallelPandas.initialize(n_cpu=cpu_count, split_factor=1, disable_pr_bar=True)
    df = pd.DataFrame({'act_oh_s': act_oh_s.tolist(), 'pred_oh_prob_s': pred_oh_prob_s.tolist()})
    result_df = df[['act_oh_s', 'pred_oh_prob_s']].p_apply(lambda row: apply_row(row, signal_encoder), axis=1)

    print(f"df.size: {format_df_memory(df)}")
    print(f"result_df.size: {format_df_memory(result_df)}")

    act_clazz_s = result_df['act_clazz_s'].to_list()
    act_label_s = result_df['act_label_s'].to_list()

    pred_oh_s = result_df['pred_oh_s'].to_list()
    pred_label_s = result_df['pred_label_s'].to_list()

    return act_clazz_s, act_label_s, pred_oh_s, pred_label_s


def produce_signal_encoder():
    from SRC.LIBRARIES.SignalEncoderStage4 import SignalEncoder
    signal_encoder = SignalEncoder()

    return signal_encoder


def produce_dummy_dataset_factory(model_suffix=None):
    from SRC.LIBRARIES.DummyDatasetStage4 import DummyDataset

    return lambda net_cpu_empty, samples_count: DummyDataset(net_cpu_empty, samples_count)


def produce_dummy_net_cpu(net, _produce_dummy_dataset):
    import torch

    dummy_input, dummy_output = next(enumerate(_produce_dummy_dataset(net, 1)))[1]
    if isinstance(dummy_input, tuple):
        dummy_input_s_0 = dummy_input[0].unsqueeze(0).float()
        dummy_input_s_1 = dummy_input[1].unsqueeze(0).float()

        if torch.cuda.is_available() and USE_GPU():
            net = net.cuda()
            dummy_input_s_0 = dummy_input_s_0.cuda()
            dummy_input_s_1 = dummy_input_s_1.cuda()

        net(dummy_input_s_0, dummy_input_s_1)
    else:
        dummy_input_s = dummy_input.unsqueeze(0).float()

        if torch.cuda.is_available() and USE_GPU():
            net = net.cuda()
            dummy_input_s = dummy_input_s.cuda()

        net(dummy_input_s)

    from SRC.NN.BaseContinousNN import BaseContinousNN
    if not check_env_true(_FINE_TUNE_NET) and not isinstance(net, BaseContinousNN):
        net.initialize_weights()

    return net


def parse_model_name(name):
    # Extract numbers after 'x' and 'EP'
    try:
        x_val = int(name.split('x')[1].split('-')[0])
    except:
        x_val = -1
    try:
        ep_val = int(name.split('-')[-1].replace("EP", ""))
    except:
        ep_val = -1
    return (x_val, ep_val)


def get_model_name_s(include_model_name_s=[], exclude_model_name_s=[]):
    folder_path = f"{MODEL_FOLDER_PATH()}"
    os.makedirs(folder_path, exist_ok=True)

    model_name_s = [m for m in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, m)) and m.endswith(".pt")]
    if len(include_model_name_s) > 0:
        model_name_s = [model_name for model_name in model_name_s if model_name.replace(".pt", "") in include_model_name_s]

    if len(exclude_model_name_s) > 0:
        model_name_s = [model_name for model_name in model_name_s if model_name.replace(".pt", "") not in exclude_model_name_s]

    model_present_s = [model_name.replace(".pt", "") for model_name in model_name_s]

    # Sort by x first (ascending), then by EP (descending)
    sorted_model_present_s = sorted(model_present_s, key=lambda n: (parse_model_name(n)[0], parse_model_name(n)[1]), reverse=False)

    # If you want EP descending within each x group:
    # sorted_model_present_s = sorted(model_present_s, key=lambda n: (parse_model_name(n)[0], -parse_model_name(n)[1]))

    return sorted_model_present_s


def TEST__get_model_name_s():
    include_model_name_s = []
    exclude_model_name_s = ['C3_x3-DUMMY_stage4-EP38', 'C3_x3-DUMMY_stage4-EP40']

    model_name_s = get_model_name_s(include_model_name_s, exclude_model_name_s)

    print(model_name_s)


def normalize_weigths(weights):
    return np.array([x / sum(weights) for x in list(weights)])


def assert__data_config__clazz_map__equal(df_concat, threshold):
    discretization = df_concat.iloc[0][_DISCRETIZATION]
    weight_diff_dist_data_map = produce__weight_diff_dist_data_map(discretization, threshold)
    config_label_s = weight_diff_dist_data_map['labels']
    config_clazzes_s = list(range(len(config_label_s)))
    config_clazzes_count = len(config_clazzes_s)
    data_clazz_s = list(sorted(df_concat[__DD_CL(threshold)].unique()))
    data_clazzes_count = len(data_clazz_s)

    assert_error_title = f"Clazzes count: {config_clazzes_count} [CONFIG] != {data_clazzes_count} [DATA]"
    assert_error_details = f"CONFIG LABELS: {config_label_s}\r\nCONFIG CLAZZES: {config_clazzes_s}\r\nDATA CLAZZES: {data_clazz_s}"
    assert config_clazzes_count == data_clazzes_count, f"{assert_error_title}\r\n\r\n{assert_error_details}"


def assert__input_features_exists(df_s, discretization_feature_s):
    symbol = df_s[0].iloc[0][_SYMBOL]
    valid_s = []
    for discretization, feature_s in discretization_feature_s.items():
        df = [df for df in df_s if df.iloc[0][_DISCRETIZATION] == discretization][0]
        valid = all(feature in df.columns for feature in feature_s)
        valid_s.append(valid)
        
    assert all(valid_s), f"FAILED INPUT FEATURES VALIDATION: {symbol} | {discretization_feature_s}"
        

def produce__weight_diff_dist_data_map(discretization, threshold):
    try:
        discretization_feature_weight_diff_dist_data_map = read_json(WEIGHTS_DIFF_DIST_DATA_MAP_FILE_PATH())
        weight_diff_dist_data_map = discretization_feature_weight_diff_dist_data_map[discretization][__DD_CL(threshold)]

        return weight_diff_dist_data_map
    except KeyError as err:
        print_populated_char_n_times('#', 30, f"THERE IS NO MAP: {discretization} | {threshold}")
        traceback.print_exc()

        raise


def cache__exchange_info(configs_suffix):
    from SRC.LIBRARIES.binance_helpers import produce_binance_client_singleton

    unset_parent_process = produce_parent_process_delegate()

    configs = read_json(CONFIGS_FILE_PATH(suffix=configs_suffix))
    all_symbol_s = set([conf['symbol'] for conf in configs['train']])

    client = produce_binance_client_singleton()

    exchange_info_spot = client.get_exchange_info()
    write_json(exchange_info_spot, EXCHANGE_INFO_FILE_PATH())

    exchange_info_futures = client.futures_exchange_info()
    write_json(exchange_info_futures, EXCHANGE_INFO_FUTURES_FILE_PATH())

    exchange_info_futures_coin = client.futures_coin_exchange_info()
    write_json(exchange_info_futures_coin, EXCHANGE_INFO_FUTURES_COIN_FILE_PATH())

    global_trading_pairs = []
    for exchange_info in [exchange_info_spot, exchange_info_futures]:
        trading_pairs = [symbol['symbol'] for symbol in exchange_info['symbols'] if symbol['status'] == 'TRADING' and 'USDT' in symbol['symbol']]
        global_trading_pairs.extend(trading_pairs)

    global_trading_pairs = set(global_trading_pairs)
    printmd(f"**EXCHANGE INFO CACHED**\r\n\r\n***SPOT & FUTURES*** | len = {len(global_trading_pairs)}")
    print(global_trading_pairs)

    filtered_trading_pairs = [trading_pair for trading_pair in global_trading_pairs if trading_pair in all_symbol_s]
    printmd(f"PROCESSED SYMBOLS | len = {len(filtered_trading_pairs)}")
    print(filtered_trading_pairs)

    filtered_trading_pairs = [trading_pair for trading_pair in global_trading_pairs if trading_pair not in all_symbol_s and trading_pair[-4:] == 'USDT']
    printmd(f"NOT PROCESSED SYMBOLS | len = {len(filtered_trading_pairs)}")
    print(filtered_trading_pairs)

    unset_parent_process()


def calc_distribution_histogram(data, symmetric_space):
    abs_max = abs(max(min(data), max(data), key=abs))
    counts, bins = np.histogram(data, bins=symmetric_space, range=(-abs_max, abs_max))
    weights = [0 if count == 0 else 1 / count for count in counts]
    weights_norm = [weight / sum(weights) for weight in weights]

    # print(f"BINS: {bins} \r\n COUNTS: {counts} \r\n WEIGHTS: {weights} \r\n Weights NORM: {weights_norm}")

    return bins, counts, weights, weights_norm


def clean_df_threshold__unwrap(arg):
    try:
        symbol = arg['symbol']
        discretization = arg['discretization']
        threshold_s = arg['threshold_s']

        df = retrieve(symbol, discretization)
        for threshold in threshold_s:
            df = df[[col for col in df.columns if str(threshold) not in col]]

        write_cache_df([df], print_out=False)

        return df
    except:
        print(f"CLEAN THRESHOLD ERROR: {symbol} | {discretization} | {threshold_s}")
        traceback.print_exc()
        return None


def parse_differentify_notebook_setup(notebook_name):
    match = re.search(r'_(\d+[A-Z])_(\d+)', notebook_name)
    if match:
        discretization = match.group(1)
        raw_number = match.group(2)
        threshold = float(f"0.{raw_number}")
        return discretization, threshold
    return None, None


def clean_dfs_discretization_thresholds(discretization_s, threshold_s, handle_symbols=[], num_workers=1):
    if len(threshold_s) == 0 or len(discretization_s) == 0:
        print(f"NO DISCRETIZATIONS OR THRESHOLDS FOR CLEAN")

        return

    data_folder_path = DATA_FOLDER_PATH()
    symbol_s = [dir for dir in os.listdir(data_folder_path) if '___' not in dir and 'USDT' in dir]
    if len(handle_symbols) > 0:
        symbol_s = [symbol for symbol in symbol_s if symbol in handle_symbols]

    for discretization in discretization_s:
        arg_s = [{'symbol': symbol, 'discretization': discretization, 'threshold_s': threshold_s} for symbol in symbol_s]
        func_multi_process(clean_df_threshold__unwrap, arg_s, num_workers=num_workers, finished_title=f'CLEAN THRESHOLD: {threshold_s}', print_result_full=is_running_under_pycharm())


def candelify(df, symbol, discretization, shift_idx=False, shift_open_close=False, target_feature='price'):
    interval_partition = INTERVAL_PARTITION(discretization)

    df_ohlc = df[target_feature].resample(interval_partition).ohlc()
    if shift_idx:
        df_ohlc.index = df_ohlc.index + TIME_DELTA(discretization)

    if shift_open_close:
        df_ohlc['open'] = df_ohlc['close'].shift(1).combine_first(df_ohlc['open'])

    df_ohlc[_SYMBOL] = symbol
    df_ohlc[_DISCRETIZATION] = discretization
    df_ohlc[_KIEV_TIMESTAMP] = df_ohlc.apply(lambda row: as_kiev_tz(row.name), axis=1)
    df_ohlc[_UTC_TIMESTAMP] = df_ohlc.apply(lambda row: as_utc_tz(row.name), axis=1)

    df_ohlc = order_main_cols_df(df_ohlc)

    return df_ohlc


def pdf_featurize_timestamp_s(pdf, rule_s):
    for rule in rule_s:
        target_col = rule[0]
        result_col = rule[1]
        func = rule[2]

        if target_col == 'index':
            pdf[result_col] = pdf.index.map(lambda target_col: func(target_col))
            # pdf[result_col] = list(map(lambda target_col: func(target_col), pdf.index.to_list()))
        else:
            pdf[result_col] = pdf.apply(lambda row: func(row[target_col]), axis=1)

    return pdf


def calc_geometric_mean(metric_value_s):
    geometric_mean = math.prod(metric_value_s) ** (1 / len(metric_value_s))

    return geometric_mean


def TEST__COMPARE__GROUPING_DF_LIST():
    grouping_key_s = ['key1', 'key2', 'key_12']

    # ------------------------------------------

    rows_count = 100
    data_s = [
        {
            "key1": random.uniform(30, 50),
            "key2": random.uniform(30, 50),
            "key3": random.uniform(30, 50),
            "key4": random.uniform(30, 50),
            "key5": random.uniform(30, 50),

        } for i in range(rows_count)
    ]

    data_df = pd.DataFrame(data_s)

    # ------------------------------------------

    def calc_dict(item):
        return {
            "key_12": item["key1"] + item["key2"],
            "key_45": item["key4"] - item["key5"]
        }

    def calc_df(row):
        return pd.Series(calc_dict(row))

    # ------------------------------------------

    orig_data_s = data_s
    data_s = []
    for data in orig_data_s:
        data_s.append(merge_dicts(data, calc_dict(data)))

    data_df[["key_12", "key_45"]] = data_df.apply(calc_df, axis=1)

    # ------------------------------------------

    for data in data_s:
        data['group_key'] = "|".join([str(data[key]) for key in grouping_key_s])

    grouped_data_s = defaultdict(list)
    for data in data_s:
        grouped_data_s[data['group_key']].append(data)

    grouped_data_df = data_df.groupby(grouping_key_s)

    # ------------------------------------------

    res_data = []
    for group_key, groupe_data_s in grouped_data_s.items():
        key_12_3 = [data['key_12'] for data in groupe_data_s]

        geometric_mean = calc_geometric_mean(key_12_3)
        res_data.append(geometric_mean)
        print(f"{group_key}: {geometric_mean}")

    print(f"RESULT DATA SUM: {sum(res_data)}")

    print(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    res_df = []
    for (key1, key2, key_12), g in grouped_data_df:
        key_12_3 = g['key_12'].to_list()

        geometric_mean = calc_geometric_mean(key_12_3)
        res_df.append(geometric_mean)
        print(f"{(key1, key2, key_12)}: {geometric_mean}")

    print(f"RESULT DF SUM: {sum(res_df)}")


def TEST__DIFF_DIST_BIN_CL_OH():
    threshold = 0.01
    out_discretization = '30M'
    out_feature = f'dd_cl_{threshold}'

    discretization_feature_weight_diff_dist_data_map = read_json(WEIGHTS_DIFF_DIST_DATA_MAP_FILE_PATH())
    weight_diff_dist_data_map = discretization_feature_weight_diff_dist_data_map[out_discretization][out_feature]
    diff_dist_data = hashabledict(weight_diff_dist_data_map['diff_dist_data'])

    diff_cl_s, dist_cl_s = diff_dist_data['diff']['classes'], diff_dist_data['dist']['classes']
    diff_bin_s, dist_bin_s = diff_dist_data['diff']['bins'], diff_dist_data['dist']['bins']

    clazz_1 = diff_dist_bin__dd_cl(diff_cl_s, dist_cl_s, diff_bin_s, dist_bin_s, hashablelist([-0.045, 4]), True)
    clazz_2 = diff_dist_bin__dd_cl(diff_cl_s, dist_cl_s, diff_bin_s, dist_bin_s, hashablelist([0.0176, 17]), False)
    clazz_3 = diff_dist_bin__dd_cl(diff_cl_s, dist_cl_s, diff_bin_s, dist_bin_s, hashablelist([0.0176, 17]), True)
    clazz_4 = diff_dist_bin__dd_cl(diff_cl_s, dist_cl_s, diff_bin_s, dist_bin_s, hashablelist([np.nan, np.nan]), np.nan)
    clazz_5 = diff_dist_bin__dd_cl(diff_cl_s, dist_cl_s, diff_bin_s, dist_bin_s, hashablelist([0.05, 18]), True)

    clazz_s = [clazz_1, clazz_2, clazz_3, clazz_4, clazz_5]
    for clazz in clazz_s:
        print(clazz)
        diff_dist_cl = dd_cl__diff_dist_cl(diff_cl_s, dist_cl_s, clazz)
        print(diff_dist_cl)
        dd_cl = diff_dist_cl__dd_cl(diff_cl_s, dist_cl_s, diff_dist_cl)
        print(dd_cl)
        oh = diff_dist_cl__oh(diff_cl_s, dist_cl_s, diff_dist_cl)
        print(oh)
        diff_dist_cl_2 = oh__diff_dist_cl(diff_cl_s, dist_cl_s, oh)
        print(diff_dist_cl_2)
        dd_cl_2 = oh__dd_cl(diff_cl_s, dist_cl_s, oh)
        print(dd_cl_2)
        oh_2 = dd_cl__oh(diff_cl_s, dist_cl_s, dd_cl)
        print(oh_2)
        print_populated_char_n_times('-', 50)


if __name__ == "__main__":
    TEST__COMPARE__GROUPING_DF_LIST()
    TEST__DIFF_DIST_BIN_CL_OH()
    TEST__get_model_name_s()