import distutils
import functools
import hashlib
import importlib
import inspect
import itertools
import json
import math
import multiprocessing
import os
import random
import shutil
import socket
import subprocess
import sys
import threading
import ta
import time
import traceback
import uuid
from asyncio import Queue
from collections import OrderedDict, deque
from datetime import datetime, timedelta
from functools import lru_cache
from functools import wraps
from os.path import getmtime
from pathlib import Path

import binance.exceptions
import diskcache as dc
import numpy as np
import pandas as pd
import requests
from IPython.core.display import Markdown
from IPython.core.display_functions import DisplayHandle, display
from IPython.display import HTML
from filelock import FileLock
from filelock import Timeout
from requests import ReadTimeout
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

from SRC.CORE.MPDeque import MPDeque
from SRC.CORE._CONSTANTS import CPU_COUNT, MODEL_FOLDER_PATH, _TRADES_FILE_PATH, _TRADES_RUNNER_FILE_PATH, project_root_dir, _TOP_UP_NET_FOLDER_PATH, OUT_SEGMENT, _DASHBOARD_SEGMENT, KIEV_TZ, _BALANCE_FILE_PATH, NO_TRADES_TIMEOUT, \
    OVER_TIMEOUT, _CONFIGS_SUFFIX, USE_GPU, _SYMBOL, _DISCRETIZATION, _MODEL_SUFFIX, _RESOURCES_FORMAT_JUPYTER, _DASHBOARD_SEGMENT_NET_FULL_PATH, _BACKTESTING, _AUTOTRADING
from SRC.CORE._CONSTANTS import _CANDLE_FILE_PATH, _DATETIME_PRICE_FILE_PATH, _DATETIME_PRICES_FILE_PATH
from SRC.CORE.debug_utils import get_processing_device, produce_measure, is_cloud, format_memory, get_current_notebook_name, is_running_in_notebook, DEBUG, DEBUG_SPLITTED, ERROR_SPLITTED, EXCEPTION_SPLITTED, produce_formatters, \
    is_running_under_pycharm_debugger, is_running_under_pycharm, log_context, CONSOLE, NOTICE, IS_DEBUG, CONSOLE_SPLITTED
from SRC.CORE.debug_utils import printmd
from SRC.CORE.server_setup import parse_port_mapping
from SRC.CORE.utils import datetime_h_m_s__d_m_Y, write_json, datetime_h_m_s, datetime_Y_m_d__h_m_s, read_json_safe_retry
from SRC.CORE.utils import read_json_safe
from SRC.LIBRARIES.concurrent_utils import iterate_multiprocess_pool, iterate_multiprocess_executor, iterate_multithread_executor
from SRC.LIBRARIES.time_utils import TIME_DELTA, kiev_now, kiev_now_formatted


try:
    from backports.zoneinfo import ZoneInfo
except:
    from zoneinfo import ZoneInfo



def set_without_reordering(input_list):
    ordered_dict = OrderedDict.fromkeys(input_list)
    result_set = list(ordered_dict.keys())

    return result_set

def mrc_supersmoother(src, length):
    """
    Supersmoother filter by John Ehlers
    """
    a1 = np.exp(-1.414 * np.pi / length)
    b1 = 2 * a1 * np.cos(1.414 * np.pi / length)
    c3 = -a1 * a1
    c2 = b1
    c1 = 1 - c2 - c3

    ss = np.zeros_like(src)
    ss[:2] = src[:2]

    for i in range(2, len(src)):
        ss[i] = c1 * src[i] + c2 * ss[i-1] + c3 * ss[i-2]

    return ss

def mrc_sak_filter(filter_type, src, length):
    """
    Ehlers Swiss Army Knife filters
    """
    pi = np.pi
    cycle = 2 * pi / length

    c0, c1 = 1.0, 0.0
    b0, b1, b2 = 1.0, 0.0, 0.0
    a1, a2 = 0.0, 0.0
    alpha, beta, gamma = 0.0, 0.0, 0.0

    if filter_type == "Ehlers EMA":
        alpha = (np.cos(cycle) + np.sin(cycle) - 1) / np.cos(cycle)
        b0 = alpha
        a1 = 1 - alpha

    elif filter_type == "Gaussian":
        beta = 2.415 * (1 - np.cos(cycle))
        alpha = -beta + np.sqrt(beta * beta + 2 * beta)
        c0 = alpha * alpha
        a1 = 2 * (1 - alpha)
        a2 = -(1 - alpha) * (1 - alpha)

    elif filter_type == "Butterworth":
        beta = 2.415 * (1 - np.cos(cycle))
        alpha = -beta + np.sqrt(beta * beta + 2 * beta)
        c0 = alpha * alpha / 4
        b1, b2 = 2, 1
        a1 = 2 * (1 - alpha)
        a2 = -(1 - alpha) * (1 - alpha)

    elif filter_type == "BandStop":
        beta = np.cos(cycle)
        gamma = 1 / np.cos(cycle * 2 * 0.1)  # delta = 0.1
        alpha = gamma - np.sqrt(gamma * gamma - 1)
        c0 = (1 + alpha) / 2
        b1 = -2 * beta
        b2 = 1
        a1 = beta * (1 + alpha)
        a2 = -alpha

    elif filter_type == "SMA":
        c1 = 1 / length
        b0 = 1 / length
        a1 = 1

    elif filter_type == "EMA":
        alpha = 2 / (length + 1)
        b0 = alpha
        a1 = 1 - alpha

    elif filter_type == "RMA":
        alpha = 1 / length
        b0 = alpha
        a1 = 1 - alpha

    output = np.zeros_like(src)
    output[:3] = src[:3]

    for i in range(3, len(src)):
        output[i] = (c0 * (b0 * src[i] +
                          b1 * (src[i-1] if i-1 >= 0 else 0) +
                          b2 * (src[i-2] if i-2 >= 0 else 0)) +
                    a1 * output[i-1] +
                    a2 * output[i-2] -
                    c1 * (src[i-length] if i-length >= 0 else 0))

    return output

def add_mrc_indicators(df, source_type='hlc3', filter_type='SuperSmoother',
                       length=200, innermult=1.0, outermult=2.415):
    """
    Добавляет в DataFrame колонки MRC (meanline, upband1, loband1, upband2, loband2).
    Работает без побочных эффектов и глобальных переменных.

    Parameters:
    df : pandas.DataFrame - должен содержать колонки 'open', 'high', 'low', 'close'
    source_type : 'hlc3', 'ohlc4', 'close'
    filter_type : тип фильтра (например, 'SuperSmoother')
    length : период (по умолчанию 200)
    innermult : множитель для внутреннего канала
    outermult : множитель для внешнего канала

    Returns:
    pandas.DataFrame с добавленными колонками:
        'meanline', 'meanrange', 'upband1', 'loband1', 'upband2', 'loband2'
    """
    import math
    import numpy as np

    df = df.copy()  # работаем с копией, чтобы не изменять оригинал

    # Вычисляем источник
    if source_type == 'hlc3':
        source = (df['high'] + df['low'] + df['close']) / 3
    elif source_type == 'ohlc4':
        source = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    else:
        source = df['close']

    source_vals = source.values

    # Истинный диапазон
    tr = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            (df['high'] - df['close'].shift(1)).abs(),
            (df['low'] - df['close'].shift(1)).abs()
        )
    ).fillna(0).values

    # Применяем фильтр
    if filter_type == 'SuperSmoother':
        meanline = mrc_supersmoother(source_vals, length)
        meanrange = mrc_supersmoother(tr, length)
    else:
        meanline = mrc_sak_filter(filter_type, source_vals, length)
        meanrange = mrc_sak_filter(filter_type, tr, length)

    pi = np.pi
    mult = pi * innermult
    mult2 = pi * outermult

    df['meanline'] = meanline
    df['meanrange'] = meanrange
    df['upband1'] = meanline + meanrange * mult
    df['loband1'] = meanline - meanrange * mult
    df['upband2'] = meanline + meanrange * mult2
    df['loband2'] = meanline - meanrange * mult2

    return df

def mrc_calculate(mrc_df, df, source_type='hlc3', filter_type='SuperSmoother', innermult=1.0, outermult=2.415):
    mrc_df = add_mrc_indicators(mrc_df, source_type, filter_type, 200, innermult, outermult)
    # Переносим только нужную часть в df
    for col in ['meanline', 'meanrange', 'upband1', 'loband1', 'upband2', 'loband2']:
        df[col] = mrc_df.loc[df.index, col]

    return mrc_df

# Функция для подготовки данных с буфером
def prepare_buffer_data(df_main, df_display, buffer_size):
    combined = pd.concat([df_main.iloc[-(buffer_size + len(df_display)):], df_display])
    combined = combined[~combined.index.duplicated(keep='last')].sort_index()
    return combined

def rsi(df, rsi_calc_df):
    df['rsi'] = ta.momentum.RSIIndicator(close=rsi_calc_df['close'], window=14).rsi().loc[df.index]

    return df

def stochastic_tradingview(df, stoch_calc_df, periodK=14, smoothK=3, periodD=3):
    lowest_low = stoch_calc_df['low'].rolling(window=periodK).min()
    highest_high = stoch_calc_df['high'].rolling(window=periodK).max()
    raw_k = 100 * (stoch_calc_df['close'] - lowest_low) / (highest_high - lowest_low)
    stoch_k = raw_k.rolling(window=smoothK).mean()
    stoch_d = stoch_k.rolling(window=periodD).mean()
    df['stoch_k'] = stoch_k.loc[df.index]
    df['stoch_d'] = stoch_d.loc[df.index]

    return df

def macd(df, macd_calc_df):
    macd = ta.trend.MACD(
        close=macd_calc_df['close'],
        window_slow=26,
        window_fast=12,
        window_sign=9
    )
    df['macd_line'] = macd.macd().loc[df.index]
    df['macd_signal'] = macd.macd_signal().loc[df.index]
    df['macd_histogram'] = macd.macd_diff().loc[df.index]

    return df, macd

def atr(df, atr_calc_df):
    atr_full = ta.volatility.AverageTrueRange(
        high=atr_calc_df['high'],
        low=atr_calc_df['low'],
        close=atr_calc_df['close'],
        window=14
    ).average_true_range()
    df['atr'] = atr_full.loc[df.index]

    return df

def plot_evaluated_model(train_config, plot_data_queue, initial_validation_data, train_validation_data, final_validation_data, train_loss_s, test_loss_s, train_validation_s, test_validation_s, epochs_count, epoch_exec_time_s, current_exec_time_seconds, started_at, is_finished=False):
    last_updated_at = kiev_now()
    plot_data = {
        'test_validation_data': dict(initial_validation_data) if initial_validation_data is not None else None,
        'train_validation_data': dict(train_validation_data) if train_validation_data is not None else None,
        'final_validation_data': dict(final_validation_data) if final_validation_data is not None else None,

        'train_loss_s': list(train_loss_s) if train_loss_s is not None else None,
        'test_loss_s': list(test_loss_s) if test_loss_s is not None else None,

        'train_validation_s': list(train_validation_s) if train_validation_s is not None else None,
        'test_validation_s': list(test_validation_s) if test_validation_s is not None else None,

        'epoch_exec_time_s': list(epoch_exec_time_s) if epoch_exec_time_s is not None else None,
        'current_exec_time_seconds': current_exec_time_seconds,
        'epochs_count': epochs_count,

        'started_at': started_at,
        'last_updated_at': last_updated_at,
        'is_finished': is_finished,

        'train_config': train_config
    }

    for plot_data_queue_ in plot_data_queue:
        plot_data_queue_.append(plot_data)

    if check_env_true('WRITE_PLOT_DATA_JSON', False):
        write_json(plot_data, f"{project_root_dir()}/plot_data.json")


def calculate_auc_roc_deprecated(actuals, predictions, class_count):
    actuals = np.array(actuals)
    predictions = np.array(predictions)
    fpr = {}
    tpr = {}
    thresh = {}
    roc_auc = {}

    if np.isnan(actuals).any():
        print(actuals)
        traceback.print_stack()
        raise RuntimeError

    if np.isnan(predictions).any():
        print(predictions)
        traceback.print_stack()
        raise RuntimeError

    for i in range(class_count):
        fpr[i], tpr[i], thresh[i] = roc_curve(actuals, predictions[:, i], pos_label=i)
        roc_auc[i] = auc(fpr[i], tpr[i])

    return fpr, tpr, roc_auc


def compute_roc_auc_unwrap(arg):
    clazz = arg['clazz']
    actuals = arg['actuals']
    predictions_clazz = arg['predictions_clazz']

    # fpr, tpr, thresh = roc_curve(actuals == clazz, predictions == clazz)
    # roc_auc = auc(fpr, tpr)

    fpr, tpr, thresh = roc_curve(actuals, predictions_clazz, pos_label=clazz)
    roc_auc = auc(fpr, tpr)

    return clazz, fpr, tpr, thresh, roc_auc


def calculate_auc_roc_concurrent(actuals, predictions, class_count, cpu_count):
    arg_s = [{'clazz': clazz, 'actuals': actuals, 'predictions_clazz': predictions[:, clazz]} for clazz in range(class_count)]
    if cpu_count > 1:
        with multiprocessing.Pool(cpu_count) as pool:
            results = pool.map(compute_roc_auc_unwrap, arg_s)
    else:
        results = [compute_roc_auc_unwrap(arg) for arg in arg_s]

    fpr = {}
    tpr = {}
    thresh = {}
    roc_auc = {}

    for clazz, fpr_i, tpr_i, thresholds_i, roc_auc_i in results:
        clazz_str = str(clazz)
        fpr[clazz_str] = fpr_i
        tpr[clazz_str] = tpr_i
        thresh[clazz_str] = thresholds_i
        roc_auc[clazz_str] = roc_auc_i

    return {'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc}


def calculate_metrics_deprecated(signal_encoder, act_oh_s, pred_oh_prob_s, label_set, norm_conf_mtx=None):
    from SRC.LIBRARIES.new_data_utils import calculate_backprop_metircs_deprecated

    title = f"---------DEPRECATED APPROACH---------"
    printmd(title)
    print(format_memory(calc_memory_size(pred_oh_prob_s) + calc_memory_size(act_oh_s)))

    epoch_post_measure = produce_measure(title)

    local_measure = produce_measure('calculate_backprop_metircs_deprecated')
    act_clazz_s, act_label_s, pred_oh_s, pred_clazz_s, pred_label_s = calculate_backprop_metircs_deprecated(signal_encoder, act_oh_s, pred_oh_prob_s)
    local_measure(print_out=True)

    local_measure = produce_measure('calculate_auc_roc_deprecated')
    auc_roc = calculate_auc_roc_deprecated(act_clazz_s, pred_oh_prob_s, signal_encoder.clazzes_count())
    local_measure(print_out=True)

    local_measure = produce_measure('confusion_matrix')
    conf_mtx = confusion_matrix(act_label_s, pred_label_s, labels=label_set, normalize=norm_conf_mtx)
    local_measure(print_out=True)

    # print(f"act_oh_s: \r\n{act_oh_s[:1]}")
    # print(f"act_clazz_s: \r\n{act_clazz_s[:1]}")
    # print(f"pred_oh_prob_s: \r\n{pred_oh_prob_s[:1]}")
    # print(f"pred_oh_s: \r\n{pred_oh_s[:1]}")
    # print(f"pred_clazz_s: \r\n{pred_clazz_s[:1]}")

    epoch_post_measure(print_out=True)

    return auc_roc, conf_mtx


def calculate_discreate_metrics_optimized(net_cpu_empty, act_oh_a, pred_oh_prob_a, cpu_count, norm_conf_mtx=None):
    from SRC.LIBRARIES.new_data_utils import calculate_backprop_metircs_concurrent

    # title = f"---------OPTIMIZED APPROACH---------"
    # printmd(title)
    # print(format_memory(act_oh_a.nbytes + pred_oh_prob_a.nbytes))

    # epoch_post_measure = produce_measure(title)

    # local_measure = produce_measure('calculate_backprop_metircs_concurrent')
    act_clazz_s, pred_oh_s, pred_clazz_s = calculate_backprop_metircs_concurrent(net_cpu_empty, act_oh_a, pred_oh_prob_a, cpu_count)
    # local_measure(print_out=True)

    # local_measure = produce_measure('calculate_auc_roc_concurrent')
    auc_roc_map = calculate_auc_roc_concurrent(act_clazz_s, pred_oh_prob_a, net_cpu_empty.clazzes_count(), cpu_count)
    # local_measure(print_out=True)

    # local_measure = produce_measure('confusion_matrix')
    if norm_conf_mtx == 'custom':
        unnorm_conf_mtx = confusion_matrix(act_clazz_s, pred_clazz_s, labels=list(range(net_cpu_empty.clazzes_count())))
        conf_mtx = np.round(unnorm_conf_mtx / np.min(unnorm_conf_mtx[unnorm_conf_mtx != 0])).astype(int)
    else:
        conf_mtx = confusion_matrix(act_clazz_s, pred_clazz_s, labels=list(range(net_cpu_empty.clazzes_count())), normalize=norm_conf_mtx)
    # local_measure(print_out=True)

    # print(f"act_oh_s: \r\n{act_oh_s[:1]}")
    # print(f"act_clazz_s: \r\n{act_clazz_s[:1]}")
    # print(f"pred_oh_prob_s: \r\n{pred_oh_prob_s[:1]}")
    # print(f"pred_oh_s: \r\n{pred_oh_s[:1]}")
    # print(f"pred_clazz_s: \r\n{pred_clazz_s[:1]}")

    # epoch_post_measure(print_out=True)

    return auc_roc_map, conf_mtx


def upload_data_folders(symbol_s):
    import subprocess
    for symbol in symbol_s:
        subprocess.call([f'cd ../../integration && ./data_folder___upload__pod.sh {symbol}USDT'], shell=True)


def run_multi_process(_lambda, arg_s, is_parralel_execution=True, finished_title="", print_result_full=True):
    num_workers = CPU_COUNT() if is_parralel_execution else 1

    return func_multi_process(_lambda, arg_s, num_workers, finished_title, print_result_full=print_result_full)


def handle_dict(result):
    str_s = []
    for key, value in result.items():
        if isinstance(value, list):
            list_s = []
            for item in value:
                if isinstance(item, pd.DataFrame):
                    list_s.append(handle_df(item))
                else:
                    list_s.append(item)
            list_present = str(list_s)
            str_s.append(f"{key}: {list_present}")
            continue

        if isinstance(value, pd.DataFrame):
            value = handle_df(value)
            str_s.append(f"{key}: {value}")
            continue

        if isinstance(value, dict):
            value = handle_dict(value)
            str_s.append(f"{key}: {value}")
            continue

        str_s.append(f"{key}: {value}")
    str_present = ' | '.join(str_s)

    return str_present


def handle_df(result):
    try:
        str_present = f"{str(result.meta_data)}"
    except:
        cols_present_s = []
        if _SYMBOL in result.columns:
            cols_present_s.append(f"{result.iloc[0][_SYMBOL]}")

        if _DISCRETIZATION in result.columns:
            cols_present_s.append(f"{result.iloc[0][_DISCRETIZATION]}")

        cols_present_s.append(f"{result.iloc[0].name} - {result.iloc[-1].name}")

        str_present = " | ".join(cols_present_s)

    str_present = f"DF [{str_present} | {format_num(len(result))}]"

    return str_present


def func_multi(_iterator, _func, arg_s, num_workers=1, finished_title="", print_result_full=True, print_out=True):
    measure = produce_measure()
    _bi, _b, nl_ = produce_formatters(mode=None)

    finished_count = 0
    total_count = len(arg_s)
    progress_handle = produce_display_handler_MARKDOWN()

    def printer(result):
        if not print_out:
            return

        nonlocal finished_count
        finished_count += 1
        remain_count = total_count - finished_count

        if result is None:
            finished_title_ = f"{_b(finished_title)} | " if len(finished_title) > 0 else ""
            title = f"{finished_title_}R: {_b(remain_count)} | F: {_b(finished_count)} | T: {_b(total_count)} || NO PRESENTATION RESULT"
            if print_result_full:
                printmd(title)
            else:
                progress_handle(title)

            return

        elif isinstance(result, str) and 'REMAINS' in result:
            str_present = result % (remain_count, finished_count, total_count)
        elif isinstance(result, pd.DataFrame):
            str_present = handle_df(result)
        elif isinstance(result, dict):
            str_present = handle_dict(result)
        else:
            str_present = str(result)

        current_exception = 'ex' in str_present.lower() or 'broken' in str_present.lower() or 'failed' in str_present.lower() or 'error' in str_present.lower()
        if len(finished_title) > 0 and current_exception:
            finished_title_ = f'!!! FAILED !!! [{_b(finished_title)}] | '
        if len(finished_title) == 0 and current_exception:
            finished_title_ = f'!!! FAILED !!! | '
        if len(finished_title) > 0 and not current_exception:
            finished_title_ = f'{_b(finished_title)} | '
        if len(finished_title) == 0 and not current_exception:
            finished_title_ = ''

        if print_result_full:
            title = f"{finished_title_}R: {_b(remain_count)} | F: {_b(finished_count)} | T: {_b(total_count)} || {str_present}"
            printmd(title)
        else:
            title = f"{finished_title_}R: {_b(remain_count)} | F: {_b(finished_count)} | T: {_b(total_count)} || {str_present}"
            progress_handle(title)

    finished_title_ = f"{finished_title} | " if len(finished_title) > 0 else finished_title

    result_s = []
    if num_workers > 1 and len(arg_s) > 1 and not is_running_under_pycharm_debugger():
        if print_out:
            printmd(f'=== START {finished_title_}CONCURRENT EXECUTION | WORKERs: {_b(num_workers)} | JOBs: {_b(len(arg_s))} || STARTED: {_b(datetime_Y_m_d__h_m_s(kiev_now()))} ===')
        for result in _iterator(_func, arg_s, num_workers):
            if result is None:
                printer(result)
                continue

            printer(result)
            result_s.append(result)

        duration = measure()
        if print_out:
            printmd(f'=== END {finished_title_}EXECUTION | finished: {_b(datetime_Y_m_d__h_m_s(kiev_now()))} | DURATION: {_b(duration)} ===')
    else:
        if print_out:
            printmd(f'--- START {finished_title_}SEQUENTIAL EXECUTION | JOBs: {_b(len(arg_s))} || STARTED: {_b(datetime_Y_m_d__h_m_s(kiev_now()))} ---')
        for arg in arg_s:
            result = _func(arg)
            if result is None:
                printer(result)
                continue

            printer(result)
            result_s.append(result)

        duration = measure()
        if print_out:
            printmd(f'--- END {finished_title_}EXECUTION | FINISHED: {_b(datetime_Y_m_d__h_m_s(kiev_now()))} | DURATION: {_b(duration)} ---')

    time.sleep(2)

    return result_s


def func_multi_process(_func, arg_s, num_workers=1, finished_title="", print_result_full=is_running_under_pycharm(), print_out=True):
    return func_multi(iterate_multiprocess_pool, _func, arg_s, num_workers, finished_title, print_result_full, print_out)


def func_multi_process_executor(_func, arg_s, num_workers=1, finished_title="", print_result_full=True, print_out=True):
    return func_multi(iterate_multiprocess_executor, _func, arg_s, num_workers, finished_title, print_result_full, print_out)


def func_multi_thread_executor(_func, arg_s, num_workers=1, finished_title="", print_result_full=True, print_out=True):
    return func_multi(iterate_multithread_executor, _func, arg_s, num_workers, finished_title, print_result_full, print_out)


def func_multi_thread(_func, arg_s, num_workers):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    result_s = []

    if num_workers > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_data = {executor.submit(_func, arg): arg for arg in arg_s}

            for future in as_completed(future_to_data):
                try:
                    result = future.result()

                    result_s.append(result)
                except Exception as exc:
                    print(f"Task generated an exception: {exc}")
    else:
        for arg in arg_s:
            result = _func(arg)

            result_s.append(result)

    return result_s


def save_net(loss_s, epochs_count, _save_model, optimizer):
    train_loss = loss_s[:-1] if len(loss_s) - 2 == epochs_count else loss_s
    epoch = len(train_loss) - 1
    title_details = f"EPOCH: {epoch} | TIME: {datetime_h_m_s__d_m_Y(kiev_now())}"
    if loss_s[-1] <= min(train_loss[:-1]) or True:
        print(f'~~~ FORCE SAVE MODEL | {title_details} ~~~')
        _save_model(optimizer, epoch)
    else:
        print(f'!!! IGNORE | {title_details} >>>> loss_s[-1]={loss_s[-1]} | min(train_loss[:-1]={min(train_loss[:-1])} | train_loss={train_loss}!!!')


def write_model(net, optimizer=None, model_name_suffix=None, print_out=True):
    import torch
    from torch import nn

    folder = f'{MODEL_FOLDER_PATH()}'
    if not os.path.exists(folder):
        os.makedirs(folder)

    file_path = f'{folder}/{model_name_suffix}.pt'

    if torch.cuda.is_available() and torch.cuda.device_count() > 1 and isinstance(net, nn.DataParallel):
        state_dict = {
            "model": net.module.state_dict(),
        }
    else:
        state_dict = {
            "model": net.state_dict()
        }

    if optimizer is not None:
        state_dict["optimizer"] = optimizer.state_dict()

    net.write_configs(state_dict)
    torch.save(state_dict, file_path)

    if print_out:
        print(f"Model written: {file_path}")


def produce_empty_net(name_suffix) -> 'IModelBase':
    module_name = name_suffix.split("-")[0].split("__")[0]

    if 'main' in module_name:
        module = importlib.import_module(f'__{module_name}__')
    else:
        module = importlib.import_module(f'SRC.NN.{module_name}')

    class_ = getattr(module, 'NN')
    if class_.mro()[-5].__name__ == 'BaseNN' or class_.mro()[-4].__name__ == 'BaseNN':
        net = class_()
    elif class_.mro()[-5].__name__ == 'A_Base' or class_.mro()[-4].__name__ == 'A_Base':
        net = class_(True)
    elif class_.mro()[-5].__name__ == 'S_Base' or class_.mro()[-4].__name__ == 'S_Base':
        net = class_()
    else:
        raise RuntimeError(f"INSTANTIATE ERROR: {name_suffix} SHOULD BE INSTANCE OF: BaseNN | A_Base | S_Base")

    try:
        model_suffix = name_suffix.split("-")[1]
        os.environ[_MODEL_SUFFIX] = model_suffix
    except:
        pass

    return net


def is_close_to_zero(value, tolerance=1e-10):
    return abs(value) < tolerance


def floor(float_, order):
    round_order = math.pow(10, order)
    return math.floor(float_ * round_order) / round_order


def round_down_price_symbol(float_, price_tick_size):
    return floor(float_, get_round_order(float(price_tick_size)))


def round_price_tick(float_, tick):
    return round(float_, get_round_order(tick))


def round_price_symbol(float_, price_tick_size):
    return round_price_tick(float_, float(price_tick_size))


def num_zeros(decimal):
    from math import floor, log10, inf, isinf, nan, isnan

    if isnan(decimal):
        return nan

    if isinf(decimal):
        return -inf

    num_zeros = inf if decimal == 0 else -floor(log10(abs(decimal))) - 1

    return num_zeros


def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("localhost", port))
        except OSError:
            return True
        else:
            return False


def get_available_ports_count():
    public_ip, data_dict = parse_port_mapping()

    return len(data_dict.items())


def get_next_available_port():
    public_ip, data_dict = parse_port_mapping()
    for in_port_str, ex_port_str in data_dict.items():
        in_port = int(in_port_str)
        if in_port == 22:
            continue

        if not is_port_in_use(in_port):
            printmd(f"Local PORT: **{in_port}**")
            return in_port

    raise Exception("No free port available..")


def killme_price_aligner_checker(counter, title):
    from SRC.LIBRARIES.binance_helpers import get_market_symbol_ticker_current_price

    price_file_path = _DATETIME_PRICE_FILE_PATH()

    net_folder = price_file_path.split("/")[-2]
    net_data = parse_net_folder(net_folder)
    symbol = net_data['symbol']
    market = net_data['market']
    market_type = market.split("__")[0]

    price_local = get_datetime_price()['price']
    price_cloud = tryall_delegate(lambda: get_market_symbol_ticker_current_price(symbol, market_type), 'GET PRICE', tryalls_count=5)

    print(f"{title} -- KILLME PRICE [{symbol}]: {price_local} >> {price_cloud}")


def write_price_change_on_each(counter):
    multiplier = 1_000
    if counter < 1 * multiplier:
        write_every = 3
    elif counter < 2 * multiplier:
        write_every = 7
    elif counter < 4 * multiplier:
        write_every = 15
    elif counter < 8 * multiplier:
        write_every = 31
    else:
        write_every = 60

    return write_every


async def set_price(candles_secs_queue: Queue, stop_queue, stop_event=None, print_out=False):
    DEBUG_SPLITTED(f"ENTERED: PRICE(s) WRITER")

    price_file_path = _DATETIME_PRICE_FILE_PATH()
    prices_file_path = _DATETIME_PRICES_FILE_PATH()

    create_folder_file(price_file_path, clear_file=False)
    create_folder_file(prices_file_path, clear_file=False)

    def writer(candle, counter):
        datetime_price = {'date_time': candle['close_time'], 'price': candle['close']}
        write_json(datetime_price, price_file_path)

        if counter % write_price_change_on_each(counter) == 0:
            price_s = read_json_safe(prices_file_path, [])
            price_s.append(datetime_price)
            write_json(price_s, prices_file_path)

        # killme_price_aligner_checker(counter, 'EACH-1')
        #
        # if counter % 21 == 0 and counter > 0:
        #     time.sleep(17)
        #     killme_price_aligner_checker(counter, 'EACH-21')

    await write_candle_price(writer, candles_secs_queue, stop_queue, stop_event=stop_event, print_out=print_out)


async def set_candle(candles_secs_queue: Queue, stop_queue, stop_event=None, print_out=False):
    DEBUG_SPLITTED(f"ENTERED: CANDLE(s) WRITER")

    candle_file_path = _CANDLE_FILE_PATH()
    create_folder_file(candle_file_path, clear_file=True)

    def writer(candle, counter):
        write_json(candle, candle_file_path)

    await write_candle_price(writer, candles_secs_queue, stop_queue, stop_event=stop_event, print_out=print_out)


async def write_candle_price(_writer, candles_secs_queue, stop_queue, stop_event=None, print_out=False):
    try:
        counter = 0
        while True:
            DEBUG(f"PRICE TICK CANDLES STREAM WRITER QUEUE SIZE:   {candles_secs_queue.qsize()} | counter: {counter}")
            if candles_secs_queue.qsize() > 5:
                ERROR_SPLITTED(f"!! PRICE TICK CANDLES STREAM WRITER QUEUE SIZE:   {candles_secs_queue.qsize()} | counter: {counter} !!")

            if stop_event is not None and stop_event.is_set():
                break

            candle = await candles_secs_queue.get()
            _writer(candle, counter)

            if print_out:
                DEBUG(get_pretty_candle(candle))

            counter += 1

        if stop_queue is not None:
            await stop_queue.put({})
    except Exception as ex:
        EXCEPTION_SPLITTED()

        if stop_queue is not None:
            await stop_queue.put(ex)


def get_candle():
    rel_path = '/..' if _DASHBOARD_SEGMENT in os.environ and os.environ[_DASHBOARD_SEGMENT] == 'EVAL' else ''
    candle_file_path = _CANDLE_FILE_PATH(rel_path=rel_path)
    
    candle = read_json_safe_retry(candle_file_path)

    return candle


def get_balance_s():
    rel_path = '/..' if _DASHBOARD_SEGMENT in os.environ and os.environ[_DASHBOARD_SEGMENT] == 'EVAL' else ''
    balance_file_path = _BALANCE_FILE_PATH(rel_path=rel_path)
    
    balance_s = read_json_safe_retry(balance_file_path)

    return balance_s


def get_datetime_price_s():
    rel_path = '/..' if _DASHBOARD_SEGMENT in os.environ and os.environ[_DASHBOARD_SEGMENT] == 'EVAL' else ''
    datetime_prices_file_path = _DATETIME_PRICES_FILE_PATH(rel_path=rel_path)

    datetime_price_s = read_json_safe_retry(datetime_prices_file_path)

    return datetime_price_s


def get_datetime_price():
    rel_path = '/..' if _DASHBOARD_SEGMENT in os.environ and os.environ[_DASHBOARD_SEGMENT] == 'EVAL' else ''
    datetime_price_file_path = _DATETIME_PRICE_FILE_PATH(rel_path=rel_path)

    datetime_price = read_json_safe_retry(datetime_price_file_path)

    return datetime_price


def write_datetime_price_candle(group):
    price_file_path = _DATETIME_PRICE_FILE_PATH()
    candle_file_path = _CANDLE_FILE_PATH()

    create_folder_file(price_file_path, clear_file=False)
    create_folder_file(price_file_path, clear_file=False)

    target_df = group[0]
    last_row = target_df.iloc[-1]
    last_row_utc_ts = last_row['utc_timestamp'].to_pydatetime()
    candle = {'close_time': last_row_utc_ts, 'open': last_row['open'], 'high': last_row['high'], 'low': last_row['low'], 'close': last_row['close']}
    datetime_price = {'date_time': candle['close_time'], 'price': candle['close']}
    write_json(datetime_price, price_file_path)
    write_json(candle, candle_file_path)


def show_notebook_train_result(notebook_name):
    from IPython.display import Image, display
    from SRC.CORE._CONSTANTS import MODEL_EVALUATION_IMG_PATH
    try:
        img_name = f"{notebook_name}.png"
        img_path = f"{MODEL_EVALUATION_IMG_PATH()}/{img_name}"

        display(Image(filename=img_path))
    except:
        printmd(f"!!No saved **{img_name}** model result was found!!")


def reverse_lines_in_file(in_file_path, out_file_path=None):
    if out_file_path is None:
        out_file_path = in_file_path

    try:
        with open(in_file_path, 'r') as f:
            lines = f.readlines()

        reversed_lines = reversed(lines)

        with open(out_file_path, 'w') as f:
            for line in reversed_lines:
                f.write(line)
    except IOError:
        print("Error: File not found or cannot be accessed.")


def string_bool(string_bool):
    return bool(distutils.util.strtobool(string_bool))


def set_env(key, value):
    value = str(value)

    if key in os.environ:
        if os.environ[key] == value:
            print(f"env: {key}={os.environ[key]}")
        else:
            printmd(f"ENV: {key}={value} >> ***{os.environ[key]}***")
    else:
        os.environ[key] = value
        print(f"env: {key}={os.environ[key]}")


def env_string(env_key, default=None):
    value = os.environ[env_key] if env_key in os.environ else default
    value = None if value == "None" else value

    return value


def env_float(env_key, default=np.nan):
    return float(os.environ[env_key]) if env_key in os.environ else default


def env_int(env_key, default=np.nan):
    return int(os.environ[env_key]) if env_key in os.environ else default


def env_list(env_key, default=[]):
    env_list = os.environ[env_key].split("|") if env_key in os.environ else []

    return env_list


def check_env_true(env_key, default_val=False):
    try:
        return string_bool(os.environ[env_key])
    except:
        return default_val


def calculate_weighted_average_price(fills):
    total_value = sum(float(fill["qty"]) * float(fill["price"]) for fill in fills)
    total_amount = sum(float(fill["qty"]) for fill in fills)

    weighted_average_price = total_value / total_amount

    return weighted_average_price


def printmd_log(*msg_log_s, log_file, to_console=False, to_file=True):
    _print = lambda txt: printmd(txt)
    print_log(*msg_log_s, log_file=log_file, to_console=to_console, to_file=to_file, _print=_print)


def printHTML_log(*msg_log_s, log_file, to_console=False, to_file=True):
    _print = lambda txt: display(HTML(txt.replace('"', '').replace('***', '').replace('**', '').replace('`', '').replace('\r\n\r\n', '\r\n').replace("&#36;", "$")))
    print_log(*msg_log_s, log_file=log_file, to_console=to_console, to_file=to_file, _print=_print)


def print_log(*msg_log_s, log_file, to_console=False, to_file=True, _print=lambda txt: print(txt)):
    formatted_msg_log_s = [msg_log.replace("&#36;", "$") for msg_log in msg_log_s if not inspect.isfunction(msg_log)]

    if to_console:
        for formatted_msg_log in formatted_msg_log_s:
            _print(formatted_msg_log)
            sys.stdout.flush()

    if to_file:
        append_file(log_file, *formatted_msg_log_s)


def printmd_log_trades(*msg_log_s, to_console=False, to_file=True):
    printmd_log(*msg_log_s, log_file=_TRADES_FILE_PATH(), to_console=to_console, to_file=to_file)


def printHTML_log_trades(*msg_log_s, to_console=False, to_file=True):
    printHTML_log(*msg_log_s, log_file=_TRADES_FILE_PATH(), to_console=to_console, to_file=to_file)


def print_log_trades(*msg_log_s, to_console=False, to_file=True):
    print_log(*msg_log_s, log_file=_TRADES_FILE_PATH(), to_console=to_console, to_file=to_file)


def printmd_log_runner(*msg_log_s, to_console=False, to_file=True):
    printmd_log(*msg_log_s, log_file=_TRADES_RUNNER_FILE_PATH(), to_console=to_console, to_file=to_file)


def print_log_runner(*msg_log_s, to_console=False, to_file=True):
    print_log(*msg_log_s, log_file=_TRADES_RUNNER_FILE_PATH(), to_console=to_console, to_file=to_file)


def produce_net_folder(data):
    hash = generate_net_hash(filter_data_features(data))
    market = data['market'].strip()
    if 'symbol_slash' in data:
        dash_symbol = _symbol_dash(data['symbol_slash']).strip()
    elif 'symbol' in data:
        dash_symbol = _symbol_dash(data['symbol']).strip()
    else:
        dash_symbol = data['symbol_dash'].strip()

    if 'model_name' in data:
        net_name = data['model_name'].strip()
    elif 'model' in data:
        net_name = data['model'].strip()
    else:
        raise RuntimeError(f"NO `model_name` NOR `model` PROPERTY `net_data`")

    subfolder_segment = f"{market}|{dash_symbol}|{net_name}|{hash}"

    # subfolder_segment = encode_path_segment(subfolder_segment)

    return subfolder_segment


# File-safe encoding mapping
ENCODE_MAP = {
    "|": "__PIPE__",
    ":": "__COLON__",
    "/": "__SLASH__",
    "\\": "__BACKSLASH__",
    "?": "__Q__",
    "*": "__STAR__",
    "<": "__LT__",
    ">": "__GT__",
    '"': '__QUOTE__',
}

# Invert the mapping for decoding
DECODE_MAP = {v: k for k, v in ENCODE_MAP.items()}


def encode_path_segment(segment: str) -> str:
    for char, replacement in ENCODE_MAP.items():
        segment = segment.replace(char, replacement)
    return segment


def decode_path_segment(encoded: str) -> str:
    for replacement, char in DECODE_MAP.items():
        encoded = encoded.replace(replacement, char)
    return encoded


def parse_net_folder(net_folder):
    # net_folder = decode_path_segment(net_folder)

    sub_string = [sub_str.strip() for sub_str in net_folder.split("|")]

    data = {
        'market': sub_string[0].strip(),
        'symbol': f"{_symbol_join(sub_string[1])}".strip(),
        'model_name': sub_string[2].strip()
    }

    return data


@lru_cache(maxsize=None)
def parse_net_folder_hashed(net_folder):
    # net_folder = decode_path_segment(net_folder)

    sub_string = net_folder.split("|")

    data = {
        'market': sub_string[0].strip(),
        'symbol': f"{_symbol_join(sub_string[1])}".strip(),
        'model_name': sub_string[2].strip()
    }

    data['hash'] = generate_net_hash(filter_data_features(data))

    return data


def get_symbol_autotradingregime_markettype(folder_data, _log_func=DEBUG_SPLITTED):
    with log_context(_log_func) as log_s:
        log_s.append(folder_data)

        net_data = folder_data
        if isinstance(folder_data, str):
            net_data = parse_net_folder_hashed(folder_data)

        log_s.append(net_data)

        market = net_data['market']
        market_type = market.split("__")[0]
        symbol = net_data['symbol']

        log_s.append(market)

        _regime = 'REGIME_'
        var_s = parse_string_variables(market, [_regime])
        autotrading_regime = var_s[_regime]

        symbol_markettype = f"{symbol}-{autotrading_regime}-{market_type}"

        log_s.append(symbol_markettype)

        return symbol_markettype


def is_valid_net_folder(net_folder):
    try:
        parse_net_folder_hashed(net_folder)

        return True
    except:
        return False


def filter_data_features(data):
    data_f = {key: data[key].strip() for key in data if key in ['market', 'symbol', 'model_name']}

    return data_f


def is_net_data_data_equals(data, _data):
    data_f = filter_data_features(data)
    _data_f = filter_data_features(_data)

    net_data_data_equals = data_f == _data_f

    return net_data_data_equals


def is_net_folder_data_equals(data, net_folder):
    _data = parse_net_folder(net_folder)
    folder_data_equals = is_net_data_data_equals(data, _data)

    return folder_data_equals


def test__produce_parse_subfolder_segment():
    data = {
        'market': 'SPOT',
        'symbol': 'BTCUSDT',
        'max_orders_count': 2,
        'model_name': 'Linn1D',
    }

    data['hash'] = generate_net_hash(data)

    net_folder = produce_net_folder(data)
    _data = parse_net_folder(net_folder)

    assert is_net_folder_data_equals(_data, net_folder), f"!!WRONG PRODUCE-PARSE LOGIC!!"
    print('Success')


def get_net_folder_s(dashboard_segment) -> []:
    directory = Path(f"{project_root_dir()}/{OUT_SEGMENT()}/{dashboard_segment}/")
    os.makedirs(directory, exist_ok=True)
    folders = get_subfolder_s(directory)

    return folders


def get_subfolder_s(parent_dir_path):
    all_contents = os.listdir(parent_dir_path)
    folders = [item for item in all_contents if os.path.isdir(os.path.join(parent_dir_path, item)) and 'ipynb_checkpoints' not in item]

    return [folder for folder in folders if is_valid_net_folder(folder)]


def get_netfolder_by_hash(hash):
    dashboard_segment_s = [_AUTOTRADING, _BACKTESTING]
    net_folder_s = [
        item
        for dashboard_segment in dashboard_segment_s
        for item in get_net_folder_s(dashboard_segment)
    ]

    filtered_net_folder_s = [net_folder for net_folder in net_folder_s if hash in net_folder]
    if len(filtered_net_folder_s) > 0:
        return filtered_net_folder_s[0]
    else:
        return None


def filter_already_simulated_data(data_s, regime_segment):
    net_folders = ['|'.join(folder.split('|')[:-1]) for folder in get_net_folder_s(regime_segment)]

    filtered_already_exist_data_s = []
    for data in data_s:
        net_folder = '|'.join(produce_net_folder(data).split('|')[:-1])
        if net_folder not in net_folders:
            filtered_already_exist_data_s.append(data)

    net_len = len(net_folders)
    filtered_len = len(filtered_already_exist_data_s)
    total_len = len(data_s)

    printmd(f"EXISTING NET FOLDERS: **{net_len}** | REMAINING NET FOLDERS: **{filtered_len}** | CURRENT SETUP: **{total_len}**")

    return filtered_already_exist_data_s


def rename_project_folder(original_folder, renamed_folder):
    current_name = f'{project_root_dir()}/{OUT_SEGMENT()}/{original_folder}'
    new_name = f'{project_root_dir()}/{OUT_SEGMENT()}/{renamed_folder}'
    os.rename(current_name, new_name)


def generate_hash(dictionary):
    json_string = json.dumps(dictionary, sort_keys=True)
    hash_value = hashlib.sha256(json_string.encode()).hexdigest()

    return hash_value


def generate_net_hash(net_data):
    net_hash = generate_hash(net_data)[:5]

    return net_hash


def find_duplicates(strings):
    count_dict = {}
    duplicates = []

    for string in strings:
        count_dict[string] = count_dict.get(string, 0) + 1

    for string, count in count_dict.items():
        if count > 1:
            duplicates.append(string)

    return duplicates


def list_dict_unique(list_of_dicts, key):
    seen = set()
    filtered_list = []

    for d in list_of_dicts:
        if d[key] not in seen:
            seen.add(d[key])
            filtered_list.append(d)

    return filtered_list


def filter_dict(original_dict, filter_field_s, *, exclude=True):
    if exclude:
        _predicate = lambda key: key not in filter_field_s
    else:
        _predicate = lambda key: key in filter_field_s

    return {
        key: value for key, value in original_dict.items() if _predicate(key)
    }


def find_list_items(item_s, key, val):
    if key == '':
        filtered_item_s = [item for item in item_s if item == val]
    else:
        filtered_item_s = [item for item in item_s if item[key] == val]

    return filtered_item_s


def find_first_list_item(item_s, key, val, default=None):
    filtered_item_s = find_list_items(item_s, key, val)
    if len(filtered_item_s) > 0:
        return filtered_item_s[0]
    else:
        return default


def read_file(file_path):
    with open(file_path, 'r') as file:
        file_content = file.read()

    return file_content


def write_file(content, file_path):
    create_folder_file(file_path)

    with open(file_path, 'a') as file:
        file.write(content)
        file.write("\r\n")


def append_file(file_path, *msg_s):
    create_folder_file(file_path)

    while True:
        try:
            with open(file_path, "a") as file:
                for msg in msg_s:
                    file.write(msg)
                    file.write("\r\n")

            return
        except FileNotFoundError:
            raise
        except:
            wait = random.uniform(0, 1)
            time.sleep(wait)


def write__rootdir__killme(content):
    write_file(f"{content}\r\n", f"/Users/andriulik/Documents/CRYPTO_BOT/killme.txt")
    # write_file(f"{project_root_dir()}/killme.txt", content)


def rename_net_folders_ids_hash(regime_segment):
    runner_file_path = _TRADES_RUNNER_FILE_PATH()
    runner_file_content = read_file(runner_file_path)
    net_folder_s = get_net_folder_s(regime_segment)
    net_folder_strip_s = ['|'.join(folder.split('|')[:-1]) for folder in net_folder_s]

    duplicates_net_folder_s = find_duplicates(net_folder_strip_s)
    if len(duplicates_net_folder_s) > 0:
        raise Exception(f"Duplicates net_folder_strip_s: {duplicates_net_folder_s}")

    net_data_hash_s = []
    net_folder_hash_s = []
    for net_folder in net_folder_s:
        net_data = parse_net_folder(net_folder)
        net_data['hash'] = generate_net_hash(filter_data_features(net_data))
        net_data_hash_s.append(net_data)
        net_folder_hash = produce_net_folder(net_data)
        net_folder_hash_s.append(net_folder_hash)

    duplicates_net_folder_hash_s = find_duplicates(net_folder_hash_s)
    if len(duplicates_net_folder_hash_s) > 0:
        raise Exception(f"Duplicates net_folder_hash_s: {duplicates_net_folder_hash_s}")

    for i in range(len(net_folder_s)):
        net_folder = net_folder_s[i]
        net_data_hash = net_data_hash_s[i]
        net_folder_hash = net_folder_hash_s[i]
        idx_hash = net_folder.split('|')[-1]
        hash = net_data_hash['hash']
        if idx_hash == hash:
            # join_symbol = symbol_join(net_data_hash['symbol_slash']11)
            # dash_symbol = symbol_dash(net_data_hash['symbol_slash']11)
            # net_folder_hash_dashed_symbol = net_folder_hash.replace(join_symbol, dash_symbol)
            # rename_project_folder(f"{regime_segment}/{net_folder_hash}", f"{regime_segment}/{net_folder_hash_dashed_symbol}")
            continue

        rename_project_folder(f"{regime_segment}/{net_folder}", f"{regime_segment}/{net_folder_hash}")
        runner_file_content = runner_file_content.replace(f"[{idx_hash}]", f"[{hash}]").replace(f"|{idx_hash}**", f"|{hash}**")

    write_file(runner_file_content, runner_file_path)

    net_folder_hash_s = get_net_folder_s(regime_segment)
    for net_folder_hash in net_folder_hash_s:
        net_data_hash = parse_net_folder(net_folder_hash)
        single_equals = [is_net_data_data_equals(net_data, net_data_hash) for net_data in net_data_hash_s].count(True) == 1
        if single_equals:
            continue
        else:
            print(net_data)
            print(net_data_hash)
            raise Exception(f"WRONG MAPPING {net_folder_hash}")

    print(f'FOLDERS RENAMED | COUNT: {len(net_folder_hash_s)}')


def create_folder_file(file_path, clear_file=False):
    dir_name = os.path.dirname(file_path)
    if dir_name != '':
        os.makedirs(dir_name, exist_ok=True)

    if clear_file:
        with open(file_path, 'w'):
            pass


def empty_folder(folder_path):
    if os.path.isdir(folder_path) and os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)  # remove file or symlink
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path, ignore_errors=True)  # remove directory


def top_up_net_folder(folder_path=None):
    with FileLock(f"{project_root_dir()}/locks/{folder_path}.lock"):
        folder_path = _TOP_UP_NET_FOLDER_PATH() if folder_path is None else folder_path
        os.makedirs(folder_path, exist_ok=True)
        os.rmdir(folder_path)


def INITIALIZE_NETFOLDER(automation_type, reset=False):
    if reset:
        net_folder_full_path = _DASHBOARD_SEGMENT_NET_FULL_PATH()
        shutil.rmtree(net_folder_full_path, ignore_errors=True)

    def CLEAR_FILES():
        top_up_net_folder()

        if automation_type == _BACKTESTING:
            create_folder_file(_CANDLE_FILE_PATH(), clear_file=True)
        else:
            create_folder_file(_DATETIME_PRICES_FILE_PATH(), clear_file=True)

        create_folder_file(_BALANCE_FILE_PATH(), clear_file=True)
        create_folder_file(_DATETIME_PRICE_FILE_PATH(), clear_file=True)
        create_folder_file(_TRADES_FILE_PATH(), clear_file=True)

    return CLEAR_FILES


def _symbol_dash(symbol):
    if '_' in symbol:
        return symbol

    if '/' in symbol:
        return symbol.replace("/", "_")

    quote_currencies = ["USDT", "USD", "BUSD", "USDC"]
    for quote in quote_currencies:
        if symbol.endswith(quote):
            base = symbol[:-len(quote)]
            return f"{base}_{quote}"

    return symbol


def _symbol_slash(symbol):
    if '/' in symbol:
        return symbol

    if '_' in symbol:
        return symbol.replace("_", "/")

    quote_currencies = ["USDT", "USD", "BUSD", "USDC"]
    for quote in quote_currencies:
        if symbol.endswith(quote):
            base = symbol[:-len(quote)]
            return f"{base}/{quote}"

    return symbol


def _symbol_join(symbol):
    if '/' in symbol:
        return symbol.replace("/", "")

    if '_' in symbol:
        return symbol.replace("_", "")

    return symbol


async def empty_coroutine():
    pass


def read_text_to_list(file_path):
    text_s = []

    with open(file_path) as file_to_read:
        for line in file_to_read:
            text_s.append(line)

    return text_s


def read_text_to_list_safe(file_path):
    try:
        return read_text_to_list(file_path)
    except Exception as ex:
        print(ex)

        return []


def update_trade_net_folder_name(origin_net_folder, net_folder):
    try:
        net_data = parse_net_folder(net_folder)
        net_data['hash'] = generate_net_hash(filter_data_features(net_data))
        net_folder_hash = produce_net_folder(net_data)
        new_net_folder = f"TRADE/{net_folder_hash}"
        new_net_folder_path = f'{project_root_dir()}/{OUT_SEGMENT()}/{new_net_folder}'
        new_net_folder_path_killme = f"{new_net_folder_path}/killme"

        existing_net_folders = get_net_folder_s('TRADE')
        existing_net_folder = [existing_net_folder for existing_net_folder in existing_net_folders if origin_net_folder in existing_net_folder][0]
        origin_net_folder = f"TRADE/{existing_net_folder}"
        rename_project_folder(origin_net_folder, new_net_folder)

        print(f"UPDATED TRADE NET FOLDER")
        print(f"{origin_net_folder}")
        print(f"{new_net_folder}")

        top_up_net_folder(new_net_folder_path_killme)
    except IndexError as ex:
        if os.path.exists(new_net_folder_path):
            print(f"NO NEED UPDATE TRADE NET FOLDER | EXISTING: {new_net_folder}")
            return
        print(f"FAILED UPDATE TRADE NET FOLDER | NO EXISTING: {origin_net_folder}")
        raise
    except Exception as ex:
        print(f"FAILED UPDATE TRADE NET FOLDER | NO EXISTING: {origin_net_folder}")
        raise


def remove_non_empty_folder(folder_path):
    # Check if the folder exists
    if os.path.exists(folder_path):
        # Check if the folder is not empty
        if os.path.isdir(folder_path) and os.listdir(folder_path):
            # Remove the folder and its contents
            shutil.rmtree(folder_path)
            print(f"Folder '{folder_path}' and its contents have been removed.")
        else:
            print(f"Folder '{folder_path}' is either empty or not a directory.")
    else:
        print(f"Folder '{folder_path}' does not exist.")


def remove_net_folder_matching_s(net_folder_matching_s, regime_segment):
    os.environ[_DASHBOARD_SEGMENT] = regime_segment
    net_folder_s = get_net_folder_s(os.environ[_DASHBOARD_SEGMENT])
    remove_net_folder_s = []
    for net_folder_matching in net_folder_matching_s:
        for net_folder in net_folder_s:
            if net_folder_matching in net_folder:
                remove_net_folder_s.append(net_folder)

    for remove_net_folder in remove_net_folder_s:
        remove_net_folder_path = f"{project_root_dir()}/OUT/{regime_segment}/{remove_net_folder}"
        remove_non_empty_folder(remove_net_folder_path)


def nexter(item_s):
    idx = 0

    def next():
        nonlocal idx
        nonlocal item_s
        if idx >= len(item_s):
            idx = 0
        item = item_s[idx]
        idx += 1

        return item

    return next


def build_conf_mtx_info_str(conf_mtx, signal_encoder: 'SignalEncoder'):
    arrow_up_symbol = "\u2191"
    arrow_down_symbol = "\u2193"
    double_arrow_up_symbol = "\u21D1"
    double_arrow_down_symbol = "\u21D3"

    clazz_s = list(range(signal_encoder.clazzes_count()))
    # if len(clazz_s) > 3:
    #     return None

    info_s = []
    for clazz in range(signal_encoder.clazzes_count()):
        other_clazz_s = [clz for clz in clazz_s if clz != clazz]
        clazz_true = conf_mtx[clazz][clazz]
        clazz_false = sum([conf_mtx[other_clazz][clazz] for other_clazz in other_clazz_s])

        clazz_false_true = clazz_false / clazz_true
        clazz_info_format = f"{signal_encoder.label_s()[clazz]}: <b>{arrow_down_symbol} {round(clazz_false_true, 2)}</b>"
        if clazz % 7 == 0:
            clazz_info_format = f"{clazz_info_format}\r\n"

        info_s.append(clazz_info_format)

    info_format = ' | '.join(info_s)

    return info_format


def get_round_order(input_number):
    result = -math.log10(input_number)
    return int(result)


def get_pretty_candle(candle):
    # pretty_candle = datetime_h_m_s__d_m_Y(candle['close_time'].astimezone(KIEV_TZ))

    dt_kiev_present = datetime_h_m_s__d_m_Y(candle['close_time'].astimezone(KIEV_TZ))
    pretty_candle = f"{dt_kiev_present} || open: {candle['open']} | high: {candle['high']} | low: {candle['low']} | close: {candle['close']}"

    return pretty_candle


def get_pretty_datetime_price(datetime_price):
    # pretty_candle = datetime_h_m_s__d_m_Y(candle['close_time'].astimezone(KIEV_TZ))
    date_time = datetime_price['date_time']
    price = datetime_price['price']
    dt_kiev_present = datetime_h_m_s__d_m_Y(date_time.astimezone(KIEV_TZ))
    pretty_candle = f"{dt_kiev_present} || price: {price}"

    return pretty_candle


def is_no_trades_timeout():
    file_modified_ts = getmtime(_BALANCE_FILE_PATH())
    is_trades_timeout = (datetime.now() - datetime.fromtimestamp(file_modified_ts)) > NO_TRADES_TIMEOUT()

    return is_trades_timeout


def is_over_timeout(created_dt):
    is_trades_timeout = (datetime.now() - created_dt) > OVER_TIMEOUT()

    return is_trades_timeout


def is_recently_updated(file_path, selected_interval):
    if isinstance(selected_interval, timedelta):
        selected_interval_time_delta = selected_interval
    else:
        selected_interval_time_delta = TIME_DELTA(selected_interval)

    if os.path.exists(file_path):
        now_dt = datetime.now()
        file_modified_dt = datetime.fromtimestamp(getmtime(file_path))
        file_modify_age = now_dt - file_modified_dt
        is_recently_updated = file_modify_age <= selected_interval_time_delta

        return is_recently_updated

    return False


def is_running(net_folder_data, selected_interval, dashboard_segment=None):
    if isinstance(selected_interval, str):
        selected_interval = TIME_DELTA(selected_interval)

    if isinstance(net_folder_data, dict):
        net_folder_data = produce_net_folder(net_folder_data)

    balance_file_path = _BALANCE_FILE_PATH(dashboard_segment=dashboard_segment, net_folder=net_folder_data)
    price_file_path = _DATETIME_PRICE_FILE_PATH(dashboard_segment=dashboard_segment, net_folder=net_folder_data)
    is_running = is_recently_updated(balance_file_path, selected_interval) or is_recently_updated(price_file_path, selected_interval)

    return is_running


def tryall_delegate(func, label=None, tryalls_count=3):
    exception = None
    for i in range(tryalls_count):
        msg = " | ".join(([f"!![{label}]!!", f"{i} of {tryalls_count}"] if label is not None else [f"{i} of {tryalls_count}"]))
        try:
            result = func()

            return result
        except (ReadTimeout, ConnectionError, requests.exceptions.ConnectionError) as error:
            exception = error
            ERROR_SPLITTED(f"READ TIMEOUT or CONNECTION ERROR: {msg}")
            time.sleep(random.randint(2*(i+1), 5*(i+1)))
            continue
        except binance.exceptions.BinanceAPIException as binanceEx:
            exception = binanceEx
            if binanceEx.code == -1008:
                ERROR_SPLITTED(f"SERVER OVERLOADED: {msg}")
                time.sleep(random.randint(2, 5))
                continue
            elif binanceEx.code == -1003:
                ERROR_SPLITTED(f"TOO MANY REQUESTS: {msg}")
                time.sleep(random.randint(2, 10))
                continue
            elif binanceEx.code == -3044:
                ERROR_SPLITTED(f"SYSTEM BUSY: {msg}")
                time.sleep(random.randint(5, 10))
                continue
            elif binanceEx.code == -2013:
                ERROR_SPLITTED(f"NO ORDER FOUND: {msg}")
                time.sleep(random.randint(2, 3))
                continue
            else:
                raise

    raise Exception(f"UNABLE TO: {label} | {tryalls_count} times") from exception


def is_called_from_parent_notebook():
    import os

    called_from_parent = os.getenv('PARENT_NOTEBOOK') == 'True'

    return called_from_parent


def get_subprocesses(parent_pid):
    import psutil

    try:
        parent = psutil.Process(parent_pid)
        children = parent.children(recursive=True)
        child_pids = [child.pid for child in children]
        return child_pids
    except psutil.NoSuchProcess:
        return []


def get_gpu_memory():
    import torch

    #
    # return np.array([[ 2682257408,  9932111872, 12884901888],
    #                  [ 2623537152, 10000269312, 12884901888],
    #                  [ 4448059392,  8175747072, 12884901888]]), 0

    used_gpu_memory = [int(x) for x in subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'], encoding='utf-8').strip().split('\n')]
    free_gpu_memory = [int(x) for x in subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'], encoding='utf-8').strip().split('\n')]
    total_gpu_memory = [int(x) for x in subprocess.check_output(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,nounits,noheader'], encoding='utf-8').strip().split('\n')]

    gpu_memory = np.rot90(np.array([used_gpu_memory, free_gpu_memory, total_gpu_memory]) * 1024 ** 2, k=1)
    max_gpu_memory_allocated_curr_process = torch.cuda.max_memory_allocated()

    return gpu_memory, max_gpu_memory_allocated_curr_process


def RAM():
    import psutil

    memory_info = psutil.virtual_memory()

    return memory_info


def RAM_format(ram=None):
    from SRC.CORE.debug_utils import format_memory

    if ram is None:
        memory_info = RAM()
    else:
        memory_info = ram

    ram_format = {
        "total": format_memory(memory_info.total),  # Total physical memory in bytes
        "available": format_memory(memory_info.available),  # Available memory
        "used": format_memory(memory_info.used),  # Used memory
        "free": format_memory(memory_info.free),  # Free memory
        "percentage": memory_info.percent  # Percentage of memory used
    }

    return ram_format


def produce_resource_usage_format(mode=_RESOURCES_FORMAT_JUPYTER):
    cumulate_RAM_used_byte_s = []
    cum_gpu_max = {0: [], 1: [], 2: [], 3:[]}
    total_max = []

    _bi, _b, nl_ = produce_formatters(mode)

    def get_resources_usage_format():
        device, device_name_formatted = get_processing_device(mode=mode)

        ram = RAM()
        cumulate_RAM_used_byte_s.append(ram.used)
        ram_format = RAM_format(ram)
        max_ram_used_format = format_memory(max(cumulate_RAM_used_byte_s))
        ram_format_present = f"[AVAIL: {_bi(ram_format['available'])} | FREE: {_bi(ram_format['free'])} | USED: {_bi(ram_format['used'])} || MAX USED: {_bi(max_ram_used_format)}]"
        total, used, free = shutil.disk_usage("/")
        storage_format_present = f"[AVAIL: {_bi(format_memory(total))} | FREE: {_bi(format_memory(free))} | USED: {_bi(format_memory(used))}]"

        if str(device) == 'cuda' and is_cloud():
            gpu_memory, max_gpu_memory_allocated_curr_process = get_gpu_memory()
            gpu_memory_formatted_s = []
            total_used = 0
            total_free = 0
            total_available = gpu_memory[:, 2].sum()
            for idx, memory in enumerate(gpu_memory):
                used = memory[0]
                free = memory[1]
                available = memory[2]

                cum_gpu_max[idx].append(used)
                total_used += used
                total_free += free

                gpu_memory_formatted = f"{_b(idx)}: [USED: {_bi(format_memory(used))} | MAX: {_bi(format_memory(max(cum_gpu_max[idx])))}]"
                gpu_memory_formatted_s.append(gpu_memory_formatted)

            total_max.append(total_used)

            gpu_format_present = f"[AVAIL: {_bi(format_memory(total_available))} | FREE: {_bi(format_memory(total_free))} | USED: {_bi(format_memory(total_used))} | MAX USED: {_bi(format_memory(max(total_max)))}] || {' || '.join(gpu_memory_formatted_s)}"
            resources_usage_format = f"{device_name_formatted}{nl_}{_b('RAM')} {ram_format_present}{nl_}{_b('GPU')} {gpu_format_present}{nl_}{_b('DISK')} {storage_format_present}"

            return resources_usage_format
        else:
            resources_usage_format = f"{device_name_formatted}{nl_}{_b('RAM')} {ram_format_present}{nl_}{_b('DISK')} {storage_format_present}"

            return resources_usage_format

    return get_resources_usage_format


def print_resources_usage_jupyter():
    resources_usage_format_jupyter = produce_resource_usage_format(_RESOURCES_FORMAT_JUPYTER)

    display(Markdown(resources_usage_format_jupyter()))


def RAM_usage():
    from SRC.CORE.debug_utils import format_memory
    import psutil
    import os
    import gc

    gc.collect()
    current_pid = os.getpid()
    root_process = psutil.Process(current_pid)
    # while True:
    #     process = root_process.parent()
    #     if process is None or process.name() in ['kernel_task', 'launchd']:
    #         break
    #
    #     root_process = process

    parent_process_memory_usage_bytes = root_process.memory_info().rss

    subprocess_ids = get_subprocesses(current_pid)
    sub_processes_memory_usage_bytes = 0
    for subprocess_id in subprocess_ids:
        try:
            process = psutil.Process(subprocess_id)
            sub_processes_memory_usage_bytes += process.memory_info().rss
        except AttributeError:
            pass
        except psutil.ZombieProcess:
            pass
        except psutil.NoSuchProcess:
            pass

    # print(f"Parent process memory usage: {format_memory(parent_process_memory_usage_bytes)}")
    # print(f"Sub processes memory usage: {format_memory(sub_processes_memory_usage_bytes)}")

    # total_ram_usage = parent_process_memory_usage_bytes + sub_processes_memory_usage_bytes
    # total_ram_usage_format = format_memory(total_ram_usage)
    total_ram_usage_format = format_memory(parent_process_memory_usage_bytes)

    return total_ram_usage_format


def split_list_into_chunks(input_list, chunk_size):
    if chunk_size == 0:
        return [input_list]

    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]


def clean_notebook(notebook_name, rel_path='NOTEBOOKS'):
    from SRC.CORE.utils import read_json, write_json

    if rel_path is not None:
        notebook_file_full_path = f"{project_root_dir()}/SRC/{rel_path}/{notebook_name}.ipynb"

    notebook_json = read_json(notebook_file_full_path)

    for cell in notebook_json['cells']:
        if 'execution_count' in cell:
            del cell['execution_count']

        if 'collapsed' in cell['metadata']:
            del cell['metadata']['collapsed']

        if 'jupyter' in cell['metadata'] and 'outputs_hidden' in cell['metadata']['jupyter']:
            del cell['metadata']['jupyter']['outputs_hidden']

        if 'pycharm' in cell['metadata'] and 'is_executing' in cell['metadata']['pycharm']:
            del cell['metadata']['pycharm']['is_executing']

    write_json(notebook_json, notebook_file_full_path)


def remove_list_duplicates(input_list):
    unique_list = []
    for item in input_list:
        if item not in unique_list:
            unique_list.append(item)

    return unique_list


def remove_list_duplicates_by_key(data, key):
    seen = set()
    unique_data = []

    for item in data:
        key_val = item[key]
        if key_val not in seen:
            unique_data.append(item)
            seen.add(key_val)

    return unique_data


def slice_list_start_end(arr, start, end):
    # Handle the special case where end is 0, which should return the whole array
    if end == 0:
        return arr[start:]  # Equivalent to [:] if start is 0
    return arr[start:-end]


def populate_char_n_times(char, times, title=None):
    if title is None:
        result_str = "".join([char for i in range(times)])
    else:
        times = int(times / 2)
        char_str = "".join([char for i in range(times)])
        result_str = f"{char_str}   {title}   {char_str}"

    return result_str


def print_populated_char_n_times(char, times, title=None):
    print(populate_char_n_times(char, times, title=title))


def printmd_populated_char_n_times(char, times, title=None, decorate='', color=None, override=True):
    if decorate == '':
        printmd(populate_char_n_times(char, times, title=title), color=color)
    else:
        if override:
            if decorate == '**':
                printmd(f"<b>{populate_char_n_times(char, times, title=title)}<b>", color=color)
            if decorate == '***':
                printmd(f"<b><i>{populate_char_n_times(char, times, title=title)}</i></b>", color=color)
        else:
            printmd(f"{decorate}{populate_char_n_times(char, times, title=title)}{decorate}", color=color)


def merge_dict_s(target_dicts):
    final_dict = {}
    for target_dict in target_dicts:
        final_dict = merge_dicts(target_dict, final_dict)

    return final_dict


def merge_dicts(origin_dict, target_dict):
    for key, value in target_dict.items():
        if key in origin_dict:
            if isinstance(origin_dict[key], dict) and isinstance(value, dict):
                # If both are dictionaries, recursively merge
                origin_dict[key] = merge_dicts(origin_dict[key], value)
            else:
                # Otherwise, override the value
                origin_dict[key] = value
        else:
            # Add new key if not present in dict1
            origin_dict[key] = value

    return origin_dict


def calc_circle_segment(start_angle=0, end_angle=360, radius=1):
    x_s = []
    y_s = []
    for i in range(start_angle, end_angle):
        x = radius * math.cos(math.radians(i))
        y = radius * math.sin(math.radians(i))
        x_s.append(x)
        y_s.append(y)

    return np.array(x_s), np.array(y_s)


def set_torch_multiprocess_start_method():
    if os.environ['START_METHOD'] == 'SPAWN':
        import torch.multiprocessing as mp

        mp.set_start_method('spawn', force=True)

    # if 'START_METHOD' in os.environ:
    #     import torch.multiprocessing as mp
    #
    #     if os.environ['START_METHOD'] == 'SPAWN':
    #         mp.set_start_method('spawn', force=True)
    #
    #     if os.environ['START_METHOD'] == 'FORK':
    #         mp.set_start_method('fork', force=True)
    #
    #     start_method = mp.get_start_method()
    #
    #     printmd(f"**START METHOD:** ***{str(start_method).upper()}***")


def set_multiprocess_start_method():
    if 'START_METHOD' in os.environ:
        import multiprocessing as mp

        if os.environ['START_METHOD'] == 'SPAWN':
            mp.set_start_method('spawn', force=True)

        if os.environ['START_METHOD'] == 'FORK':
            mp.set_start_method('fork', force=True)

        start_method = mp.get_start_method()

        printmd(f"**START METHOD:** ***{str(start_method).upper()}***")


def change_multiprocessing_start_method_delegate(start_method):
    import multiprocessing as mp

    original_start_method = mp.get_start_method(allow_none=True)
    mp.set_start_method(start_method, force=True)
    target_start_method = mp.get_start_method(allow_none=True)
    printmd(f"CHANGE MP Start method [ORIGINAL]: ***{original_start_method}*** >>> [TARGET]: ***{target_start_method}***")

    def delegate():
        nonlocal original_start_method
        nonlocal target_start_method
        mp.set_start_method(original_start_method, force=True)
        original_start_method = mp.get_start_method(allow_none=True)
        printmd(f"ROLL BACK MP Start method [TARGET]: ***{target_start_method}*** >>> [ORIGINAL]: ***{original_start_method}***")

    return delegate


def _func(arg_s):
    chunk_i = arg_s['chunk_i']
    input_data = arg_s['input_data']
    _lambda = arg_s['_lambda']

    output_data = _lambda(input_data)

    return {'chunk_i': chunk_i, 'output_data': output_data}


def list_concurrent_map(input_data_s, _lambda, cpu_count):
    split_chunk_size = int(len(input_data_s) / (cpu_count - 1)) if cpu_count > 1 else cpu_count
    input_data_chunk_s = split_list_into_chunks(input_data_s, split_chunk_size)

    arg_s = [{'chunk_i': i, 'input_data': input_data_chunk_s[i], '_lambda': _lambda} for i in range(len(input_data_chunk_s))]
    result_chunk_s = func_multi_process(_func, arg_s, num_workers=cpu_count, print_result_full=False)
    result_chunk_s_sorted = sorted(result_chunk_s, key=lambda x: x['chunk_i'])
    output_data_chunk_s = [chunk['output_data'] for chunk in result_chunk_s_sorted]
    output_data_s = list(itertools.chain(*output_data_chunk_s))

    return output_data_s


def generate_random_one_hot__s(classes_count, samples_count, dtype=None):
    if dtype is None:
        one_hot_array = np.zeros((samples_count, classes_count))
        random_classes = np.random.randint(0, classes_count, size=samples_count)
    else:
        one_hot_array = np.zeros((samples_count, classes_count), dtype=dtype)
        random_classes = np.random.randint(0, classes_count, size=samples_count, dtype=dtype)

    one_hot_array[np.arange(samples_count), random_classes] = 1

    return one_hot_array


def generate_random_clazz__s(classes_count, samples_count, dtype=None):
    if dtype is None:
        random_clazz_s = np.random.randint(0, classes_count, size=samples_count)
    else:
        random_clazz_s = np.random.randint(0, classes_count, size=samples_count, dtype=dtype)

    return random_clazz_s


def generate_random_one_hot_prob__s(classes_count, samples_count, dtype=None):
    if dtype is None:
        random_clazz_s = np.random.random((samples_count, classes_count))
    else:
        random_clazz_s = np.random.random((samples_count, classes_count)).astype(dtype)

    return random_clazz_s


def calc_memory_size(obj):
    """Recursively calculate the memory size of an object in bytes."""
    # If it's a list, we need to sum the sizes of its elements
    if isinstance(obj, list):
        total_size = sys.getsizeof(obj)  # Size of the list object itself
        for item in obj:
            total_size += calc_memory_size(item)  # Add the size of each item
        return total_size

    # If it's a tuple, we treat it similarly to a list
    elif isinstance(obj, tuple):
        total_size = sys.getsizeof(obj)
        for item in obj:
            total_size += calc_memory_size(item)
        return total_size

    # If it's a dict, we calculate the size of keys and values
    elif isinstance(obj, dict):
        total_size = sys.getsizeof(obj)
        for key, value in obj.items():
            total_size += calc_memory_size(key)
            total_size += calc_memory_size(value)
        return total_size

    # For all other types, we return the size using sys.getsizeof
    else:
        return sys.getsizeof(obj)


def run_test_single_net(_net_producer):
    import torch

    torch.cuda.empty_cache()

    from SRC.LIBRARIES.new_data_utils import produce_dummy_dataset_factory

    config_suffix = 'DUMMY_stage4'
    os.environ[_CONFIGS_SUFFIX] = config_suffix

    dummy_dataset_factory = produce_dummy_dataset_factory(config_suffix)
    net = _net_producer()

    test_single(net, dummy_dataset_factory, config_suffix)

    torch.cuda.empty_cache()


def test_single(net, _produce_dummy_dataset, config_suffix):
    import os
    import sys

    import torch
    from torch import nn
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.data import DataLoader

    from SRC.CORE._CONSTANTS import USE_GPU, _USE_GPU_DATA_PARALLEL, MODEL_FOLDER_PATH
    from SRC.CORE.debug_utils import printmd
    from SRC.LIBRARIES.new_data_utils import initialize_empty_net_cpu, produce_dummy_dataset_factory
    from SRC.NN.IModelBase import IModelBase
    from SRC.NN.BaseDiscreateNN import BaseDiscreateNN

    net_cpu_empty = net

    net = initialize_empty_net_cpu(net, _produce_dummy_dataset)

    dummy_dataset_factory = produce_dummy_dataset_factory(config_suffix)

    if torch.cuda.is_available() and USE_GPU():
        net = net.cuda()

        if torch.cuda.device_count() > 1 and check_env_true(_USE_GPU_DATA_PARALLEL):
            printmd(f"**GPU DATA PARALLEL ENABLED** | GPU count: ***{torch.cuda.device_count()}***")
            net = nn.DataParallel(net)

    samples_count = 32
    batch_size = 100
    lr = 0.001
    eta_min = lr / 10
    t_max = 10

    def data_loader_batched(title):
        data_loader_batched = DataLoader(dataset=dummy_dataset_factory(net_cpu_empty, samples_count), batch_size=batch_size)

        for j, (input_array, meta_array, output_array) in enumerate(data_loader_batched):
            yield input_array, meta_array, output_array

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)

    performance_training = net.perform_training('BATCHED TRAIN SIMULATION', data_loader_batched, net, optimizer, scheduler)
    performance_evaluation = net.perform_evaluation(data_loader_batched, net)

    print(f"TRAIN PERFORMANCE: {filter_dict(performance_training, filter_field_s=['mean_loss'], exclude=False)}")
    print(f"EVALUATION PERFORMANCE: {filter_dict(performance_evaluation, filter_field_s=['mean_loss'], exclude=False)}")

    suffix = 'killme'
    train_suffix = suffix
    model_suffix = config_suffix
    model_name_suffix = f"{net_cpu_empty.__class__.__module__.split('.')[-1].replace('__', '')}__{train_suffix}-{str(model_suffix)}-EP0"

    write_model(net.cpu(), model_name_suffix=model_name_suffix)
    net_inference = read_net_inference(model_name_suffix)

    IModelBase.test_single_inference(net_inference)

    print(f'!!!TEST {model_name_suffix} PASSED!!!')
    os.remove(f'{MODEL_FOLDER_PATH()}/{model_name_suffix}.pt')
    sys.stdout.flush()


def run_resource_monitor(interval_secs=60):
    import time
    from SRC.LIBRARIES.time_utils import kiev_now
    from SRC.CORE.debug_utils import printmd, set_parent_process

    set_parent_process()

    while True:
        print(f'---------------------------------------{kiev_now()}-----------------------------------------------------')
        resources_usage_format = produce_resource_usage_format()
        resources_usage_present = resources_usage_format()
        printmd(resources_usage_present)

        time.sleep(interval_secs)


def run_notebook_cell(cell_num, name=None, rel_path='NOTEBOOKS'):
    from SRC.CORE._CONSTANTS import project_root_dir
    import nbformat
    from IPython import get_ipython

    if name is None:
        notebook_name = get_current_notebook_name()
        name = notebook_name.split("_")[0]

    if rel_path is None:
        full_path = f'{project_root_dir()}/SRC'
    else:
        full_path = f'{project_root_dir()}/SRC/{rel_path}'

    notebook_path = f'{full_path}/{name}.ipynb'
    print(f"Running: [{cell_num}] {notebook_path}")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    cell_to_run = nb.cells[cell_num-1]

    if cell_to_run.cell_type == 'code':
        ipython = get_ipython()
        result = ipython.run_cell(cell_to_run.source)

        # 🧠 result.error_before_exec and result.error_in_exec hold exceptions
        if result.error_in_exec is not None:
            raise result.error_in_exec
        elif result.error_before_exec is not None:
            raise result.error_before_exec


def format_num(num):
    return format(num, ',').replace(',', ' ')


def normalize(value, x_min, x_max, new_min, new_max):
    normalized_value = ((value - x_min) / (x_max - x_min)) * (new_max - new_min) + new_min

    return normalized_value


def produce_display_handler(_wrap_producer):
    display_handle: DisplayHandle = None
    last_exception_presented = False

    def display(text):
        nonlocal display_handle
        nonlocal last_exception_presented

        current_exception = 'ex' in text.lower() or 'broken' in text.lower() or 'failed' in text.lower() or 'error' in text.lower()
        is_new_line = False
        if not current_exception and last_exception_presented:
            is_new_line = True

        if current_exception:
            is_new_line = True

        if display_handle is None:
            is_new_line = True

        if is_new_line:
            display_handle = DisplayHandle()
            display_handle.display(_wrap_producer(text))
        else:
            display_handle.update(_wrap_producer(text))

        last_exception_presented = current_exception

    return display


def produce_display_handler_print():
    return lambda text: print(text)


def produce_display_handler_HTML():
    from IPython.display import HTML

    return produce_display_handler(lambda text: HTML(text))


def produce_display_handler_MARKDOWN():
    # print(f"is_running_in_notebook(): {is_running_in_notebook()}")
    if is_running_in_notebook():
        from IPython.display import Markdown
        return produce_display_handler(lambda text: Markdown(text))
    else:
        return produce_display_handler_print()


def produce_display_handler_IMAGE():
    from IPython.display import Image

    return produce_display_handler(lambda img_path: Image(img_path))


def produce_progress_display_handler_MARKDOWN():
    display_handler = produce_display_handler_MARKDOWN()
    counter = 0

    def iterate(text=None):
        nonlocal counter

        if counter == 0:
            title_dot = ''
        else:
            if counter % 4 == 0:
                counter = 0

            title_dot = f"**{populate_char_n_times('.', counter)}** " if counter > 0 else ''

        final_title = f"{title_dot}{text}"
        # printmd(final_title)
        display_handler(final_title)

        counter += 1

    return iterate


def run_safety_interrupter(title, count_down_secs=5):
    from SRC.LIBRARIES.time_utils import kiev_now

    counter = count_down_secs
    display_handler = produce_display_handler_MARKDOWN()
    _bi, _b, nl_ = produce_formatters(mode=None)

    while True:
        if not is_cloud():
            display_handler = produce_display_handler_MARKDOWN()

        display_handler(f"{title} in: {_b(int(counter))} secs.. | {datetime_h_m_s(kiev_now())}")

        if counter <= 1:
            time.sleep(1)
            display_handler(f"")

            break

        counter -= 1

        time.sleep(1)


def color_with_opacity(color_name, opacity):
    import matplotlib.colors as mcolors

    rgba = mcolors.to_rgba(color_name, opacity)

    return f"rgba({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)}, {rgba[3]})"


def produce_enter_condition_present(_condition, current_price, target_price=None):
    enter_condition_present = inspect.getsource(_condition).split(":")[1].strip()
    if target_price is not None:
        enter_condition_present = enter_condition_present.replace('target_price', f"{str(target_price)}")

    enter_condition_present = enter_condition_present.replace('price', f"***{str(current_price)}***")

    return enter_condition_present


def produce_symbol_name_from_manual_trading_notebook():
    notebook_name = get_current_notebook_name()
    symbol = notebook_name.replace("_", "")
    if 'Copy' in symbol:
        symbol = symbol.split("-")[0]

    return symbol


def get_input_feature_col_s(df: pd.DataFrame):
    all_col_s = list(df.columns)
    input_col_s = [col for col in all_col_s if '.' not in col]

    return input_col_s


def get_threshold_col_s(df: pd.DataFrame, threshold=None):
    all_col_s = list(df.columns)
    all_threshold_col_s = [col for col in all_col_s if '.' in col]

    if threshold is None:
        return all_threshold_col_s
    else:
        threshold_col_s = [col for col in all_threshold_col_s if str(threshold) == col.split("_")[-1]]

        return threshold_col_s


def parse_string_variables(input_string, key_s):
    import re

    result = {}
    for key in key_s:
        # Match everything after key until the next double underscore or end of string
        pattern = rf'{re.escape(key)}([^\_]+(?:_[^\_]+)*)'
        match = re.search(pattern, input_string)
        if match:
            result[key] = match.group(1)
    return result


PRICE_COMPARE_RELATIVE_TOLERANCE = 0.0035


def is_close_or_lower(current_value, target_value, rtol=PRICE_COMPARE_RELATIVE_TOLERANCE):
    return np.isclose(current_value, target_value, rtol=rtol, atol=0) or current_value < target_value


def is_close_or_higher(current_value, target_value, rtol=PRICE_COMPARE_RELATIVE_TOLERANCE):
    return np.isclose(current_value, target_value, rtol=rtol, atol=0) or current_value > target_value


def _L(current_value, target_value, rtol=PRICE_COMPARE_RELATIVE_TOLERANCE):
    return is_close_or_lower(current_value, target_value, rtol=rtol)


def _H(current_value, target_value, rtol=PRICE_COMPARE_RELATIVE_TOLERANCE):
    return is_close_or_higher(current_value, target_value, rtol=rtol)


def timed_cache(ttl: int):
    from time import time

    def wrapper(func):
        # Add a cache and timestamp attribute to the function
        func = lru_cache(maxsize=None)(func)
        _cache_timestamp = {}

        def wrapped_func(*args, **kwargs):
            nonlocal _cache_timestamp

            key = str(args + tuple(kwargs.items()))

            now = time()
            # Check if the cache has expired

            if key in _cache_timestamp:
                if now - _cache_timestamp[key] > ttl:
                    cache_clear(*args, **kwargs)
            else:
                cache_clear(*args, **kwargs)

            return func(*args, **kwargs)

        def cache_clear(*args, **kwargs):
            nonlocal _cache_timestamp

            key = str(args + tuple(kwargs.items()))
            now = time()

            func.cache_clear()
            _cache_timestamp[key] = now

        wrapped_func.cache_clear = cache_clear

        return wrapped_func

    return wrapper


def json_file_cache(filename: str, ttl: int = math.inf, lock_timeout_secs=5, hash_key: bool=False):
    """
    A decorator that caches function results in a local JSON file with a TTL.
    If the key exists in the file and the file's last modified time is within the TTL,
    it returns the cached result. Otherwise, it refreshes the cache.

    :param filename: Path to the JSON file used for caching.
    :param ttl: Time-to-live for the cache in seconds.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):

            if hash_key:
                key = hashlib.md5(str(args[0]).encode()).hexdigest()

                # msg = f"KILLME: [{filename}|{key}|{len(args[0])}]"
                # NOTICE(msg)
                # append_file(f'{filename}'.replace('json', 'txt'), msg)
            else:
                key = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True)

            current_time = time.time()

            lock = FileLock(filename.replace('json', 'lock'), timeout=lock_timeout_secs)
            with lock:
                cache = read_json_safe(filename, {})

                if key in cache and current_time - cache[key]['last_modified'] < ttl:
                    return cache[key]['data']

                result = func(*args, **kwargs)

                current_time = time.time()

                cache[key] = {
                    'last_modified': current_time,
                    'data': result
                }

                write_json(cache, filename)

            return result

        return wrapper
    return decorator


def parametric_lru_cache(hash_func, maxsize=None):
    def decorator(func):
        last_args = None
        last_kwargs = None

        @lru_cache(maxsize=maxsize)
        def _cache(stringable):
            nonlocal last_args
            nonlocal last_kwargs

            key = hashlib.md5(str(stringable).encode()).hexdigest()

            print(f"PARAMETRIC LRU CACHE KEY [{kiev_now_formatted()}|{key}|{os.getpid()}|{threading.currentThread().ident}]:\r\n{stringable}")

            cached_result = func(*last_args, **last_kwargs)

            return cached_result

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal last_args, last_kwargs
            last_args = args
            last_kwargs = kwargs
            hash_key = hash_func(*args, **kwargs)

            result = _cache(hash_key)

            return result

        return wrapper
    return decorator


def parametric_json_file_cache(hash_func, filename: str, ttl: int = math.inf, lock_timeout_secs=5, clear_on_restart=False):
    if clear_on_restart:
        dir_name = os.path.dirname(filename)
        if dir_name == '':
            json_file_path = f"{project_root_dir()}/{filename}"
        else:
            json_file_path = filename

        if os.path.exists(json_file_path):
            os.remove(json_file_path)
            CONSOLE(f"REMOVED PARAMETRIC JSON FILE CACHE: {json_file_path}")

        lock_file_path = json_file_path.replace('json', 'lock')
        if os.path.exists(lock_file_path):
            os.remove(lock_file_path)

        txt_file_path = json_file_path.replace('json', 'txt')
        if os.path.exists(txt_file_path):
            os.remove(txt_file_path)

    def decorator(func):
        last_args = None
        last_kwargs = None

        @json_file_cache(filename=filename, ttl=ttl, lock_timeout_secs=lock_timeout_secs, hash_key=True)
        def _cache(stringable):
            nonlocal last_args
            nonlocal last_kwargs

            key = hashlib.md5(str(stringable).encode()).hexdigest()

            msgs = [
                f"PARAMETRIC JSON FILE CACHE KEY [{kiev_now_formatted()}|{filename}|{key}|{len(stringable)}|||{os.getpid()}|{threading.currentThread().ident}]:"
            ]

            if IS_DEBUG():
                msgs.append(str(stringable))
                DEBUG_SPLITTED(*msgs)
            else:
                NOTICE(*msgs)

            append_file(f'{filename}'.replace('json', 'txt'), *msgs)

            cached_result = func(*last_args, **last_kwargs)

            return cached_result

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal last_args, last_kwargs
            last_args = args
            last_kwargs = kwargs
            stringable = hash_func(*args, **kwargs)

            result = _cache(stringable)

            return result

        return wrapper
    return decorator


def multiprocess_cache(cache_path, ttl: int = None):
    cache = dc.Cache(cache_path)
    """
    Multiprocess-safe caching decorator using diskcache.

    :param expire: TTL in seconds for the cached entry (None = never expires)
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Build a cache key based on arguments
            key = f"{func.__name__}:{str(args)}:{str(kwargs)}"

            # Atomic read–compute–write to prevent race conditions
            with cache.transact():
                value = cache.get(key, default=None)
                if value is not None:
                    return value

                # Compute the result
                result = func(*args, **kwargs)

                # Store in cache with optional expiration
                cache.set(key, result, expire=ttl)
                return result

        return wrapper

    return decorator


def lock_with_file(lockfile='function.lock', timeout=10):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            lock = FileLock(lockfile, timeout=timeout)
            try:
                with lock:
                    DEBUG(f"Acquired lock on {lockfile}")
                    result = func(*args, **kwargs)
                    return result
            except Timeout:
                DEBUG(f"Could not acquire lock on {lockfile} within {timeout} seconds")
                raise
            finally:
                DEBUG(f"Released lock on {lockfile}")
        return wrapper
    return decorator


def enumerate_list_s(*list_s):
    assert all(len(sublist) == len(list_s[0]) for sublist in list_s), "Not all sublists have the same size"

    for idx, item in enumerate(list_s[0]):
        item_s = [item]
        for list_n in list_s[1:]:
            item_i = list_n[idx]
            item_s.append(item_i)

        yield idx, item_s


@lru_cache(maxsize=None)
def produce_state_singleton(lock_name=None):
    return produce_state(lock_name=lock_name)


def produce_state(initial_state={}, lock_name=None):
    if lock_name is not None:
        state_buffer = MPDeque(maxlen=1)

        out_trade_lock_path = f"{project_root_dir()}/locks/{lock_name}.lock"
        lock = FileLock(out_trade_lock_path)

        def set_state(state):
            with lock:
                old_state = get_state()
                new_state = merge_dicts(old_state, state)
                state_buffer.append(new_state)

        def get_state():
            with lock:
                if len(state_buffer) > 0:
                    return state_buffer[-1]
                return {}
    else:
        state_buffer = deque(maxlen=1)

        def set_state(state):
            old_state = get_state()
            new_state = merge_dicts(old_state, state)
            state_buffer.append(new_state)

        def get_state():
            if len(state_buffer) > 0:
                return state_buffer[-1]
            return {}

    set_state(initial_state)

    return set_state, get_state


def __uuid4_12():
    id = str(uuid.uuid4())

    return id[-12:]


def delayed_call(_lambda, ttw=3):
    threading.Timer(ttw, lambda data: _lambda(), args=[{}]).start()


def normalize_for_cnn(x, _mean, _std):
    """
    Nonlinear normalization to [-1, 1] using tanh.
    Works well for time series or CNN input features.

    Parameters
    ----------
    x : array-like
        Input data.
    mean : float
        Global mean of the feature (from training data).
    std : float
        Global standard deviation of the feature (from training data).

    Returns
    -------
    z : np.ndarray
        Nonlinearly normalized array in (-1, 1).
    """
    x = np.asarray(x)
    z = (x - _mean) / _std          # standardize
    normalized = np.tanh(z)

    return normalized              # squash to (-1, 1)


def denormalize_from_cnn(z, _mean, _std):
    """
    Reverse of nonlinear normalization using arctanh.

    Parameters
    ----------
    z : array-like
        Normalized data in (-1, 1).
    mean : float
        Same global mean used during normalization.
    std : float
        Same global std used during normalization.

    Returns
    -------
    x : np.ndarray
        Original scale array.
    """
    z = np.asarray(z)
    z = np.clip(z, -0.999999, 0.999999)  # prevent numerical overflow
    denormalized = np.arctanh(z) * _std + _mean

    return denormalized


def ENV_INT_LESS_THAN(env_key, default):
    if env_key in os.environ:
        env_int_val = int(os.environ[env_key])
        return env_int_val if is_cloud() else default if env_int_val > default else env_int_val
    else:
        return default


def remove_web_app_out_caches(ext_log_s=None):
    with log_context(CONSOLE_SPLITTED) as log_s:
        backtesting_out_folder = f"{project_root_dir()}/OUT/BACKTESTING"
        backtesting_net_folder_s = {
            f for f in os.listdir(backtesting_out_folder)
            if os.path.isdir(os.path.join(backtesting_out_folder, f))
        }

        autotrading_out_folder = f"{project_root_dir()}/OUT/AUTOTRADING"
        autotrading_net_folder_s = {
            f for f in os.listdir(autotrading_out_folder)
            if os.path.isdir(os.path.join(backtesting_out_folder, f))
        }

        web_app_out_cache_html_folder_path = f"{project_root_dir()}/OUT/WEBAPP/dashboard/html"
        web_app_out_cache_img_folder_path = f"{project_root_dir()}/OUT/WEBAPP/dashboard/img"

        for target_dir in [web_app_out_cache_html_folder_path, web_app_out_cache_img_folder_path]:
            for filename in os.listdir(target_dir):
                target_path = os.path.join(target_dir, filename)
                filename_without_ext = os.path.splitext(filename)[0]
                if filename_without_ext not in [*backtesting_net_folder_s, *autotrading_net_folder_s]:
                    if os.path.isfile(target_path):
                        os.remove(target_path)
                        ext_log_s.append(f"REMOVED WEB APP CACHE FILE: {target_path}")


def TEST__MERGE_DICTS():
    origin_dict = {
        'name': 'Jhon',
        'options': {
            'age': 12,
            'height': 178,
        },
        'relatives': [
            {
                'name': 'Dave',
                'relatives': []
            },
            {
                'name': 'Megan',
                'relatives': []
            }
        ]
    }

    modified_dict = {
        'name': 'Jeniffer',
        'options': {
            'age': 18
        },
        'relatives': [
            {
                'name': 'Biden'
            }
        ]
    }

    print(merge_dicts(origin_dict, modified_dict))


def TEST__PARAMETRIC_LRU_CACHE():
    @parametric_lru_cache(lambda d: "|".join([i['key1'] for i in d]), maxsize=1)
    def calculate_data(data):
        result = [*(x for d in data for x in d['key3'])]

        key = "|".join([i['key1'] for i in data])
        print(f'CALCULATED [{key}]: {result}')

        return result

    data = [
        {
            'key1': 'val1',
            'key2': 'val2',
            'key3': [0, 1, 2, 3, 4, 5, 6]
        }, {
            'key1': 'val3',
            'key2': 'val4',
            'key3': [0, 1, 2, 3, 4, 5, 6]
        },
    ]

    calculate_data(data)

    data[0]['key1'] = 'adasdasd'
    calculate_data(data)

    data[0]['key1'] = 'adasdasasdasdasdda'
    calculate_data(data)

    data[0]['key1'] = 'adasdasd'
    calculate_data(data)


def TEST__TIME_LRU_CACHE():
    import time

    @timed_cache(ttl=1)
    def expensive_computation(x):
        print(f"Computing {x}...")
        return x * 2

    print(expensive_computation(2))
    time.sleep(0.5)
    print(expensive_computation(2))

    print("-------")

    print(expensive_computation(2))
    time.sleep(2)
    print(expensive_computation(2))


def TEST__JSON_LRU_CACHE():
    @json_file_cache(f"{project_root_dir()}/killme_cache.json", ttl=5)  # Cache valid for 300 seconds (5 minutes)
    def expensive_computation(x, y):
        print(f"Computing {x} + {y}...")
        return {'x': x, 'y': y, 'z': x + y}

    # Testing the cache
    print(expensive_computation(2, 3))  # Computes and caches
    print(expensive_computation(2, 3))  # Returns from cache
    print(expensive_computation(2, 3))  # Returns from cache

    time.sleep(7)

    # Testing the cache
    print(expensive_computation(2, 3))  # Computes and caches
    print(expensive_computation(2, 3))  # Returns from cache
    print(expensive_computation(2, 3))  # Returns from cache

    # Testing the cache
    print(expensive_computation(1, 4))  # Computes and caches
    print(expensive_computation(1, 4))  # Returns from cache
    print(expensive_computation(1, 4))  # Returns from cache

    time.sleep(7)

    # Testing the cache
    print(expensive_computation(1, 4))  # Computes and caches
    print(expensive_computation(1, 4))  # Returns from cache
    print(expensive_computation(1, 4))  # Returns from cache


def TEST__FUNCTION_LOCK_DECORATOR():
    # Example usage:
    @lock_with_file(lockfile='my_function.lock', timeout=5)
    def my_function():
        import time
        print("Executing function...")
        time.sleep(3)
        print("Function execution finished.")

    my_function()


def TEST_killme_random_prediction():
    from SRC.WEBAPP._PREDICTION.PredictionServiceApp import killme_random_prediction

    orig_prediction = {'entry_clazz_s': [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12], 'pred_clazz': 7, 'pred_label': '0.00350 | 0.00847', 'signal': 'IGNORE', 'take_profit_ratio': 0}
    prediction = killme_random_prediction(random_setup="1.0055-1.01")
    print(prediction)


def TEST__MULTIPROCESS_CACHE():
    from multiprocessing import Process

    for i in range(5):
        process1 = Process(target=TEST_killme_random_prediction, args=())
        process1.start()
        process2 = Process(target=TEST_killme_random_prediction, args=())
        process2.start()

        # check_killme_random_prediction()
        # check_killme_random_prediction()

        print('===========================================================================================================')
        print('===========================================================================================================')
        time.sleep(12)


if __name__ == "__main__":
    TEST__PARAMETRIC_LRU_CACHE()
    # TEST__MERGE_DICTS()
    # TEST__TIME_LRU_CACHE()
    # TEST__JSON_LRU_CACHE()
    # TEST__FUNCTION_LOCK_DECORATOR()
    # TEST__MULTIPROCESS_CACHE()