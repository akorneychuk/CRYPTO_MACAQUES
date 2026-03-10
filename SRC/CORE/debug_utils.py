import argparse
import functools
import gc
import inspect
import logging
import os
import socket
import subprocess
import sys
import time
import traceback
from collections import OrderedDict, deque
from contextlib import contextmanager
from datetime import datetime
from datetime import timedelta
from datetime import timezone as tz
from functools import lru_cache
from inspect import getframeinfo, stack
from pathlib import Path
from threading import Thread
from importlib.metadata import version

import psutil
import pytz
from IPython.display import Markdown, display, HTML
from filelock import FileLock

from SRC.CORE._CONSTANTS import TIME_ZONE_HOURS_OFFSET, IS_MEMORY_MONITOR_ENABLED, NOTEBOOK_NAME_KEY, MEMORY_MONITOR_UPDATE_INTERVAL_MINS, \
    CPU_COUNT, GPU_COUNT, PARENT_PROCESS_ID_KEY, PYTHON_STARTING_DIR_KEY, string_bool, USE_GPU, _LOGGING_PROCESS_LOCK_FILE_PATH, _AUTOMATION_TYPE, KIEV_TZ, _USE_PROXY_CLIENT
from SRC.CORE._CONSTANTS import project_root_dir, _SYMBOL_JOIN, _SYMBOL_SLASH, _SYMBOL_DASH, _COIN_ASSET, _STABLE_ASSET, _CALLERS_APP_HOST, _IS_REDIS_CLOUD, _IS_BINANCE_PROD, _IS_CLOUD, _PREDICTION_SERVICE_APP_HOST, \
    _RESOURCES_FORMAT_JUPYTER, _RESOURCES_FORMAT_PLOTLY, _AUTOTRADING

LOG_LEVEL = 'LOG_LEVEL'
LOG_LEVEL_HIGH = '3'
LOG_LEVEL_MEDIUM = '2'
LOG_LEVEL_LOW = '1'
LOG_LEVEL_NONE = '0'

logging.EXCEPTION = 45
logging.CONSOLE = 25
logging.NOTICE = 15


def produce_measure_high(title_start=None, print_on_start=False):
    return produce_measure(title_start, LOG_LEVEL_HIGH, print_on_start=print_on_start)


def produce_measure_medium(title_start=None, print_on_start=False):
    return produce_measure(title_start, LOG_LEVEL_MEDIUM, print_on_start=print_on_start)


def produce_measure_low(title_start=None, print_on_start=False):
    return produce_measure(title_start, LOG_LEVEL_LOW, print_on_start=print_on_start)


def produce_measure(title_start=None, log_level=LOG_LEVEL_NONE, print_on_start=False):
    return _produce_measure(print_on, title_start, log_level, print_on_start=print_on_start)


def produce_measure_code_frame():
    file_name_start, line_number_start = get_caller_file_line(2)
    measure = produce_measure(title_start=f"{file_name_start.split('/')[-1]}-{line_number_start}")

    def measure_code_frame(print_out=True):
        file_name_end, line_number_end = get_caller_file_line(2)

        duration = measure(title_end=f"{file_name_end.split('/')[-1]}-{line_number_end}", print_out=print_out)
        sys.stdout.flush()

        return duration

    return measure_code_frame


def produce_measure_md_high(title_start=None):
    return produce_measure_md(title_start, LOG_LEVEL_HIGH)


def produce_measure_md_medium(title_start=None):
    return produce_measure_md(title_start, LOG_LEVEL_MEDIUM)


def produce_measure_md_low(title_start=None):
    return produce_measure_md(title_start, LOG_LEVEL_LOW)


def produce_measure_md(title_start=None, log_level=LOG_LEVEL_MEDIUM):
    return _produce_measure(printmd_on, title_start, log_level)


def _produce_measure(print_on_func, title_start=None, log_level=LOG_LEVEL_MEDIUM, print_on_start=False):
    if print_on_start:
        print_on_func(f'{title_start}', log_level)

    start_time = datetime.now()

    def measure(title_end=None, print_out=False):
        end_time = datetime.now()

        time_diff = end_time - start_time

        if title_start is None and title_end is None:
            if print_out:
                print_on_func(f'duration = {time_diff}', log_level)

            return time_diff

        if title_start is None:
            if print_out:
                print_on_func(f'{title_end} || duration = {time_diff}', log_level)

            return time_diff

        if title_end is None:
            if print_out:
                if not print_on_start:
                    print_on_func(f'{title_start} || duration = {time_diff}', log_level)
                else:
                    print_on_func(f'duration = {time_diff}', log_level)

            return time_diff

        if print_out:
            if not print_on_start:
                print_on_func(f'{title_start} || {title_end} || duration = {time_diff}', log_level)
            else:
                print_on_func(f'{title_end} || duration = {time_diff}', log_level)

        return time_diff

    return measure


def is_log_level(log_level):
    is_level = (LOG_LEVEL in os.environ and int(os.environ[LOG_LEVEL]) >= int(log_level)) or log_level == LOG_LEVEL_NONE

    return is_level


def is_high_log_level():
    is_high = is_log_level(LOG_LEVEL_HIGH)

    return is_high


def is_medium_log_level():
    is_medium = is_log_level(LOG_LEVEL_MEDIUM)

    return is_medium


def is_low_log_level():
    is_low = is_log_level(LOG_LEVEL_LOW)

    return is_low


def is_none_log_level():
    is_none = is_log_level(LOG_LEVEL_NONE)

    return is_none


def display_on(df, log_level):
    if is_log_level(log_level):
        display(df)


def display_high(df):
    if is_high_log_level():
        display(df)


def display_medium(df):
    if is_medium_log_level():
        display(df)


def display_low(df):
    if is_low_log_level():
        display(df)


def run_on(_lambda, _lambda_else=None, log_level=LOG_LEVEL_HIGH):
    if is_log_level(log_level):
        _lambda()
    else:
        if _lambda_else is not None:
            _lambda_else()


def run_high(_lambda, _lambda_else=None):
    run_on(_lambda, _lambda_else, LOG_LEVEL_HIGH)


def run_medium(_lambda, _lambda_else=None):
    run_on(_lambda, _lambda_else, LOG_LEVEL_MEDIUM)


def run_low(_lambda, _lambda_else=None):
    run_on(_lambda, _lambda_else, LOG_LEVEL_LOW)


def colorize_md_text(text, color='black'):
    return f"<span style='color:{color}'>{text}</span>"


def printmd(string, color=None):
    if is_parent_process() and is_running_in_notebook():
        # string = string.replace(' ', '&nbsp;').replace("$", "&#36;")
        if color is None:
            display(Markdown(string))
        else:
            colorstr = colorize_md_text(string, color)
            display(Markdown(colorstr))
    else:
        string = string.replace('***', '').replace('**', '').replace('`', '').replace('\r\n\r\n', '\r\n').replace("&#36;", "$")
        print(string)


def printmd_HTML(string):
    string = string.replace('"', '').replace('***', '').replace('**', '').replace('`', '').replace('\r\n\r\n', '\r\n').replace("&#36;", "$")
    display(HTML(string))


def printmd_on(string, log_level):
    if is_log_level(log_level):
        printmd(string)

    log_module(string)


def printmd_high(string):
    if is_high_log_level():
        printmd(string)

    log_module(string)


def printmd_medium(string):
    if is_medium_log_level():
        printmd(string)

    log_module(string)


def printmd_low(string):
    if is_low_log_level():
        printmd(string)

    log_module(string)


def print_on(text, log_level):
    if is_log_level(log_level):
        print(text)

    log_module(text)


def print_high(text):
    if is_high_log_level():
        print(text)

    log_module(text)


def print_medium(text):
    if is_medium_log_level():
        print(text)

    log_module(text)


def print_low(text):
    if is_low_log_level():
        print(text)

    log_module(text)


def get_log_level():
    return os.environ[LOG_LEVEL]


def set_log_level(log_level):
    os.environ[LOG_LEVEL] = log_level


def set_log_level_NONE():
    os.environ[LOG_LEVEL] = LOG_LEVEL_NONE
    printmd(f'Log level: **NONE**')


def set_log_level_LOW():
    os.environ[LOG_LEVEL] = LOG_LEVEL_LOW
    printmd(f'Log level: **LOW**')


def set_log_level_MEDIUM():
    os.environ[LOG_LEVEL] = LOG_LEVEL_MEDIUM
    printmd(f'Log level: **MEDIUM**')


def set_log_level_HIGH():
    os.environ[LOG_LEVEL] = LOG_LEVEL_HIGH
    printmd(f'Log level: **HIGH**')


def get_logger(buffer):
    i = 0
    pr = []

    def log(price):
        nonlocal i
        if i > buffer:
            print(pr)
            return False
        i += 1

        pr.append(price)

        return True

    return log


def format_memory(size_bytes):
    # Size suffixes
    suffixes = ['B', 'KB', 'MB', 'GB', 'TB']

    # Base conversion valueF
    base = 1024

    # Determine the appropriate suffix index
    suffix_index = 0
    while size_bytes >= base and suffix_index < len(suffixes) - 1:
        size_bytes /= base
        suffix_index += 1

    # Format the size with the appropriate suffix
    size_formatted = "{:.2f} {}".format(size_bytes, suffixes[suffix_index])

    return size_formatted


def format_df_memory(df):
    column_memory = df.memory_usage()
    total_memory = df.memory_usage().sum()

    return format_memory(total_memory)


last_memory_printed = datetime.now() - timedelta(minutes=1)
highest_memory = 0
def print_memory(df=None, is_memory_monitor=False):
    global last_memory_printed
    global highest_memory
    if (datetime.now() - last_memory_printed) < timedelta(seconds=10):
        return

    last_memory_printed = datetime.now()

    if is_medium_log_level() or (is_memory_monitor and is_low_log_level()):
        gc.collect()

        memory = psutil.virtual_memory()
        total_memory = memory[0]
        available_memory = memory[1]
        usage_percent_memory = memory[2]
        usage_memory = memory[3]

        now_memory_formatted = format_memory(usage_memory)
        if usage_memory > highest_memory:
            highest_memory = usage_memory
        highest_memory_formatted = format_memory(highest_memory)

        date_time_formatted = (datetime
                               .now()
                               .astimezone(tz=tz(timedelta(hours=TIME_ZONE_HOURS_OFFSET)))
                               .strftime("%H:%M:%S, %d-%m-%Y"))
        if df is not None:
            df.info(memory_usage='deep')

        memory_details_formatted = f'Use={now_memory_formatted} ({usage_percent_memory}%) | Top={highest_memory_formatted} | Free={format_memory(available_memory)} | Total={format_memory(total_memory)}'
        if is_memory_monitor:
            log_message = f"***`-- {memory_details_formatted}`*** >> `{date_time_formatted}`"
        else:
            debuginfo = debug_info()
            log_message = f"***`-- {memory_details_formatted}`*** >> `{date_time_formatted} | {debuginfo}`"

        printmd(log_message)
        printmd(" | ".join([f'{key.capitalize()}: **{format_memory(val)}**' for key, val in memory._asdict().items()]))
        log_module(log_message)


def log_finished_message(print_out=True):
    date_time_formatted = (datetime
                           .now()
                           .astimezone(tz=tz(timedelta(hours=TIME_ZONE_HOURS_OFFSET)))
                           .strftime("%H:%M:%S, %d-%m-%Y"))

    log_message = f"***`--- FINISHED ---`*** >> `{date_time_formatted}`"

    if print_out:
        printmd(log_message)

    log_module(log_message)


def finished(output_folder_path, monitor_thread=None, interrupt_notebook=True, suffix=None):
    if not interrupt_notebook:
        log_finished_message(print_out=False)
        save_notebook()

        if IS_MEMORY_MONITOR_ENABLED and monitor_thread is not None:
            monitor_thread.join()

        return

    log_finished_message()
    save_notebook()

    if IS_MEMORY_MONITOR_ENABLED and monitor_thread is not None:
        monitor_thread.join()

    if is_cloud():
        export_notebook(output_folder_path, suffix)

    if interrupt_notebook:
        log_terminated_message()
        prevent_notebook_continue_execution("NOTEBOOK EXPORTED & INTERRUPTED")


def log_terminated_message():
    terminate_msg = f"***`!!! --- TERMINATED --- !!!`***"
    printmd(terminate_msg)
    log_module(terminate_msg)
    

def prevent_notebook_continue_execution(msg=None):
    raise KeyboardInterrupt(msg if msg is not None else "INTERRUPTED")


def run_memory_monitor(interval_secs=60):
    printmd_low(f"Memory monitor: **ENABLED ({MEMORY_MONITOR_UPDATE_INTERVAL_MINS()} mins)**")
    print_memory(is_memory_monitor=True)

    def _lambda():
        global counter
        while True:
            log_file_path = get_log_file_path()
            with open(log_file_path, 'r') as file:
                logs = file.read().rstrip()
                if '--- FINISHED ---' in logs:
                    time.sleep(10)
                    return

            time.sleep(interval_secs)
            print_memory(is_memory_monitor=True)

    thread = Thread(target=_lambda)
    thread.start()

    return thread


def get_file_path_caller_lineno(stack_deep):
    try:
        caller = getframeinfo(stack()[stack_deep][0])
        directory, file_name = os.path.split(caller.filename)
        file_path = f'{directory.split(os.path.sep)[-1:][0]}/{file_name}'
        file_path_caller_lineno = f"{file_path}:{caller.lineno}"
    except:
        return ""

    return file_path_caller_lineno


def includes_string_from_array(input_string, string_array):
    for element in string_array:
        if element in input_string:
            return True
    return False


def debug_info():
    exclude = ['interactiveshell', 'ipykernel', 'debug_utils']
    file_path_caller_lineno_s = [get_file_path_caller_lineno(i) for i in range(10)]
    file_path_caller_lineno_reversed_s = reversed(file_path_caller_lineno_s)
    file_path_caller_lineno_reversed_fildered_s = filter(lambda x: not includes_string_from_array(x, exclude), file_path_caller_lineno_reversed_s)
    debug_info = ' > '.join(list(OrderedDict.fromkeys(file_path_caller_lineno_reversed_fildered_s)))

    return debug_info


def emulate_memory_management():
    import pandas as pd
    import numpy as np

    num_rows = 10 ** 8  # 1 million
    print_memory()

    # Generate random data for the DataFrame
    data = {
        'column1': np.random.randint(0, 100, size=num_rows),
        'column2': np.random.rand(num_rows),
        'column3': np.random.rand(num_rows),
        'column4': np.random.rand(num_rows),
        'column5': np.random.rand(num_rows),
        'column6': np.random.rand(num_rows),
        'column7': np.random.rand(num_rows),
    }

    # Create the DataFrame
    df = pd.DataFrame(data)

    print_memory()

    del data

    print_memory()

    df = df[['column1', 'column2']]

    print_memory()

    del df

    print_memory()


def log_module(log_message):
    try:
        log_message = str(log_message)
        log_file_path = get_log_file_path()
        with open(log_file_path, "a") as file_object:
            log_message = log_message.replace("\r\n\r\n", "\r\n")
            if '===========' in log_message:
                log_message = f"{log_message}"
            elif '>' in log_message and '..' in log_message:
                log_message = f"\r\n{log_message}"
            elif 'MEMORY' in log_message:
                log_message = f"\r\n{log_message}"
            elif log_message.isupper() or '*' in log_message:
                log_message = f'\r\n{log_message}\r\n'
            else:
                log_message = f"{log_message}\r\n"

            exclude_chars = set(['`', '*'])
            log_message = ''.join([c for c in log_message if c not in exclude_chars])

            file_object.write(log_message)
    except:
        return
        print('Unable to log..')


def clear_log():
    log_file_path = get_log_file_path()
    with open(log_file_path, "w") as file_object:
        file_object.write("")


def get_log_file_path():
    import pathlib

    log_file_name = get_notebook_log_file_name()
    directory_path = os.environ[PYTHON_STARTING_DIR_KEY]
    log_file_name = f"../{pathlib.PurePath(Path.cwd()).name}/{log_file_name}.txt"
    log_file_path = f'{directory_path}/{log_file_name}'

    if 'CORE' in log_file_path:
        raise Exception(f"SOMETHING WENT WRONG LOGGER CONFIG: {log_file_path}")

    return log_file_path

#TODO: Think about of separating logs on execution and pair level if concurrent
def separated_logs():
    import threading
    import psutil
    import time

    print(psutil.Process(os.getpid()))

    def _lambda(pair):
        for i in range(10):
            print(f'{pair} >> {threading.currentThread().ident} > {psutil.Process(os.getpid())}')
            time.sleep(1)

    threads = []
    for i in range(4):
        thread = threading.Thread(target=lambda: _lambda(f'PAIR{i}'))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()


def get_current_notebook_name():
    try:
        import ipynbname
        nb_fname = ipynbname.name()

        return nb_fname
    except Exception as ex:
        try:
            nb_fname = os.environ[NOTEBOOK_NAME_KEY]
        except KeyError:
            try:
                nb_fname = os.environ['JPY_SESSION_NAME'].split("/")[-1].split(".")[0]
            except:
                return "UNABLE TO FIND"

        return nb_fname


def get_notebook_log_file_name():
    current_notebook_name = get_current_notebook_name()
    log_file_name = f"log_{current_notebook_name.replace('crypto_', '')}"

    return log_file_name


def get_notebook_export_file_name():
    current_notebook_name = get_current_notebook_name()
    export_file_name = current_notebook_name.replace("crypto_", "")

    return export_file_name


def export_notebook(output_folder_path, suffix=None):
    from ipylab import JupyterFrontEnd
    import os

    app = JupyterFrontEnd()
    app.commands.execute('docmanager:save')
    def process_save():
        nonlocal output_folder_path

        time.sleep(5)

        output_folder_path = f"../___OUT/{output_folder_path}"
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)

        nb_name = get_current_notebook_name()
        nb_export_name = get_notebook_export_file_name()

        now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        output_file_name = f'{nb_export_name}__{suffix}__{now}' if suffix is not None else f'{nb_export_name}__{now}'

        print(f"{output_file_name} {nb_name}.ipynb")
        export_command = f"jupyter nbconvert --WebPDFExporter.paginate=False --to=html --allow-chromium-download --output {output_folder_path}/{output_file_name} {nb_name}.ipynb"
        print(export_command)
        os.system(export_command)

    process_save()


def produce_formatters(mode=_RESOURCES_FORMAT_JUPYTER):
    if mode == _RESOURCES_FORMAT_JUPYTER:
        _bi = lambda text: f'***{text}***'
        _b = lambda text: f'**{text}**'
        nl_ = '\r\n\r\n'
    elif mode == _RESOURCES_FORMAT_PLOTLY:
        _bi = lambda text: f'<b><i>{text}</i></b>'
        _b = lambda text: f'<b>{text}</b>'
        nl_ = '<br>'
    elif mode is None:
        _bi = lambda text: text
        _b = lambda text: text
        nl_ = ''
    else:
        raise RuntimeError(f"mode must be {_RESOURCES_FORMAT_JUPYTER} or {_RESOURCES_FORMAT_PLOTLY}")

    return _bi, _b, nl_


def get_processing_device(print_out=False, mode=_RESOURCES_FORMAT_JUPYTER):
    import os
    import torch

    _bi, _b, nl_ = produce_formatters(mode)

    host = 'CLOUD' if is_cloud() else 'MAC'

    memory = psutil.virtual_memory()
    total_memory_formatted = format_memory(memory[0])

    if torch.cuda.is_available() and USE_GPU():
        device = torch.device('cuda')
        device_name_formatted = f"{_b(host)}: {_bi(device.type.upper())} | CPU: {_bi(CPU_COUNT())} | GPU: {_bi(GPU_COUNT())} | RAM: {_bi(total_memory_formatted)}"
    elif 'COLAB_TPU_ADDR' in os.environ:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        device_name_formatted = f"{_b(host)}: {_bi(device.type.upper())} | CPU: {_bi(CPU_COUNT())} | TPU: {_bi(os.environ['COLAB_TPU_ADDR'])} | RAM: {_bi(total_memory_formatted)}"
    else:
        device = torch.device('cpu')
        device_name_formatted = f"{_b(host)}: {_bi(device.type.upper())} | CPU: {_bi(CPU_COUNT())} | RAM: {_bi(total_memory_formatted)}"

    if print_out:
        printmd(device_name_formatted)

    return device, device_name_formatted


def get_local_ip(print_out=False):
    import socket
    try:
        local_ip = socket.gethostbyname(socket.gethostname())
    except Exception as ex:
        print(f'Exception: {ex}')
        traceback.print_exc()

        local_ip = socket.gethostbyname("")

    if print_out:
        printmd(f"Local IP: **{local_ip}**")

    return local_ip


def get_local_ip(print_out=False):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Doesn't actually send data, just used to get the local IP address
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"  # fallback localhost
    finally:
        s.close()

    if print_out:
        printmd(f"Local IP: **{ip}**")

    return ip


def get_next_available_port():
    import socket
    for port in [8050, 8051, 8052, 8053, 8054, 8055, 8056, 8057, 8058, 8059]:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('0.0.0.0', port))
        if result == 0:
            continue
        else:
            return port

        sock.close()


def produce_dash_config(dash_in_port=8050):
    # dash_app.run(jupyter_mode="inline", port=8051, host='10.1.102.44')#TODO: SOLVE EMPTY SCREEN IN CLOUD

    if is_cloud():
        # mode = 'external'
        mode = 'inline'
        dash_in_ip = "0.0.0.0"
        # mode = 'external'
        # mode = 'jupyterlab'
        # dash_in_ip = "10.1.102.44"
        # dash_in_ip = "91.236.201.223"
        # dash_in_port=31147
    else:
        mode = 'inline'
        dash_in_ip = 'localhost'

    dash_config = dict(
        MODE=mode,
        DASH_IN_IP=dash_in_ip,
        DASH_IN_PORT=dash_in_port,
    )

    return dash_config


def produce_dash_config_trade_simulate(dash_in_port=8050):
    from IPython.display import display, HTML
    from SRC.CORE.server_setup import print_public_url_for_port

    if is_cloud():
        print_public_url_for_port(dash_in_port)
        mode = 'external'
        dash_in_ip = "0.0.0.0"
    else:
        mode = 'external'
        dash_in_ip = 'localhost'
        url = f"http://{dash_in_ip}:{dash_in_port}"

        link_html = f'<a href="{url}" target="_blank">{url}</a>'
        display(HTML(link_html))

    dash_config = dict(
        MODE=mode,
        DASH_IN_IP=dash_in_ip,
        DASH_IN_PORT=dash_in_port,
    )

    return dash_config


def is_cloud():
    # TODO: find more elegant solution
    return _IS_CLOUD in os.environ and os.environ[_IS_CLOUD] == 'True'


def save_notebook(notebook_name):
    from ipylab import JupyterFrontEnd

    app = JupyterFrontEnd()
    app.commands.execute('docmanager:save')

    # subprocess.run([f"jupyter nbconvert --to pdf {notebook_name}.ipynb"], shell=True)


def print_action_title_description__low(title, description=""):
    run_on(lambda: print_action_title_description(title, description), log_level=LOG_LEVEL_LOW)


def print_action_title_description__medium(title, description=""):
    run_on(lambda: print_action_title_description(title, description), log_level=LOG_LEVEL_MEDIUM)


def print_action_title_description__high(title, description=""):
    run_on(lambda: print_action_title_description(title, description), log_level=LOG_LEVEL_HIGH)


def print_action_title_description(title, description=""):
    title = f'**{title}**'
    if description == "":
        printmd_low(f"{title}")
    else:
        description = f"`{description}`"
        printmd_low(f"{title}\r\n\r\n{description}")


def measure_print_action_title_description(title, description=""):
    print_action_title_description(title, description)

    return produce_measure(title)


def measure_print_action_title_description__low(title, description=""):
    print_action_title_description__low(title, description)

    return produce_measure_low(title)


def measure_print_action_title_description__medium(title, description=""):
    print_action_title_description__medium(title, description)

    return produce_measure_medium(title)


def measure_print_action_title_description__high(title, description=""):
    print_action_title_description__high(title, description)

    return produce_measure_high(title)


def get_public_ip(print_out=True):
    import requests
    response = requests.get('https://api64.ipify.org?format=json')
    data = response.json()
    public_ip = data['ip']

    if print_out:
        printmd(f"Public IP: **{public_ip}**")

    return public_ip


def start_notebook(notebook_name):
    import os
    from SRC.CORE._CONSTANTS import NOTEBOOK_NAME_KEY, LOG_FILE_NAME_KEY, EXPORT_FILE_NAME_KEY, PYTHON_STARTING_DIR_KEY
    from IPython import get_ipython

    set_parent_process()

    start_time = datetime.now()
    execution_type = "CLOUD" if is_cloud() else "MAC"
    title = notebook_name.replace("crypto_", "").upper()
    printmd(f'**!!!!!!!!!!!!!!!!!!!!!!!!! --- START {title} | {execution_type} | Time: {start_time.strftime("%H:%M:%S, %d-%m-%Y")} --- !!!!!!!!!!!!!!!!!!!!!!!!!**')
    os.environ[PYTHON_STARTING_DIR_KEY] = get_ipython().starting_dir
    os.environ[NOTEBOOK_NAME_KEY] = notebook_name
    os.environ[LOG_FILE_NAME_KEY] = get_notebook_log_file_name()
    os.environ[EXPORT_FILE_NAME_KEY] = get_notebook_export_file_name()
    clear_log()

    get_public_ip(print_out=True)
    get_processing_device(print_out=True)

    def finish_notebook():
        end_time = datetime.now()
        duration = end_time - start_time
        printmd(f'**!!!!!!!!!!!!!!!!!!!!!!!!! --- FINISH {title} | {execution_type} | Time: {end_time.strftime("%H:%M:%S, %d-%m-%Y")} | Duration: {duration} --- !!!!!!!!!!!!!!!!!!!!!!!!!**')

    return finish_notebook


def set_parent_process():
    os.environ[PARENT_PROCESS_ID_KEY] = str(os.getpid())


def unset_parent_process():
    if PARENT_PROCESS_ID_KEY in os.environ:
        del os.environ[PARENT_PROCESS_ID_KEY]


def is_parent_process():
    if PARENT_PROCESS_ID_KEY not in os.environ:
        return False

    parent_process_id_str = os.environ[PARENT_PROCESS_ID_KEY]
    current_process_id_str = str(os.getpid())
    is_parent_process = parent_process_id_str == current_process_id_str

    return is_parent_process


def produce_parent_process_delegate():
    is_parent = is_parent_process()
    if is_parent:
        return lambda: None

    set_parent_process()

    return unset_parent_process


def try_run_memory_monitor():
    monitor_thread = None
    if IS_MEMORY_MONITOR_ENABLED():
        monitor_thread = run_memory_monitor(60 * MEMORY_MONITOR_UPDATE_INTERVAL_MINS())

    return monitor_thread


def populate_string(pattern, simbols_count=70):
    str = "".join([pattern for i in range(int(simbols_count/len(pattern)))])

    return str


def wrap_printer(_lambda, file_path):
    try:
        use_file_printer = True if 'USE_FILE_PRINTER' not in os.environ else string_bool(os.environ['USE_FILE_PRINTER'])
        if not use_file_printer:
            return _lambda()

        original_stdout = sys.stdout
        with open(file_path, 'a') as file:
            sys.stdout = file

            result = _lambda()

        sys.stdout = original_stdout

        return result
    except IOError:
        pass


def _is_process_from_pycharm() -> bool:
    """Detect if any parent process name contains 'pycharm'."""
    try:
        import psutil
        p = psutil.Process()
        while p:
            name = p.name().lower()
            if "pycharm" in name or "idea" in name:
                return True
            p = p.parent()
    except Exception:
        pass
    return False


def is_running_under_pycharm_run() -> bool:
    """True when executed from PyCharm (Run) but not Debug."""
    return _is_process_from_pycharm() and sys.gettrace() is None


def is_running_under_pycharm_debugger() -> bool:
    """True when running under PyCharm Debug."""
    trace = sys.gettrace()
    if not trace:
        return False
    try:
        return any(
            "pydevd" in frame.filename.lower()
            for frame in sys._current_frames().values()
        )
    except Exception:
        return True


def is_running_under_jupyter() -> bool:
    """True when running inside any Jupyter Notebook / Lab / Kernel."""
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is None:
            return False
        return shell.__class__.__name__ != "TerminalInteractiveShell"
    except Exception:
        return False


def is_running_under_pycharm() -> bool:
    """True if running under PyCharm (Run, Debug, or PyCharm's Jupyter)."""
    return is_running_under_pycharm_run() or is_running_under_pycharm_debugger()



def is_running_under_test():
    return (
        'unittest' in sys.modules or
        'pytest' in sys.modules
    )


def is_running_in_notebook():
    try:
        from IPython import get_ipython
        return 'ipykernel' in str(type(get_ipython()))
    except (ImportError, AttributeError):
        return False


def get_caller_file_line(stack_level):
    # Get the current stack
    stack = inspect.stack()

    # The caller's frame is the second item in the stack (index 1)
    caller_frame = stack[stack_level]

    # Get the file name and line number from the caller's frame
    file_name = caller_frame.filename
    line_number = caller_frame.lineno

    return file_name, line_number


def hande_backprop_exception(err, input_s, pred_oh_prob, loss):
    import torch

    if 'LogSoftmaxBackward0' in str(err):
        print('!!!!EXPLODING GRADIENT!!!')
        try:
            if torch.isnan(input_s).any():
                print("------- input_s isnan -------")
                print(input_s)

            if torch.isinf(input_s).any():
                print("------- input_s isinf -------")
                print(input_s)

            if torch.isnan(pred_oh_prob).any():
                print("------- pred_oh_prob isnan -------")
                print(pred_oh_prob)

            if torch.isinf(pred_oh_prob).any():
                print("------- pred_oh_prob isinf -------")
                print(pred_oh_prob)

            if torch.isnan(loss).any():
                print("------- loss isnan -------")
                print(loss)

            if torch.isinf(loss).any():
                print("------- loss isinf -------")
                print(loss)
        except:
            pass

    raise


def print_call_stack():
    own_code_str_s = ["################### OWN CODE ###################"]
    execution_frame_s = get_own_code_excluded_name_execution_frame_s()
    for execution_frame in execution_frame_s:
        own_code_str_s.append(execution_frame)
    own_code_str_s.append("###############################################")
    DEBUG(*own_code_str_s)

    all_code_str_s = ["################### ALL CODE ###################"]
    execution_frame_s = get_excluded_name_execution_frame_s()
    for execution_frame in execution_frame_s:
        all_code_str_s.append(execution_frame)
    all_code_str_s.append("###############################################")
    DEBUG(*all_code_str_s)


default_exclude_name_s = ['wrapper', 'lambda', 'tryall_delegate', 'get_own_code_excluded_name_execution_frame_s', 'print_call_stack']


def get_excluded_name_execution_frame_s(exclude_name_s=[]):
    exclude_name_s = [*exclude_name_s, *default_exclude_name_s]
    execution_frame_s = [frame for frame in traceback.extract_stack()]
    excluded_frame_s = [execution_frame for execution_frame in execution_frame_s if not any([exclude_name in execution_frame.name for exclude_name in exclude_name_s])]

    return excluded_frame_s


def get_own_code_excluded_name_execution_frame_s(exclude_name_s=[]):
    exclude_name_s = [*exclude_name_s, *default_exclude_name_s]
    execution_frame_s = [frame for frame in traceback.extract_stack() if own_code_predicate(frame)]
    excluded_frame_s = [execution_frame for execution_frame in execution_frame_s if not any([exclude_name in execution_frame.name for exclude_name in exclude_name_s])]

    return excluded_frame_s


def get_execution_frame_present(execution_frame):
    file_name = execution_frame.filename

    if str(Path('SRC')) in execution_frame.filename:
        file_name = Path(f"{str(Path('/SRC/'))}{execution_frame.filename.split(str(Path('/SRC/')))[1]}")

    if str(Path('WEBAPP')) in execution_frame.filename:
        file_name = f"{str(Path('/WEBAPP/'))}{execution_frame.filename.split(str(Path('/WEBAPP/')))[1]}"

    execution_caller_present = f"{file_name}: {execution_frame.lineno} | {execution_frame.line}"

    return execution_caller_present


def get_execution_caller_present(depth=3):
    try:
        execution_frame = [frame for frame in traceback.extract_stack() if own_code_predicate(frame)][-depth]
        execution_frame_present = get_execution_frame_present(execution_frame)

        return execution_frame_present
    except:
        return 'notebook'


def own_code_predicate(frame):
    return 'SRC' in frame.filename or 'WEBAPP' in frame.filename or 'CRYPTO_BOT' in frame.filename


def get_own_code_stack_present_s():
    return get_stack_present_s(lambda frame: own_code_predicate(frame))


def get_stack_present_s(_predicate=lambda frame: True):
    execution_frame_s = [frame for frame in traceback.extract_stack() if _predicate(frame)]
    execution_frame_present_s = [get_execution_frame_present(ef) for ef in execution_frame_s]

    return execution_frame_present_s


@lru_cache(maxsize=None)
def get_log_file_name():
    from SRC.LIBRARIES.time_utils import kiev_now
    pid = os.getpid()
    log_file_name = f"logs_{pid}_{kiev_now().isoformat()}.txt".replace(":", "-")

    return log_file_name


@lru_cache(maxsize=None)
def get_log_file_full_path():
    log_file_name = get_log_file_name()

    log_folder = f"{project_root_dir()}/logs"
    os.makedirs(log_folder, exist_ok=True)
    log_file_full_path = os.path.join(log_folder, log_file_name)

    return log_file_full_path


class TZFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, tz=None):
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.tz = tz or pytz.UTC

    def format(self, record):
        if record.exc_info and record.levelno >= logging.EXCEPTION:
            record.levelname = "EXCEPTION"
        return super().format(record)

    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, self.tz)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.isoformat()


@lru_cache(maxsize=None)
def get_or_create_logger(name='crypto_logger'):
    log_file_full_path = get_log_file_full_path()

    if isinstance(logging.getLevelName("EXCEPTION"), str):
        def exception(self, msg, *args, **kwargs):
            if self.isEnabledFor(logging.EXCEPTION):
                self._log(logging.EXCEPTION, msg, args, **kwargs)

        logging.Logger.exception = exception
        logging.addLevelName(logging.EXCEPTION, "EXCEPTION")

    if isinstance(logging.getLevelName("NOTICE"), str):
        def notice(self, msg, *args, **kwargs):
            if self.isEnabledFor(logging.NOTICE):
                self._log(logging.NOTICE, msg, args, **kwargs)

        logging.Logger.notice = notice
        logging.addLevelName(logging.NOTICE, "NOTICE")

    if isinstance(logging.getLevelName("CONSOLE"), str):
        def console(self, msg, *args, **kwargs):
            if self.isEnabledFor(logging.CONSOLE):
                self._log(logging.CONSOLE, msg, args, **kwargs)

        logging.Logger.console = console
        logging.addLevelName(logging.CONSOLE, "CONSOLE")

    timezone = KIEV_TZ
    formatter = TZFormatter(
        fmt="%(levelname)s [%(process)d:%(threadName)s|%(asctime)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %Z",
        tz=timezone
    )

    file_handler = logging.FileHandler(Path(log_file_full_path))
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logging.basicConfig(
        level=logging.CONSOLE,
        handlers=[file_handler, stream_handler]
    )

    logger = logging.getLogger(name)

    return logger


frames_s__deque = deque([[]], maxlen=1)
log_file_path = 'killme_logs.txt'


def _SPLITTED(*msg_s, log_level, _func, timeout=20):
    from SRC.LIBRARIES.time_utils import utc_now, kiev_now
    from SRC.LIBRARIES.new_utils import append_file, env_int

    app_log_level = env_int('LOG_LEVEL', logging.NOTSET)
    if log_level < app_log_level:
        # print(f"_SPLITTED [{log_level}]: {len(msg_s)}")
        return

    frames_s = frames_s__deque[-1]
    measure_wrap = produce_measure()
    duration = None
    log_end = False
    try:
        frames_s.append(traceback.format_stack())
        now_start = utc_now()

        with FileLock(lock_file=_LOGGING_PROCESS_LOCK_FILE_PATH(), timeout=timeout):
            now_entered = utc_now()
            if (now_entered - now_start).total_seconds() > 2:
                log_end = True
                frames_s__deque.append([])
            else:
                frames_s__deque.append(frames_s)

            measure = produce_measure()

            logger = get_or_create_logger('crypto_logger_splitted')

            logger.handlers.clear()

            logger.propagate = False
            log_file_full_path = get_log_file_full_path()

            timezone = KIEV_TZ
            formatter = TZFormatter(
                fmt='-------------------------------------------------------'
                      '\r\n%(levelname)s [%(process)d:%(threadName)s||%(filename)s:%(lineno)d > %(funcName)s||%(asctime)s]: \r\n%(message)s'
                      '\r\n-------------------------------------------------------',
                datefmt="%Y-%m-%d %H:%M:%S %Z",
                tz=timezone
            )

            file_handler = logging.FileHandler(log_file_full_path)
            stream_handler = logging.StreamHandler()

            file_handler.setFormatter(formatter)
            stream_handler.setFormatter(formatter)

            logger.addHandler(file_handler)
            logger.addHandler(stream_handler)

            _func(*msg_s, logger=logger)

            logger.removeHandler(file_handler)
            logger.removeHandler(stream_handler)
            logger.propagate = True

            file_handler.close()
            stream_handler.close()

            duration = measure()
    except TimeoutError as error:
        print("Another logging process is currently writing to the log file. Skipping log entry to avoid conflicts.")
        for msg in msg_s:
            print(msg)
    finally:
        if log_end:
            frame_logs = [
                f"LOOK AT SLOW LOGS [{str(kiev_now())}]: {log_file_path}"
            ]
            print('\r\n'.join(frame_logs))

            duration_wrap = measure_wrap()
            divider = f"========================={str(kiev_now())}========================="
            log_title = f"LOG RUNS [WRAP DURATION: {duration_wrap} | DURATION: {duration} | TIMEOUT: {timeout} | MESSAGES: {len(msg_s)}"

            for frames in frames_s[::-1]:
                frame_logs.append(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                for frame in frames:
                    frame_logs.append(str(frame))

            logs = [
                divider,
                log_title,
                *frame_logs,
                divider,
            ]
            append_file(log_file_path, *logs)


def reopen_filehandler_if_missing(logger):
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            log_path = handler.baseFilename

            if not os.path.exists(log_path):
                logger.removeHandler(handler)
                handler.close()

                new_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
                new_handler.setFormatter(handler.formatter)
                new_handler.setLevel(handler.level)

                logger.addHandler(new_handler)
                return True
    return False


def make_log_record(*msg_s, stacklevel, loglevel, logger):
    from SRC.LIBRARIES.new_utils import env_int

    logger = logger if logger else get_or_create_logger()

    reopen_filehandler_if_missing(logging.getLogger())

    log_text_format = " | ".join([str(msg) for msg in msg_s])

    frame = inspect.stack()[stacklevel]

    exc_info = sys.exc_info()
    if exc_info == (None, None, None) or loglevel <= logging.ERROR:
        exc_info = None

    app_log_level = env_int('LOG_LEVEL', logging.NOTSET)
    if loglevel >= app_log_level:
        record = logger.makeRecord(
            logger.name,
            loglevel,
            fn=frame.filename,
            lno=frame.lineno,
            msg=log_text_format,
            args=(),
            exc_info=exc_info,
            func=frame.function,
            extra=None
        )
        logger.handle(record)


def EXCEPTION(*msg_s, stacklevel=0, logger=None):
    make_log_record(*msg_s, stacklevel=2+stacklevel, loglevel=logging.EXCEPTION, logger=logger)


def EXCEPTION_SPLITTED(*msg_s, stacklevel=0, timeout=20):
    log_text_format = "\r\n".join([str(msg) for msg in msg_s])
    _SPLITTED(*msg_s, log_level=logging.EXCEPTION, _func=lambda *msg_s, logger: EXCEPTION(log_text_format, stacklevel=stacklevel+3, logger=logger), timeout=timeout)


def ERROR(*msg_s, stacklevel=0, logger=None):
    make_log_record(*msg_s, stacklevel=2+stacklevel, loglevel=logging.ERROR, logger=logger)


def ERROR_SPLITTED(*msg_s, stacklevel=0, timeout=20):
    log_text_format = "\r\n".join([str(msg) for msg in msg_s])
    _SPLITTED(*msg_s, log_level=logging.ERROR, _func=lambda *msg_s, logger: ERROR(log_text_format, stacklevel=stacklevel+3, logger=logger), timeout=timeout)


def CONSOLE(*msg_s, stacklevel=0, logger=None):
    make_log_record(*msg_s, stacklevel=2 + stacklevel, loglevel=logging.CONSOLE, logger=logger)


def CONSOLE_SPLITTED(*msg_s, stacklevel=0, timeout=20):
    log_text_format = "\r\n".join([str(msg) for msg in msg_s])
    _SPLITTED(*msg_s, log_level=logging.CONSOLE, _func=lambda *msg_s, logger: CONSOLE(log_text_format, stacklevel=stacklevel+3, logger=logger), timeout=timeout)


def NOTICE(*msg_s, stacklevel=0, logger=None):
    make_log_record(*msg_s, stacklevel=2 + stacklevel, loglevel=logging.NOTICE, logger=logger)


def NOTICE_SPLITTED(*msg_s, stacklevel=0, timeout=20):
    log_text_format = "\r\n".join([str(msg) for msg in msg_s])
    _SPLITTED(*msg_s, log_level=logging.NOTICE, _func=lambda *msg_s, logger: NOTICE(log_text_format, stacklevel=stacklevel+3, logger=logger), timeout=timeout)


def DEBUG(*msg_s, stacklevel=0, logger=None):
    make_log_record(*msg_s, stacklevel=2 + stacklevel, loglevel=logging.DEBUG, logger=logger)


def DEBUG_SPLITTED(*msg_s, stacklevel=0, timeout=20):
    log_text_format = "\r\n".join([str(msg) for msg in msg_s])
    _SPLITTED(*msg_s, log_level=logging.DEBUG, _func=lambda *msg_s, logger: DEBUG(log_text_format, stacklevel=stacklevel+3, logger=logger), timeout=timeout)


def unset_loglevel():
    for key, val in os.environ.items():
        if key.startswith('IS_') and key.endswith('_LOGLEVEL_SET'):
            os.environ[key] = 'False'


def SET_NOTICE_LOGLEVEL():
    unset_loglevel()

    os.environ['IS_NOTICE_LOGLEVEL_SET'] = 'True'
    os.environ['LOG_LEVEL'] = f'{logging.NOTICE}'
    HANDLE_LOG_LEVEL()


def SET_CONSOLE_LOGLEVEL():
    unset_loglevel()

    os.environ['IS_CONSOLE_LOGLEVEL_SET'] = 'True'
    os.environ['LOG_LEVEL'] = f'{logging.CONSOLE}'
    HANDLE_LOG_LEVEL()


def SET_DEBUG_LOGLEVEL():
    unset_loglevel()

    os.environ['IS_DEBUG_LOGLEVEL_SET'] = 'True'
    os.environ['LOG_LEVEL'] = f'{logging.DEBUG}'
    HANDLE_LOG_LEVEL()


def SET_BINANCE_PROXY():
    os.environ[_USE_PROXY_CLIENT] = 'True'
    print(f"os.environ['{_USE_PROXY_CLIENT}'] = {os.environ[_USE_PROXY_CLIENT]}")


def SET_SYMBOL(symbol):
    from SRC.LIBRARIES.new_utils import _symbol_slash, _symbol_join, _symbol_dash

    os.environ[_SYMBOL_JOIN] = _symbol_join(symbol)
    os.environ[_SYMBOL_SLASH] = _symbol_slash(symbol)
    os.environ[_SYMBOL_DASH] = _symbol_dash(symbol)

    os.environ[_COIN_ASSET] = os.environ[_SYMBOL_SLASH].split("/")[0]
    os.environ[_STABLE_ASSET] = os.environ[_SYMBOL_SLASH].split("/")[1]

    DEBUG(f"JOIN: {os.environ[_SYMBOL_JOIN]} | SLASH: {os.environ[_SYMBOL_SLASH]} | DASH: {os.environ[_SYMBOL_DASH]}")


def SET_BINANCE_PROD():
    os.environ[_IS_BINANCE_PROD] = 'True'
    print(f"os.environ['{_IS_BINANCE_PROD}'] = {os.environ[_IS_BINANCE_PROD]}")
    print(f"PYTHON-BINANCE VERSION: {version('python-binance')}")


def SET_REDIS_CLOUD():
    os.environ[_IS_REDIS_CLOUD] = 'True'
    print(f"os.environ['{_IS_REDIS_CLOUD}'] = {os.environ[_IS_REDIS_CLOUD]}")


def SET_ISCLOUD():
    os.environ[_IS_CLOUD] = f'True'
    print(f"os.environ['{_IS_CLOUD}'] = {os.environ[_IS_CLOUD]}")


def SET_WEBDOCK_ISCLOUD():
    if get_public_ip(print_out=True) == '2a0f:f01:206:5c3::':
        SET_ISCLOUD()


def SET_CLOUD_IF_NOT_MAC():
    if sys.platform != "darwin":
        SET_ISCLOUD()


def IS_NOTICE():
    from SRC.LIBRARIES.new_utils import check_env_true

    return check_env_true('IS_NOTICE_LOGLEVEL_SET', default_val=False)


def IS_CONSOLE():
    from SRC.LIBRARIES.new_utils import check_env_true

    return check_env_true('IS_CONSOLE_LOGLEVEL_SET', default_val=False)


def IS_DEBUG():
    from SRC.LIBRARIES.new_utils import check_env_true

    return check_env_true('IS_DEBUG_LOGLEVEL_SET', default_val=False)


def HANDLE_FLAGS(required_flag_s=[]):
    from SRC.CORE._CONFIGS import WEBDOCK_AKCRYPTOBUFF_HOST
    from SRC.CORE._CONFIGS import get_config

    parser = argparse.ArgumentParser(description="A script that accepts parameters.")
    parser.add_argument('--notice', action='store_true', help='Set NOTICE log level', required=False)
    parser.add_argument('--console', action='store_true', help='Set CONSOLE log level', required=False)
    parser.add_argument('--debug', action='store_true', help='Set DEBUGs log level', required=False)
    parser.add_argument('--located', action='store_true', help='Set LOCATED logging', required=False)
    parser.add_argument('--redis_cloud', action='store_true', help='Set Redis PROD CLIENT', required='redis_cloud' in required_flag_s)
    parser.add_argument('--binance_emulate_algo', action='store_true', help='Set Binance EMULATE ALGO flow', required=False)
    parser.add_argument('--binance_proxy', action='store_true', help='Set Binance PROXY CLIENT', required='binance_proxy' in required_flag_s)
    parser.add_argument('--binance_prod', action='store_true', help='Set Binance PROD CLIENT', required='binance_prod' in required_flag_s)
    parser.add_argument('--binance_prod_test', action='store_true', help='Set Binance PROD-TEST CLIENT', required='binance_prod_test' in required_flag_s)
    parser.add_argument('--port', type=int, help='Port number', required='port' in required_flag_s)
    parser.add_argument('--workers', type=int, help='Num workers', required='workers' in required_flag_s)
    parser.add_argument('--spark_name', type=str, help='Spark instance name', required='spark_name' in required_flag_s)
    parser.add_argument('--delist', action='store_true', help='Run margin asset delist schedule watcher', required='delist' in required_flag_s)
    parser.add_argument('--incl', type=str, nargs='+', help='Include symbols', required=False)
    parser.add_argument('--red', action='store_true', help='Remove existing data', required=False)
    parser.add_argument('--cloud', action='store_true', help='Cloud instance', required=False)
    parser.add_argument('--callers_app_host', type=str, help='CallersApp host', required=False)
    parser.add_argument('--predictions_service_app_host', type=str, help='CallersApp host', required=False)
    parser.add_argument(f'--automation_type', type=str, help='Automation type [AUTOTRADING/BACKTESTING]', required='automation_type' in required_flag_s)
    args = parser.parse_args()

    is_console_log_level_set = args.console
    if is_console_log_level_set:
        SET_CONSOLE_LOGLEVEL()

    is_notice_log_level_set = args.notice
    if is_notice_log_level_set:
        SET_NOTICE_LOGLEVEL()

    is_debug_log_level_set = args.debug
    if is_debug_log_level_set:
        SET_DEBUG_LOGLEVEL()

    cloud = args.cloud
    if cloud:
        args.callers_app_host = args.callers_app_host or WEBDOCK_AKCRYPTOBUFF_HOST
        args.predictions_service_app_host = args.predictions_service_app_host or WEBDOCK_AKCRYPTOBUFF_HOST
        args.redis_cloud = True
        SET_ISCLOUD()

    is_binance_proxy = args.binance_proxy
    if is_binance_proxy:
        SET_BINANCE_PROXY()

    is_binance_client_real = args.binance_prod
    if is_binance_client_real:
        SET_BINANCE_PROD()

    redis_cloud = args.redis_cloud
    if redis_cloud:
        SET_REDIS_CLOUD()

    binance_emulate_algo = args.binance_emulate_algo or get_config('dashboard_signal_consumer_executor.futures_isolated_trader.binance_emulate_algo_flow', default=False)
    if binance_emulate_algo:
        os.environ['IS_EMULATE_ALGO_FLOW'] = 'True'
        print(f"os.environ['IS_EMULATE_ALGO_FLOW'] = {os.environ['IS_EMULATE_ALGO_FLOW']}")

    port = args.port
    if port:
        os.environ['PORT'] = f'{port}'
        print(f"os.environ['PORT'] = {os.environ['PORT']}")

    workers = args.workers
    if workers:
        os.environ['WORKERS'] = f'{workers}'
        print(f"os.environ['WORKERS'] = {os.environ['WORKERS']}")

    spark_name = args.spark_name
    if spark_name:
        os.environ['SPARK_NAME'] = f'{spark_name}'
        print(f"os.environ['SPARK_NAME'] = {os.environ['SPARK_NAME']}")

    remove_existing_data = args.red
    if remove_existing_data:
        os.environ['REMOVE_EXISTING_DATA'] = f'True'
        print(f"os.environ['REMOVE_EXISTING_DATA'] = {os.environ['REMOVE_EXISTING_DATA']}")

    run_delist_schedule_watcher = args.delist
    if run_delist_schedule_watcher:
        os.environ['RUN_DELIST_SCHEDULE_WATCHER'] = 'True'
        print(f"os.environ['RUN_DELIST_SCHEDULE_WATCHER'] = {os.environ['RUN_DELIST_SCHEDULE_WATCHER']}")

    include_symbol_s = args.incl if args.incl else []
    if len(include_symbol_s) > 0:
        os.environ['INCLUDE_SYMBOLS'] = "|".join(include_symbol_s)
        print(f"os.environ['INCLUDE_SYMBOLS'] = {os.environ['INCLUDE_SYMBOLS']}")

    automation_type = args.automation_type
    if automation_type:
        os.environ[_AUTOMATION_TYPE] = f"{automation_type}"
        print(f"os.environ['{_AUTOMATION_TYPE}'] = {os.environ[_AUTOMATION_TYPE]}")

    callers_app_host = args.callers_app_host
    if callers_app_host:
        os.environ[_CALLERS_APP_HOST] = f"{callers_app_host}"
        print(f"os.environ['{_CALLERS_APP_HOST}'] = {os.environ[_CALLERS_APP_HOST]}")

    predictions_service_app_host = args.predictions_service_app_host
    if predictions_service_app_host:
        os.environ[_PREDICTION_SERVICE_APP_HOST] = f"{predictions_service_app_host}"
        print(f"os.environ['{_PREDICTION_SERVICE_APP_HOST}'] = {os.environ[_PREDICTION_SERVICE_APP_HOST]}")

    caller_name = [frame for frame in traceback.extract_stack() if own_code_predicate(frame)][0].filename.split('/')[-1]
    add_logfile_execution_mapping(caller_name)


def add_logfile_execution_mapping(caller_name):
    from SRC.LIBRARIES.new_utils import append_file

    log_file_name = get_log_file_name()
    prefix = 'CLOUD' if is_cloud() else 'MAC'
    append_file(f"{project_root_dir()}/logs/_{prefix}_CALLER_LOG_FILE_MAP.txt", f"{caller_name} | {log_file_name}")

    return log_file_name


def print_own_callers():
    caller_s = get_own_code_stack_present_s()
    DEBUG(f"ACTUAL CALLERS:")
    for caller in caller_s:
        DEBUG(caller)


def print_location(message: str = ""):
    frame = inspect.currentframe().f_back
    file_path = frame.f_code.co_filename
    line_number = frame.f_lineno
    file_name = os.path.basename(file_path)

    if message:
        print(f"~~~ {file_name}:{line_number}\r\n{message}")
    else:
        print(f"~~~ {file_name}:{line_number}")


def HANDLE_LOG_LEVEL():
    delete_died_log_files()

    from SRC.LIBRARIES.new_utils import check_env_true

    project_root_dir_message = f"PROJECT ROOT DIR: {project_root_dir()}"
    if check_env_true('IS_CONSOLE_LOGLEVEL_SET'):
        get_or_create_logger().setLevel(logging.CONSOLE)
        logging.basicConfig(level=logging.CONSOLE)

        CONSOLE_SPLITTED(f"{project_root_dir_message}", stacklevel=1)

    if check_env_true('IS_NOTICE_LOGLEVEL_SET'):
        get_or_create_logger().setLevel(logging.NOTICE)
        logging.basicConfig(level=logging.NOTICE)

        NOTICE_SPLITTED(f"{project_root_dir_message}", stacklevel=1)

    if check_env_true('IS_DEBUG_LOGLEVEL_SET'):
        get_or_create_logger().setLevel(logging.DEBUG)
        logging.basicConfig(level=logging.DEBUG)

        DEBUG_SPLITTED(f"{project_root_dir_message}", stacklevel=1)

    if not check_env_true('IS_CONSOLE_LOGLEVEL_SET') and not check_env_true('IS_NOTICE_LOGLEVEL_SET') and not check_env_true('IS_DEBUG_LOGLEVEL_SET'):
        SET_CONSOLE_LOGLEVEL()


def log_mock_calls(func, name=None, depth=3, is_splitted_logs=True, shorten_return=False, log_level='NOTICE'):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        _log_func = lambda *msg_s: None

        if log_level == 'CONSOLE':
            _log_func = CONSOLE_SPLITTED if is_splitted_logs else CONSOLE

        if log_level == 'NOTICE':
            _log_func = NOTICE_SPLITTED if is_splitted_logs else NOTICE

        if log_level == 'DEBUG':
            _log_func = DEBUG_SPLITTED if is_splitted_logs else DEBUG

        returned_result = ''  # ← REQUIRED
        execution_caller = get_execution_caller_present(depth=depth)
        try:
            result = func(*args, **kwargs)
            if shorten_return:
                returned_result = f"\r\nreturned={str(result)[:shorten_return]}..." if result is not None else ''
            else:
                returned_result = f"\r\nreturned={result}" if result is not None else ''

        except BaseException as ex:
            returned_result = f"\r\nEXCEPTION={str(ex)}"

            raise
        finally:
            filtered_args = args[1:] if args and hasattr(args[0], func.__name__) else args
            _log_func(f"{name} [{execution_caller}]\r\nargs={filtered_args}, kwargs={kwargs}{returned_result}")

        return result

    return wrapper


def decor_log_mock_calls(name=None, depth=3, is_splitted_logs=True, shorten_return=False, log_level='NOTICE'):
    def decorator(func):
        return log_mock_calls(func, name, depth, is_splitted_logs, shorten_return, log_level)

    return decorator


@contextmanager
def log_context(_log_func=CONSOLE, stacklevel=0, timeout=20) -> []:
    log_s = []
    try:
        yield log_s
    finally:
        if len(log_s) > 0:
            _override_log_func = next((x for x in log_s if inspect.isfunction(x)), None)
            if _override_log_func:
                log_s = [x for x in log_s if x is not _override_log_func]
                if '_SPLITTED' in _override_log_func.__name__:
                    _override_log_func(*log_s, stacklevel=stacklevel+2, timeout=timeout)
                else:
                    _override_log_func(*log_s, stacklevel=stacklevel+2)
            else:
                if '_SPLITTED' in _log_func.__name__:
                    _log_func(*log_s, stacklevel=stacklevel+2, timeout=timeout)
                else:
                    _log_func(*log_s, stacklevel=stacklevel+2)



def is_autotrading():
    if _AUTOMATION_TYPE in os.environ:
        return os.environ[_AUTOMATION_TYPE] == _AUTOTRADING

    return False


def is_backtesting():
    return not is_autotrading()


def delete_died_log_files(age_timedelta='1D'):
    from SRC.LIBRARIES.new_utils import is_recently_updated

    folder_path = f"{project_root_dir()}/logs"

    all_files = [file for file in Path(folder_path).iterdir() if file.is_file() and file.suffix == '.txt' and 'CALLER_LOG_FILE_MAP' not in str(file)]
    sorted_files = sorted(all_files, key=lambda x: x.stat().st_mtime, reverse=False)

    log_files = [file.name for file in sorted_files]

    for log_file in log_files:
        file_path = f"{folder_path}/{log_file}"
        is_running = is_recently_updated(file_path, age_timedelta)
        if not is_running:
            os.remove(file_path)
            print(f"REMOVED: {file_path}")


def load_shell_constants(path):
    """
    Sources a .sh file and returns its exported env variables.
    """
    command = f'. "{path}" >/dev/null 2>&1 && env'
    result = subprocess.run(
        ["bash", "-c", command],
        capture_output=True,
        text=True,
        check=True,
    )

    for line in result.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k, v)


def setup__flags_to_envd(flags):
    flagd = {}

    try:
        for flag in flags.split("|"):
            if not flag.strip():
                continue

            if ':' in flag:
                key, value = flag.split(":", 1)
                # os.environ[key] = str(value)
                flagd[key] = str(value)
            else:
                key = flag
                # os.environ[key] = str(True)
                flagd[key] = str(True)
    except Exception as ex:
        DEBUG_SPLITTED([
            f"NO FLAGS SET",
            str(flags),
            str(ex)
        ])
    finally:
        return flagd


def TEST__setup__env_vars__flags():
    from SRC.LIBRARIES.new_utils import check_env_true, env_string

    flags = "DEBUG:False|FORCE:True|RANDOM:1.0035-1.035_1-15-1"
    setup__flags_to_envd(flags)

    print(check_env_true("DEBUG"))
    print(check_env_true('FORCE'))
    print(env_string('RANDOM'))


def TEST__log_context():
    SET_DEBUG_LOGLEVEL()

    def test2(ext_log_s=None):
        with log_context(CONSOLE_SPLITTED) as log_s:
            log_s = log_s if ext_log_s is None else ext_log_s

            log_s.append("test2")

    def test1(ext_log_s=None):
        with log_context(CONSOLE_SPLITTED) as log_s:
            log_s = log_s if ext_log_s is None else ext_log_s

            log_s.append("test1")
            test2(ext_log_s=log_s)

    test1()


if __name__ == "__main__":
    TEST__log_context()

    SET_SYMBOL("BTCUSDT")
    SET_SYMBOL("BTC_USDT")
    SET_SYMBOL("BTC/USDT")

    from SRC.WEBAPP.TESTS.test_cross_monitoring_trader_Dbinance import set_symbol_assets
    from SRC.LIBRARIES.new_utils import append_file, env_int, write_file

    set_symbol_assets("BTCUSDT")

    SET_NOTICE_LOGLEVEL()
    SET_DEBUG_LOGLEVEL()

    caller_file_name = get_current_notebook_name()
    log_file_path = get_log_file_full_path()
    log_file_name = get_log_file_name()
    prefix = 'CLOUD' if is_cloud() else 'MAC'
    append_file(f"{project_root_dir()}/logs/_{prefix}_CALLER_LOG_FILE_MAP.txt", f"{caller_file_name} | {log_file_name}")

    NOTICE("hello world")
    NOTICE_SPLITTED("hello world splitted located")